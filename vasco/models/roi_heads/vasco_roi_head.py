# projects/vasco/models/roi_heads/vasco_roi_head.py
from mmdet.registry import MODELS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.structures.bbox import bbox2roi
from projects.vasco.models.modules.temporal_align_aggregator import TemporalAlignAggregator
from typing import Dict, Optional, List
import torch
import torch.nn.functional as F
import numpy as np
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData, PixelData


@MODELS.register_module()
class VascoRoIHead(StandardRoIHead):
    """StandardRoIHead 增强版：集成TemporalAlignAggregator时序RoI聚合、动作/方向多任务头、语义分割头。"""
    def __init__(self, 
                temporal_aggregator: Optional[Dict] = None, 
                action_dir_head: Optional[Dict] = None,
                semantic_head: Optional[Dict] = None,
                **kwargs):
        super().__init__(**kwargs)
        self.temporal_aggregator = MODELS.build(temporal_aggregator) if temporal_aggregator else None
        self._temporal_ctx = None
        self._last_temporal_losses = None
        self.action_dir_head = MODELS.build(action_dir_head) if action_dir_head else None
        self.semantic_head = MODELS.build(semantic_head) if semantic_head else None

    def set_temporal_ctx(self, ctx: Dict):
        """由检测器在提取特征后调用，将时序上下文传入 RoIHead。"""
        self._temporal_ctx = ctx

    def _bbox_forward(self, x, rois):
        bbox_feats = None
        self._last_temporal_losses = None

        ctx = getattr(self, '_temporal_ctx', None)
        if self.temporal_aggregator is not None and ctx is not None:
            # 优先使用“每帧 FPN”序列（List[T][L]）
            feats2d_seq = ctx.get('fpn_seq', None)

            out = self.temporal_aggregator(
                rois=rois,
                roi_extractor=self.bbox_roi_extractor,
                feats2d_seq=feats2d_seq,                 # 首选路径：每帧 FPN
                feat3d=ctx.get('feat3d', None),          # 回退路径：3D 主干输出（内部1x1→256）
                roi_feats_last2d=None
            )
            if isinstance(out, tuple):
                bbox_feats, agg_losses = out
                self._last_temporal_losses = agg_losses
            else:
                bbox_feats = out

        if bbox_feats is None:
            bbox_feats = self.bbox_roi_extractor(x, rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        return dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

    def _loss_action_dir(self, x, rpn_results_list, data_samples):
        """计算末帧上GT RoI的动作/方向分类损失（仅在训练时调用）。"""
        if self.action_dir_head is None:
            # 如果没有多任务头，则返回0损失占位
            zero = x[0].new_tensor(0.)
            return dict(loss_action=zero, loss_dir=zero, action_acc=zero, dir_acc=zero)
        # 汇总每张图片的GT RoI用于提取
        rois_per_img = []
        gt_actions_all, gt_dirs_all = [], []
        total_rois = 0
        for data_sample in data_samples:
            gt_inst = data_sample.gt_instances
            gt_bboxes = gt_inst.bboxes.tensor if hasattr(gt_inst.bboxes, 'tensor') else gt_inst.bboxes
            if gt_bboxes.numel() == 0:
                # 无GT RoI时填充空
                rois_per_img.append(gt_bboxes.new_zeros((0, 4)))
                continue
            rois_per_img.append(gt_bboxes.to(x[0].device))
            gt_actions_all.append(gt_inst.actions.to(x[0].device))
            gt_dirs_all.append(gt_inst.dirs.to(x[0].device))
            total_rois += gt_bboxes.size(0)
        if total_rois == 0:
            zero = x[0].new_tensor(0.)
            return dict(loss_action=zero, loss_dir=zero, action_acc=zero, dir_acc=zero)
        # 将每张图片的GT RoI拼接成统一的 rois Tensor (R, 5)
        rois = bbox2roi(rois_per_img)  # 加上批次索引
        gt_actions = torch.cat(gt_actions_all, dim=0)
        gt_dirs = torch.cat(gt_dirs_all, dim=0)

        # 提取RoI特征（考虑时序聚合）
        roi_feats = None
        ctx = getattr(self, '_temporal_ctx', None)
        if self.temporal_aggregator and ctx and 'feat3d' in ctx:
            result = self.temporal_aggregator(
                rois=rois, roi_extractor=self.bbox_roi_extractor, feat3d=ctx['feat3d']
            )
            if isinstance(result, tuple):
                roi_feats = result[0]
                # 聚合损失存于 temporal_aggregator.last_loss_dict
            else:
                roi_feats = result
        if roi_feats is None:
            roi_feats = self.bbox_roi_extractor(x, rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        # 计算动作和方向预测以及损失
        logits_action, logits_dir = self.action_dir_head(roi_feats)
        return self.action_dir_head.loss(logits_action, logits_dir, gt_actions, gt_dirs)
    
    # ---- Stuff semantic loss on last-frame P2 ----
    def _loss_semantic(self, x, data_samples) -> Dict[str, torch.Tensor]:
        if self.semantic_head is None:
            return {}
        p2 = x[0]  # (B,C,h,w)
        gts = []
        for ds in data_samples:
            g = getattr(ds, 'gt_sem_seg', None)
            if g is None:
                gts.append(None)
                continue
            if hasattr(g, 'sem_seg'):
                seg = g.sem_seg
            elif hasattr(g, 'data'):
                seg = g.data
            else:
                seg = g
            gts.append(seg)
        if not all(s is not None for s in gts):
            return {}

        tgt = []
        for s in gts:
            s = torch.as_tensor(s, device=p2.device)
            if s.ndim == 2:
                s = s.unsqueeze(0)
            tgt.append(s)
        tgt = torch.stack(tgt, dim=0)  # (B,1,H,W)
        tgt_ds = F.interpolate(tgt.float(), size=p2.shape[-2:], mode='nearest').long().squeeze(1)

        logits = self.semantic_head(p2)  # (B,7,h,w) for example
        ce, dice = self.semantic_head.loss(logits, tgt_ds)
        return dict(loss_sem_ce=ce, loss_sem_dice=dice)

    def loss(self, x, rpn_results_list, data_samples, **kwargs):
        losses = super().loss(x, rpn_results_list, data_samples, **kwargs)
        # 并入聚合器一致性损失
        if isinstance(self._last_temporal_losses, dict) and 'loss_roi_consistency' in self._last_temporal_losses:
            losses['loss_roi_consistency'] = self._last_temporal_losses['loss_roi_consistency']
        
        losses.update(self._loss_action_dir(x, rpn_results_list, data_samples))
        losses.update(self._loss_semantic(x, data_samples))
        return losses

    def predict(self, x, rpn_results_list, data_samples, rescale: bool = False):
        """在测试/推理时调用：执行检测并输出动作/方向和语义预测。"""
        # 1) 获得基础检测结果（不缩放坐标，在resized-cropped空间）
        det_results = super().predict(x, rpn_results_list, data_samples, rescale=False)
        device = x[0].device if isinstance(x, (list, tuple)) else x.device
        # 2) 准备所有图片的检测 RoI，用于多任务头
        all_rois = []
        for i, result in enumerate(det_results):
            inst = result.pred_instances if hasattr(result, 'pred_instances') else result
            if not hasattr(inst, 'bboxes') or inst.bboxes.numel() == 0:
                continue
            # 为每个检测框添加batch索引前缀
            n = inst.bboxes.size(0)
            batch_idx = torch.full((n, 1), i, dtype=inst.bboxes.dtype, device=device)
            all_rois.append(torch.cat([batch_idx, inst.bboxes.to(device)], dim=1))
        rois = torch.cat(all_rois, dim=0) if all_rois else None

        # 3) 提取每个预测 RoI 的特征（如有时序聚合则利用之）
        roi_feats = None
        if rois is not None and rois.numel() > 0:
            ctx = getattr(self, '_temporal_ctx', None)
            if self.temporal_aggregator and ctx and 'feat3d' in ctx:
                # 确保所用特征层数与 RoIExtractor 对应
                feat3d_list = ctx['feat3d']
                num_levels = len(self.bbox_roi_extractor.roi_layers)
                feat3d_list = feat3d_list[:num_levels]  # 截取所需层数
                result = self.temporal_aggregator(rois=rois, feat3d=feat3d_list, roi_extractor=self.bbox_roi_extractor)
                roi_feats = result[0] if isinstance(result, tuple) else result
            else:
                roi_feats = self.bbox_roi_extractor(x, rois)
        # 4) 计算动作和方向预测（对每个RoI）
        if roi_feats is not None and self.action_dir_head is not None:
            logits_action, logits_dir = self.action_dir_head(roi_feats)
        else:
            logits_action = logits_dir = None
        # 5) 计算语义分割预测（使用末帧的低层次特征，例如P2层）
        sem_logits = self.semantic_head(x[0]) if self.semantic_head is not None else None

        # 后续将 logits_action, logits_dir 填入 det_results 中每个实例的 pred_instances，
        # 并将 sem_logits 恢复到原图尺寸填入 data_samples 的 pred_sem_seg（略）
        
        fixed = []
        off = 0
        for i, (out_ds, in_ds) in enumerate(zip(det_results, data_samples)):
            # ensure DetDataSample wrapper
            if isinstance(out_ds, DetDataSample):
                ds = out_ds
            else:
                ds = DetDataSample()
                ds.set_metainfo(dict(in_ds.metainfo))
                inst = out_ds if isinstance(out_ds, InstanceData) else InstanceData()
                ds.set_field(inst, 'pred_instances')

            pred = getattr(ds, 'pred_instances', None)
            if pred is None:
                pred = InstanceData(); ds.set_field(pred, 'pred_instances')

            # guarantee fields
            if not hasattr(pred, 'bboxes'): pred.bboxes = torch.zeros((0,4), dtype=torch.float32, device=device)
            if not hasattr(pred, 'scores'): pred.scores = torch.zeros((0,), dtype=torch.float32, device=device)
            if not hasattr(pred, 'labels'): pred.labels = torch.zeros((0,), dtype=torch.long, device=device)  # ← correct

            # write per-ROI actions/dirs if we computed them
            n = pred.bboxes.shape[0]
            if logits_action is not None and n > 0:
                # print('roi logits a is not none! ')
                ds_inds = slice(off, off + n)
                pred.actions = logits_action[ds_inds].argmax(dim=1)
                pred.dirs    = logits_dir[ds_inds].argmax(dim=1)
            else:
                pred.actions = torch.zeros((n,), dtype=torch.long, device=device)
                pred.dirs    = torch.zeros((n,), dtype=torch.long, device=device)
            off += n

            # map predicted boxes from resized-cropped -> ori_space: (x,y) = (x'+off)/scale
            if n > 0:
                meta = ds.metainfo
                sf = torch.as_tensor(meta.get('scale_factor', [1,1,1,1]), device=pred.bboxes.device, dtype=pred.bboxes.dtype)
                # if meta.get('crop_offset', None) is None:
                    # ds.set_metainfo({'crop_offset': (0, 0)})
                y0, x0 = meta.get('crop_offset', (0, 0))
                offvec = torch.tensor([x0, y0, x0, y0], device=pred.bboxes.device, dtype=pred.bboxes.dtype)
                bb = pred.bboxes + 0.0  # copy
                bb[:, [0,2]] = (bb[:, [0,2]] + offvec[[0,2]]) / sf[[0,2]]
                bb[:, [1,3]] = (bb[:, [1,3]] + offvec[[1,3]]) / sf[[1,3]]
                pred.bboxes = bb

            # write pred_sem_seg per image in resized-cropped space (same size as img_shape)
            if sem_logits is not None:
                # print('roi sem logits is not none! ')
                ih, iw = ds.metainfo.get('img_shape', (None, None))
                if ih and iw:
                    seg_hw = sem_logits[i].argmax(dim=0)  # (H_r, W_r)
                    # Upsample to final image size without introducing an extra channel dim
                    seg_hw = F.interpolate(
                        seg_hw[None, None, ...].float(),  # (1,1,H_r,W_r)
                        size=(ih, iw),
                        mode='nearest'
                    ).squeeze(0).squeeze(0).to(torch.long)  # -> (H, W)
                    ds.set_field(PixelData(data=seg_hw), 'pred_sem_seg')

            fixed.append(ds)

            # 一次性/每图打印（建议一次性）
            if not hasattr(self, '_once_pred_dump'):
                n = int(pred.bboxes.shape[0])
                meta = ds.metainfo
                print('[PRED-DUMP]',
                    'img_id=', meta.get('img_id'),
                    'ori=', meta.get('ori_shape'),
                    'img=', meta.get('img_shape'),
                    'sf=',  meta.get('scale_factor'),
                    'off=', meta.get('crop_offset'))
                print('[PRED-DUMP] N=', n,
                    'has_actions=', hasattr(pred, 'actions'),
                    'has_dirs=', hasattr(pred, 'dirs'),
                    'has_sem=', hasattr(ds, 'pred_sem_seg'))
                if n > 0:
                    b = pred.bboxes
                    s = pred.scores
                    print('[PRED-DUMP] boxes[min/max]=',
                        float(b[:,[0,2]].min()), float(b[:,[0,2]].max()),
                        float(b[:,[1,3]].min()), float(b[:,[1,3]].max()))
                    print('[PRED-DUMP] labels[:8]=', pred.labels[:8].detach().cpu().tolist())
                    print('[PRED-DUMP] actions[:8]=', getattr(pred,'actions')[ :8].detach().cpu().tolist())
                    print('[PRED-DUMP] dirs[:8]=',    getattr(pred,'dirs'   )[ :8].detach().cpu().tolist())
                    print('[PRED-DUMP] scores[:5]=',  s[:5].detach().cpu().tolist())
                if hasattr(ds, 'pred_sem_seg'):
                    m = ds.pred_sem_seg.data
                    print('[PRED-DUMP] sem_shape=', tuple(m.shape), 'unique=', torch.unique(m).detach().cpu().tolist()[:8])
                self._once_pred_dump = True

            if not hasattr(self, '_once_return_dump'):
                tot = sum(int(getattr(ds.pred_instances,'bboxes',torch.empty(0,4)).shape[0]) for ds in fixed)
                print('[RETURN-DUMP] batch_pred_total=', tot,
                    'has_sem_all=', [hasattr(ds,'pred_sem_seg') for ds in fixed][:4])
            self._once_return_dump = True

        return fixed
