# -*- coding: utf-8 -*-
from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.structures.bbox import bbox2roi
from mmengine.structures import InstanceData, PixelData
from mmdet.structures import DetDataSample
from mmengine.logging import MMLogger


@MODELS.register_module()
class VascoDualPathRoIHead(StandardRoIHead):
    """
    双路 RoIHead（检测走 2D，动作/方向走时序聚合；语义从 P2 出图）。
    - 训练：
        * 检测：直接 super().loss(...)（内部完成 assign+sample）
        * 动作/方向：用 GT 框构造 RoI -> (可选) TemporalAlignAggregator -> ActionDirHead.loss
        * 语义：P2 -> SemanticHead.loss（如 head 内有 loss_from_batch 则调用之）
    - 推理：
        * 先 super().predict(..., rescale=False) 得到“模型尺度”的检测框
        * 用这些未还原的框做 RoI 聚合 -> ActionDirHead.predict -> 写回 actions/dirs
        * boxes 最后手工 /scale_factor 还原到原图坐标
        * 语义直接从 P2 预测，可按需上采样到 img_shape
    Notes:
        - 不再使用 `_assign_and_sample`/`assign_and_sample`
        - 不使用 `_sanitize_scale_factor`，改为 predict() 内部 `_get_sf4` 求末帧 4 维缩放因子
    
    """

    def __init__(self,
                 temporal_aggregator: Optional[Dict] = None,
                 action_dir_head: Optional[Dict] = None,
                 semantic_head: Optional[Dict] = None,
                 ego_stopgo_head: Optional[Dict] = None,
                 ego_topk: int = 10,
                 ego_roi_score_thr: float = 0.2,
                 ego_roi_use_rpn_in_train: bool = True,
                 aux_detach: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.temporal_aggregator = MODELS.build(temporal_aggregator) if temporal_aggregator else None
        self.action_dir_head = MODELS.build(action_dir_head) if action_dir_head else None
        self.semantic_head = MODELS.build(semantic_head) if semantic_head else None
        self.ego_stopgo_head = MODELS.build(ego_stopgo_head) if ego_stopgo_head else None
        self.aux_detach = bool(aux_detach)

        self.ego_topk = int(ego_topk)
        self.ego_roi_score_thr = float(ego_roi_score_thr)
        self.ego_roi_use_rpn_in_train = bool(ego_roi_use_rpn_in_train)

        self._temporal_ctx: Optional[Dict] = None
        self._last_temporal_losses: Optional[Dict] = None


    def _roi_token_from_boxes(self,
                             x_2d_fpn: List[torch.Tensor],
                             boxes_list: List[Optional[torch.Tensor]],
                             scores_list: Optional[List[Optional[torch.Tensor]]] = None,
                             topk: int = 10,
                             score_thr: float = 0.0) -> torch.Tensor:
        """Extract per-image RoI token (B, C) from boxes using bbox_roi_extractor.

        Args:
            x_2d_fpn: list of FPN feats, each (B,C,H,W)
            boxes_list: len=B, each (N,4) in xyxy (model scale)
            scores_list: len=B, each (N,) scores (optional)
        Returns:
            token: (B, C) where C == bbox_roi_extractor.out_channels
        """
        device = x_2d_fpn[0].device
        B = x_2d_fpn[0].size(0)
        C = getattr(self.bbox_roi_extractor, 'out_channels', x_2d_fpn[0].size(1))

        rois_all = []
        keep_cnt = [0 for _ in range(B)]

        for b in range(B):
            boxes = boxes_list[b]
            if boxes is None or not torch.is_tensor(boxes) or boxes.numel() == 0:
                continue

            boxes = boxes.to(device)
            if scores_list is not None and scores_list[b] is not None and torch.is_tensor(scores_list[b]):
                scores = scores_list[b].to(device)
                m = scores >= float(score_thr)
                boxes = boxes[m]
                scores = scores[m]
                if boxes.numel() == 0:
                    continue
                k = min(int(topk), boxes.size(0))
                idx = torch.topk(scores, k=k, largest=True).indices
                boxes = boxes[idx]
            else:
                k = min(int(topk), boxes.size(0))
                boxes = boxes[:k]

            keep_cnt[b] = int(boxes.size(0))
            b_inds = torch.full((boxes.size(0), 1), float(b), device=device, dtype=boxes.dtype)
            rois_all.append(torch.cat([b_inds, boxes], dim=1))  # (k,5)

        if len(rois_all) == 0:
            return x_2d_fpn[0].new_zeros((B, C))

        rois = torch.cat(rois_all, dim=0).contiguous()
        roi_feats = self.bbox_roi_extractor(x_2d_fpn, rois)  # (sum_k,C,h,w)
        roi_vec = F.adaptive_avg_pool2d(roi_feats, 1).flatten(1)  # (sum_k,C)

        out = x_2d_fpn[0].new_zeros((B, C))
        start = 0
        for b in range(B):
            k = keep_cnt[b]
            if k > 0:
                out[b] = roi_vec[start:start + k].mean(dim=0)
                start += k
        return out
    # --------------------------------
    # 训练阶段：ego(stop/go) 损失（末帧，全局分类）
    # 输入: 末帧 2D FPN 特征 + ego_choice_id，输出: stop/go
    # --------------------------------
    def _loss_ego_stopgo(self, x_2d_fpn: List[torch.Tensor], rpn_results_list, data_samples: List) -> Dict[str, torch.Tensor]:
        if self.ego_stopgo_head is None:
            z = x_2d_fpn[0].sum() * 0
            return dict(loss_stopgo=z, stopgo_acc=z)

        # Use a single FPN level feature for global classification (default: last level)
        # feat = x_2d_fpn[-1]
        # E2: fuse ALL FPN stages by GAP-per-level then concat -> (B, 256 * L)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in x_2d_fpn]
        feat_global = torch.cat(pooled, dim=1)  # (B, 256*num_outs)
        device = feat_global.device

        # E3: Pred RoI token for ego (train-time). Use RPN proposals for stability.
        roi_tok = None
        if self.ego_roi_use_rpn_in_train and rpn_results_list is not None:
            boxes_list, scores_list = [], []
            for r in rpn_results_list:
                if isinstance(r, dict):
                    boxes_list.append(r.get('bboxes', None))
                    scores_list.append(r.get('scores', None))
                else:
                    boxes_list.append(getattr(r, 'bboxes', None))
                    scores_list.append(getattr(r, 'scores', None))
            roi_tok = self._roi_token_from_boxes(
                x_2d_fpn,
                boxes_list=boxes_list,
                scores_list=scores_list,
                topk=self.ego_topk,
                score_thr=self.ego_roi_score_thr,
            )  # (B,256)
        else:
            # If disabled/unavailable, fall back to zeros
            roi_tok = x_2d_fpn[0].new_zeros((feat_global.size(0), x_2d_fpn[0].size(1)))

        # E3 fused ego feature
        feat = torch.cat([feat_global, roi_tok], dim=1)  # (B, 256*num_outs + 256)

        ego_ids = []
        gts = []
        for ds in data_samples:
            ego = getattr(ds, 'ego_choice_id', None)
            gt = getattr(ds, 'gt_stopgo', None)
            if ego is None or gt is None:
                # If missing, skip ego loss for this batch
                z = feat_global.sum() * 0
                return dict(loss_stopgo=z, stopgo_acc=z)
            if not isinstance(ego, torch.Tensor):
                ego = torch.tensor(ego, device=device)
            if not isinstance(gt, torch.Tensor):
                gt = torch.tensor(gt, device=device)
            ego_ids.append(ego.to(device).view(1))
            gts.append(gt.to(device).view(1))

        ego_ids = torch.cat(ego_ids, dim=0).long()  # (B,)
        gt_stopgo = torch.cat(gts, dim=0).long()    # (B,)

        # Head should accept (feat, ego_ids) and return logits (B,2)
        logits = self.ego_stopgo_head(feat, ego_ids)
        losses = self.ego_stopgo_head.loss(logits, gt_stopgo)

        # Ensure standardized keys exist
        if 'loss_stopgo' not in losses:
            # allow head to return loss_ego or similar; remap if needed
            for k in ['loss_ego', 'loss_stop_go', 'loss_stop_go_ce']:
                if k in losses:
                    losses['loss_stopgo'] = losses.pop(k)
                    break
        if 'stopgo_acc' not in losses:
            # compute simple acc if head didn't provide
            pred = logits.argmax(dim=1)
            losses['stopgo_acc'] = (pred == gt_stopgo).float().mean()

        return losses

    # --------------------------------
    # Detector 在 extract_feat 后注入 3D 上下文
    # --------------------------------
    def set_temporal_ctx(self, ctx: Dict):
        """ctx 形如: {'feat3d': list[L](B,T',C,H,W), ...}"""
        self._temporal_ctx = ctx

    # --------------------------------
    # 训练阶段：动作/方向损失（基于 GT 框）
    # --------------------------------
    def _loss_action_dir(self, x_2d_fpn: List[torch.Tensor], data_samples: List) -> Dict[str, torch.Tensor]:
        """使用 GT 框构造 RoI → (可选) 时序聚合 → ActionDirHead CE/metric 损失。"""
        if self.action_dir_head is None:
            z = x_2d_fpn[0].sum() * 0
            return dict(loss_action=z, loss_dir=z, action_acc=z, dir_acc=z)

        # 组装每图的 GT 框（只 4 列，batch 索引交给 bbox2roi）
        rois_per_img, gt_actions_all, gt_dirs_all = [], [], []
        total = 0
        for ds in data_samples:
            gt = ds.gt_instances
            if not hasattr(gt, 'bboxes') or gt.bboxes.numel() == 0:
                continue
            B = gt.bboxes  # (n,4)
            n = B.size(0)
            total += n
            rois_per_img.append(B)

            ga = None
            for k in ['actions', 'gt_actions']:
                if hasattr(gt, k):
                    ga = getattr(gt, k); break
            gd = None
            for k in ['dirs', 'gt_dirs', 'directions', 'gt_directions']:
                if hasattr(gt, k):
                    gd = getattr(gt, k); break
            if ga is None:
                ga = torch.zeros((n,), dtype=torch.long, device=B.device)
            if gd is None:
                gd = torch.zeros((n,), dtype=torch.long, device=B.device)
            gt_actions_all.append(ga)
            gt_dirs_all.append(gd)

        if total == 0:
            z = x_2d_fpn[0].sum() * 0
            return dict(loss_action=z, loss_dir=z, action_acc=z, dir_acc=z)

        rois = bbox2roi(rois_per_img).to(dtype=x_2d_fpn[0].dtype).contiguous()  # (R,5)
        # print(rois.shape)

        gt_actions = torch.cat(gt_actions_all, dim=0).to(x_2d_fpn[0].device)
        gt_dirs    = torch.cat(gt_dirs_all,   dim=0).to(x_2d_fpn[0].device)

        # 2D RoI
        roi_2d = self.bbox_roi_extractor(x_2d_fpn, rois)  # (R,C,h,w)
        # print(roi_2d.shape)

        if self.aux_detach:
            roi_2d = roi_2d.detach()

        # 时序聚合
        roi_agg = roi_2d
        self._last_temporal_losses = None
        ctx = getattr(self, '_temporal_ctx', None)
        if self.temporal_aggregator is not None and ctx is not None and 'feat3d' in ctx:
            out = self.temporal_aggregator(
                rois=rois,
                roi_extractor=self.bbox_roi_extractor,
                feat3d=ctx.get('feat3d', None),
                feats2d_seq=None,
                roi_feats_last2d=roi_2d
            )
            if isinstance(out, tuple):
                roi_agg, agg_losses = out
                self._last_temporal_losses = agg_losses
            else:
                roi_agg = out

        # 动作/方向损失
        logits_action, logits_dir = self.action_dir_head(roi_agg)
        aux_losses = self.action_dir_head.loss(logits_action, logits_dir, gt_actions, gt_dirs)

        # 叠加时序一致性（如有）
        if isinstance(self._last_temporal_losses, dict) and 'loss_roi_consistency' in self._last_temporal_losses:
            aux_losses['loss_roi_consistency'] = self._last_temporal_losses['loss_roi_consistency']

        return aux_losses

    # --------------------------------
    # 训练阶段：语义分割损失（P2）
    # --------------------------------
    def _loss_semantic(self, x_2d_fpn: List[torch.Tensor], data_samples: List) -> Dict[str, torch.Tensor]:
        if self.semantic_head is None:
            z = x_2d_fpn[0].sum() * 0
            return dict(loss_sem_ce=z, loss_sem_dice=z)

        if hasattr(self.semantic_head, 'loss_from_batch'):
            # 你的 head 已提供按 batch 读取 GT 的接口
            return self.semantic_head.loss_from_batch(x_2d_fpn[:3], data_samples)

        # 兜底：简单 CE + Dice
        logits = self.semantic_head(x_2d_fpn[:3])  # (B,C,H/4,W/4)
        # 收集 GT
        gts = []
        for ds in data_samples:
            gt = None
            for k in ['gt_sem_seg', 'sem_seg', 'seg']:
                if hasattr(ds, k):
                    val = getattr(ds, k)
                    if hasattr(val, 'data'):
                        gt = val.data
                    elif hasattr(val, 'tensor'):
                        gt = val.tensor
                    else:
                        gt = val
                    break
            if gt is None:
                z = logits.sum() * 0
                return dict(loss_sem_ce=z, loss_sem_dice=z)
            if gt.dim() == 3 and gt.size(0) == 1:
                gt = gt[0]
            gts.append(gt.to(logits.device).long())
        gt = torch.stack(gts, dim=0)
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = F.interpolate(gt.unsqueeze(1).float(),
                               size=logits.shape[-2:], mode='nearest').squeeze(1).long()
        ce = F.cross_entropy(logits, gt, ignore_index=255)

        probs = logits.softmax(dim=1)
        valid = (gt != 255)
        if valid.sum() > 0:
            t = gt.clone(); t[~valid] = 0
            onehot = F.one_hot(t, num_classes=logits.size(1)).permute(0,3,1,2).to(probs.dtype)
            probs = probs * valid.unsqueeze(1)
            onehot = onehot * valid.unsqueeze(1)
            num = 2 * (probs * onehot).sum(dim=(0,2,3))
            den = (probs * probs).sum(dim=(0,2,3)) + (onehot * onehot).sum(dim=(0,2,3)) + 1e-6
            dice = (1.0 - num / den).mean()
        else:
            dice = ce.new_zeros(())
        return dict(loss_sem_ce=ce, loss_sem_dice=0.5 * dice)

    # --------------------------------
    # 总损失
    # --------------------------------
    def loss(self, x, rpn_results_list, data_samples, **kwargs):
        # 1) 检测损失（内部含 assign+sample）
        det_losses = super().loss(x, rpn_results_list, data_samples, **kwargs)
        # 2) 动作/方向损失（GT RoI）
        ad_losses = self._loss_action_dir(x, data_samples)
        # 3) 语义损失（P2）
        sem_losses = self._loss_semantic(x, data_samples)
        # 4) ego(stop/go) 损失（末帧）
        ego_losses = self._loss_ego_stopgo(x, rpn_results_list, data_samples)
        # z = x[0].sum() * 0
        # losses.update(dict(loss_sem_ce=z, loss_sem_dice=z))
        det_losses.update(ad_losses)
        det_losses.update(sem_losses)
        det_losses.update(ego_losses)
        return det_losses

    # --------------------------------
    # 预测：rescale=False + 外部手工还原
    # --------------------------------
    def predict(self, x, rpn_results_list, data_samples, rescale: bool = False):
        """标准框预测 + 语义 + 动作/方向。
        流程：
        1) super().predict(..., rescale=False) 得到“模型尺度”框（InstanceData 列表）
        2) 语义：用 FPN 特征跑 semantic_head，写入 pred_sem_seg（含 data+logits）
        3) 动作/方向：用未还原框做 RoI 聚合+分类，写回 actions/dirs
        4) 手工将 bboxes / scale_factor 还原到原图坐标
        """

        # 小工具：取末帧 4 维 scale_factor（w,h,w,h）
        def _get_sf4(ds, device, dtype):
            sf = ds.metainfo.get('scale_factor', None)
            if sf is not None:
                s = torch.as_tensor(sf, device=device, dtype=dtype).view(-1)
                if s.numel() >= 4:
                    return s[-4:]
            return torch.ones(4, device=device, dtype=dtype)

        # 1) 标准 bbox 预测（不立刻 rescale）
        inst_list = super().predict(x, rpn_results_list, data_samples, rescale=False)

        # 2) 语义分割预测：用 FPN 特征 -> semantic_head -> PixelData(data+logits)
        if getattr(self, 'semantic_head', None) is not None:
            in_ch = getattr(self.semantic_head, 'in_channels', None)
            # 训练里 semantic_head 是 in_channels=(256,256,256)，用的是 P2,P3,P4
            if isinstance(in_ch, (tuple, list)):
                feats_sem = list(x[:len(in_ch)])
            else:
                feats_sem = x[0]

            logits = self.semantic_head(feats_sem)  # (B,C_sem,Hs,Ws)
            B = logits.size(0)
            for i in range(B):
                ds = data_samples[i]
                logit_i = logits[i]                 # (C_sem,Hs,Ws)

                img_shape = ds.metainfo.get('img_shape', None)
                if img_shape is not None:
                    ih, iw = int(img_shape[0]), int(img_shape[1])
                    if logit_i.shape[-2:] != (ih, iw):
                        logit_i = F.interpolate(
                            logit_i.unsqueeze(0),
                            size=(ih, iw),
                            mode='bilinear',
                            align_corners=False
                        )[0]

                mask = logit_i.argmax(dim=0).to(torch.int64)  # (H,W)

                ds.pred_sem_seg = PixelData(
                    data=mask.unsqueeze(0),   # (1,H,W)
                    logits=logit_i           # (C_sem,H,W)
                )
                # 如果你想兼容 metainfo['seg_logits'] 这条支路，可以顺便存一份：
                # ds.set_metainfo(dict(seg_logits=logit_i.detach().cpu().numpy()))

        # 2.5) ego(stop/go) 预测：末帧全局分类（写入 data_samples）
        if getattr(self, 'ego_stopgo_head', None) is not None:
            # E2: pooled global feature
            pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in x]
            feat_global = torch.cat(pooled, dim=1)  # (B,256*num_outs)
            B = feat_global.size(0)

            # E3: Pred RoI token from final det results (deployment-realistic)
            boxes_list, scores_list = [], []
            for inst in inst_list:
                if inst is None or not hasattr(inst, 'bboxes') or inst.bboxes.numel() == 0:
                    boxes_list.append(None)
                    scores_list.append(None)
                else:
                    boxes_list.append(inst.bboxes)
                    scores_list.append(getattr(inst, 'scores', None))
            roi_tok = self._roi_token_from_boxes(
                x,
                boxes_list=boxes_list,
                scores_list=scores_list,
                topk=self.ego_topk,
                score_thr=self.ego_roi_score_thr,
            )  # (B,256)

            feat = torch.cat([feat_global, roi_tok], dim=1)  # (B,256*num_outs+256)
            ego_ids = []
            for i in range(B):
                ego = getattr(data_samples[i], 'ego_choice_id', None)
                if ego is None:
                    ego_t = torch.zeros((1,), device=feat.device, dtype=torch.long)
                else:
                    # ego_choice_id may be int/np scalar from demo metainfo; normalize to tensor
                    if not isinstance(ego, torch.Tensor):
                        ego = torch.tensor(ego, device=feat.device)
                    ego_t = ego.to(feat.device).view(1).long()
                ego_ids.append(ego_t)
            ego_ids = torch.cat(ego_ids, dim=0)  # (B,)

            with torch.no_grad():
                logits = self.ego_stopgo_head(feat, ego_ids)  # (B,2)
                prob = logits.softmax(dim=1)                  # (B,2)
                pred = prob.argmax(dim=1)                     # (B,)

            # debug: print stop/go probabilities (avoid log spam)
            if not hasattr(self, '_dbg_ego_prob'):
                self._dbg_ego_prob = 0
            if self._dbg_ego_prob < 80:
                logger = MMLogger.get_current_instance()
                # assume index 0=stop, 1=go (verify by checking prob values)
                logger.info(
                    f"[EGO-PROB] call={self._dbg_ego_prob} "
                    f"ego_id={int(ego_ids[0].detach().cpu()) if ego_ids.numel() > 0 else -1} "
                    f"p_stop={float(prob[0,0].detach().cpu()):.4f} "
                    f"p_go={float(prob[0,1].detach().cpu()):.4f} "
                    f"pred={int(pred[0].detach().cpu())}"
                )
                self._dbg_ego_prob += 1

            for i in range(B):
                # store both raw and id for evaluator/debug
                data_samples[i].pred_stopgo = pred[i].detach().cpu()
                data_samples[i].pred_stopgo_logits = logits[i].detach().cpu()

        # 3) 动作 / 方向：对当前 inst_list 做时序聚合 + 分类
        if self.action_dir_head is not None and self.temporal_aggregator is not None:
            all_rois = []
            img_offsets = []
            n0 = 0

            for i, inst in enumerate(inst_list):
                if inst is None or inst.bboxes.numel() == 0:
                    img_offsets.append((n0, n0))
                    continue
                b = inst.bboxes
                n = b.size(0)
                n0_next = n0 + n
                batch_inds = torch.full((n, 1), i, dtype=b.dtype, device=b.device)
                rois_i = torch.cat([batch_inds, b], dim=1)  # (n,5)
                all_rois.append(rois_i)
                img_offsets.append((n0, n0_next))
                n0 = n0_next

            if len(all_rois) > 0:
                rois_cat = torch.cat(all_rois, dim=0)  # (R,5)
                roi_2d = self.bbox_roi_extractor(x, rois_cat)
                roi_agg = roi_2d
                if self.temporal_aggregator is not None and self._temporal_ctx is not None:
                    out = self.temporal_aggregator(
                        rois=rois_cat,
                        roi_extractor=self.bbox_roi_extractor,
                        feat3d=self._temporal_ctx.get('feat3d', None),
                        feats2d_seq=None,
                        roi_feats_last2d=roi_2d
                    )
                    roi_agg = out[0] if isinstance(out, tuple) else out

                with torch.no_grad():
                    pa, pd = self.action_dir_head.predict(roi_agg)  # (R,)

                # 写回每图的 InstanceData
                for i, inst in enumerate(inst_list):
                    s, e = img_offsets[i]
                    if s == e:
                        continue
                    inst.actions = pa[s:e].cpu()
                    inst.dirs = pd[s:e].cpu()

        # 4) 手工将 boxes 还原到原图（除以 scale_factor）
        for i, inst in enumerate(inst_list):
            if inst is None or inst.bboxes.numel() == 0:
                continue
            sf4 = _get_sf4(data_samples[i], inst.bboxes.device, inst.bboxes.dtype)  # (4,)
            inst.bboxes = inst.bboxes / sf4

        return inst_list