# projects/vasco/models/roi_heads/vasco_singlepath_roi_head.py
# -*- coding: utf-8 -*-
from typing import Dict, Optional, List, Tuple, Any

import torch
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.structures.bbox import bbox2roi
from mmengine.structures import PixelData
from mmengine.logging import MMLogger


@MODELS.register_module()
class VascoSinglePathRoIHead(StandardRoIHead):
    """
    Single-path RoIHead (2D only):
      - Det: inherited StandardRoIHead (super().loss / super().predict)
      - Action/Dir: GT RoIs (train) or predicted boxes (test) -> RoIAlign on last-frame FPN
                   + TemporalMSDeformAttnAggregator over (multi-frame, multi-scale 2D FPN seq)
      - Seg: same as your existing semantic_head usage (optional)
      - Ego: same as your existing ego_stopgo_head usage (optional)

    Required temporal ctx (set by detector before calling roi_head.loss/predict):
      ctx = {
        'feats2d_seq' or 'feat2d_seq': List[Tensor],  # len=L, each (B,T,C,H,W), C must be 256 if FPN out=256
        'img_hw': (H_img, W_img),     # current network input size (after resize/crop/pad), for ROI center norm
      }
    """

    def __init__(self,
                 temporal_aggregator: Optional[Dict] = None,
                 action_dir_head: Optional[Dict] = None,
                 semantic_head: Optional[Dict] = None,
                 ego_stopgo_head: Optional[Dict] = None,
                 ego_topk: int = 10,
                 ego_roi_score_thr: float = 0.2,
                 ego_roi_use_rpn_in_train: bool = True,
                 aux_detach: bool = False,
                 fuse_mode: str = 'add',   # 'add' (default) or 'cat' (if you later change ActionDirHead)
                 ad_topk_per_gt: int = 0,
                 **kwargs):
        super().__init__(**kwargs)

        self.temporal_aggregator = MODELS.build(temporal_aggregator) if temporal_aggregator else None
        self.action_dir_head = MODELS.build(action_dir_head) if action_dir_head else None
        self.semantic_head = MODELS.build(semantic_head) if semantic_head else None
        self.ego_stopgo_head = MODELS.build(ego_stopgo_head) if ego_stopgo_head else None

        self.ego_topk = int(ego_topk)
        self.ego_roi_score_thr = float(ego_roi_score_thr)
        self.ego_roi_use_rpn_in_train = bool(ego_roi_use_rpn_in_train)

        self.aux_detach = bool(aux_detach)
        self.fuse_mode = str(fuse_mode)
        self.ad_topk_per_gt = int(ad_topk_per_gt)

        self._temporal_ctx: Optional[Dict] = None
        self._dbg_once = False
        self._dbg_post_ego_once = False


    def _build_downstream_inst_and_seg(self,
                                       x,
                                       rpn_results_list,
                                       data_samples: List):
        """Build detached downstream ego inputs from predicted det/action-dir/seg results.

        This helper mirrors the necessary parts of `predict(...)` but keeps everything
        in the current network-input coordinate space. It is intended only for the
        downstream ego branch, whose inputs are detached from the perception heads.
        """
        # MMDetection 3.x RPN proposals may use `priors` instead of `bboxes`.
        # StandardRoIHead.predict_bbox() expects `res.bboxes`.
        for res in rpn_results_list:
            if not hasattr(res, 'bboxes') and hasattr(res, 'priors'):
                res.bboxes = res.priors
        inst_list = super().predict(x, rpn_results_list, data_samples, rescale=False)

        # seg prediction -> write to data_samples[i].pred_sem_seg
        seg_feat = None
        if self.semantic_head is not None:
            feats_sem = list(x[:4])

            sem_out = self.semantic_head(feats_sem)
            if isinstance(sem_out, tuple):
                logits, seg_feat = sem_out
            else:
                logits = sem_out
                seg_feat = None

            B = logits.size(0)
            for i in range(B):
                ds = data_samples[i]
                logit_i = logits[i]
                img_shape = ds.metainfo.get('img_shape', None)
                if img_shape is not None:
                    ih, iw = int(img_shape[0]), int(img_shape[1])
                    if logit_i.shape[-2:] != (ih, iw):
                        logit_i = F.interpolate(
                            logit_i.unsqueeze(0),
                            size=(ih, iw),
                            mode='bilinear',
                            align_corners=False,
                        )[0]
                mask = logit_i.argmax(dim=0).to(torch.int64)
                ds.pred_sem_seg = PixelData(data=mask.unsqueeze(0), logits=logit_i.detach())

        # action/dir prediction -> enrich each instance
        if self.action_dir_head is not None:
            all_rois = []
            img_offsets = []
            n0 = 0
            for i, inst in enumerate(inst_list):
                if inst is None or (not hasattr(inst, 'bboxes')) or inst.bboxes.numel() == 0:
                    img_offsets.append((n0, n0))
                    continue
                b = inst.bboxes
                n = b.size(0)
                batch_inds = torch.full((n, 1), i, dtype=b.dtype, device=b.device)
                all_rois.append(torch.cat([batch_inds, b], dim=1))
                img_offsets.append((n0, n0 + n))
                n0 += n

            if len(all_rois) > 0:
                rois_cat = torch.cat(all_rois, dim=0).contiguous()
                roi_2d = self.bbox_roi_extractor(x, rois_cat)
                roi_agg = roi_2d
                if self.temporal_aggregator is not None and self._temporal_ctx is not None:
                    feats2d_seq = self._temporal_ctx.get('feats2d_seq', None)
                    if feats2d_seq is None:
                        feats2d_seq = self._temporal_ctx.get('feat2d_seq', None)
                    img_hw = self._temporal_ctx.get('img_hw', None)
                    if feats2d_seq is not None and img_hw is not None:
                        query_token = F.adaptive_avg_pool2d(roi_2d, 1).flatten(1)
                        fused_token = self.temporal_aggregator(
                            query=query_token,
                            rois=rois_cat,
                            feat2d_seq=feats2d_seq,
                            img_hw=img_hw
                        )
                        roi_agg = self._fuse_token_to_roi_map(roi_2d, fused_token)

                pa, pd = self.action_dir_head.predict(roi_agg)
                # cache ROI features for downstream ego head
                roi_feat_cache = roi_agg.detach()
                for i, inst in enumerate(inst_list):
                    s, e = img_offsets[i]
                    if s == e:
                        continue
                    inst.actions = pa[s:e].detach()
                    inst.dirs = pd[s:e].detach()
                    inst.roi_feats = roi_feat_cache[s:e]

        # attach semantic fusion feature for downstream ego head
        if seg_feat is not None:
            for i in range(len(data_samples)):
                data_samples[i].seg_feat = seg_feat[i].detach()

        return inst_list

    # -------------------------
    # ctx injection (2D seq)
    # -------------------------
    def set_temporal_ctx(self, ctx: Dict):
        self._temporal_ctx = ctx


    # -------------------------
    # fuse temporal token back to ROI feature map for ActionDirHead (no head change)
    # -------------------------
    def _fuse_token_to_roi_map(self, roi_map: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        """
        roi_map: (R,C,h,w)  token: (R,C)
        return: (R,C,h,w) for existing ActionDirHead
        """
        if self.fuse_mode == 'add':
            return roi_map + token[:, :, None, None]
        elif self.fuse_mode == 'cat':
            # only if you later modify ActionDirHead in_channels accordingly
            t = token[:, :, None, None].expand(-1, -1, roi_map.size(2), roi_map.size(3))
            return torch.cat([roi_map, t], dim=1)
        else:
            raise ValueError(f"Unknown fuse_mode={self.fuse_mode}")

    
    # -------------------------
    # (train) action/dir loss using proposal    
    # -------------------------
    def _loss_action_dir_from_proposals(
        self,
        x_2d_fpn: List[torch.Tensor],
        sampling_results: List,
        data_samples: List
    ) -> Dict[str, torch.Tensor]:
        """
        Proposal-based action/dir training:
          - RoI features come from positive sampled proposals
          - GT is used only for target assignment via pos_assigned_gt_inds
        """
        if self.action_dir_head is None:
            z = x_2d_fpn[0].sum() * 0
            return dict(loss_action=z, loss_dir=z, action_acc=z, dir_acc=z)

        rois_per_img = []
        gt_actions_all = []
        gt_dirs_all = []
        total = 0

        for img_idx, res in enumerate(sampling_results):
            # 兼容不同 SamplingResult 字段名
            pos_boxes = getattr(res, 'pos_priors', None)
            if pos_boxes is None:
                pos_boxes = getattr(res, 'pos_bboxes', None)

            if pos_boxes is None or (not torch.is_tensor(pos_boxes)) or pos_boxes.numel() == 0:
                continue

            gt_inds = getattr(res, 'pos_assigned_gt_inds', None)
            if gt_inds is None or gt_inds.numel() == 0:
                continue

            gt = data_samples[img_idx].gt_instances

            # -------------------------------------------------
            # top-k positive proposals per GT (by IoU / overlap quality)
            # ad_topk_per_gt <= 0 means keep all positives (old behavior)
            # -------------------------------------------------
            if self.ad_topk_per_gt > 0:
                max_overlaps = getattr(res, 'pos_assigned_gt_inds', None)
                # prefer IoU/overlap from assign_result if available
                assign_result = getattr(res, 'assign_result', None)
                pos_inds = getattr(res, 'pos_inds', None)
                if assign_result is not None and hasattr(assign_result, 'max_overlaps') and pos_inds is not None:
                    overlaps = assign_result.max_overlaps[pos_inds].to(pos_boxes.device)
                else:
                    # fallback: if overlap is unavailable, use box area as a weak stable proxy
                    wh = (pos_boxes[:, 2:] - pos_boxes[:, :2]).clamp(min=0)
                    overlaps = (wh[:, 0] * wh[:, 1]).to(pos_boxes.device)

                keep_mask = torch.zeros(pos_boxes.size(0), dtype=torch.bool, device=pos_boxes.device)
                uniq_gt = torch.unique(gt_inds)
                k = int(self.ad_topk_per_gt)
                for g in uniq_gt:
                    idx = torch.nonzero(gt_inds == g, as_tuple=False).squeeze(1)
                    if idx.numel() == 0:
                        continue
                    kk = min(k, idx.numel())
                    order = torch.argsort(overlaps[idx], descending=True)
                    keep_idx = idx[order[:kk]]
                    keep_mask[keep_idx] = True

                pos_boxes = pos_boxes[keep_mask]
                gt_inds = gt_inds[keep_mask]

            if pos_boxes.numel() == 0 or gt_inds.numel() == 0:
                continue

            n = pos_boxes.size(0)
            total += n
            rois_per_img.append(pos_boxes)

            # ---- action labels ----
            ga = getattr(gt, 'actions', None)
            if ga is None:
                ga = getattr(gt, 'gt_actions', None)

            # ---- direction labels ----
            gd = getattr(gt, 'dirs', None)
            if gd is None:
                gd = getattr(gt, 'gt_dirs', None)
            if gd is None:
                gd = getattr(gt, 'directions', None)
            if gd is None:
                gd = getattr(gt, 'gt_directions', None)

            # fallback: action invalid -> 4, dir invalid -> 5
            if ga is None:
                ga_sel = torch.full((n,), 4, dtype=torch.long, device=pos_boxes.device)
            else:
                ga = ga.to(device=pos_boxes.device, dtype=torch.long).view(-1)
                ga_sel = ga[gt_inds]
                valid_ga = (ga_sel >= 0) & (ga_sel <= 4)
                ga_sel = torch.where(valid_ga, ga_sel, ga_sel.new_full(ga_sel.shape, 4))

            if gd is None:
                gd_sel = torch.full((n,), 5, dtype=torch.long, device=pos_boxes.device)
            else:
                gd = gd.to(device=pos_boxes.device, dtype=torch.long).view(-1)
                gd_sel = gd[gt_inds]
                valid_gd = (gd_sel >= 0) & (gd_sel < 6)
                gd_sel = torch.where(valid_gd, gd_sel, gd_sel.new_full(gd_sel.shape, 5))

            gt_actions_all.append(ga_sel)
            gt_dirs_all.append(gd_sel)

        if total == 0:
            z = x_2d_fpn[0].sum() * 0
            return dict(loss_action=z, loss_dir=z, action_acc=z, dir_acc=z)

        rois = bbox2roi(rois_per_img).to(dtype=x_2d_fpn[0].dtype).contiguous()
        gt_actions = torch.cat(gt_actions_all, dim=0).to(x_2d_fpn[0].device)
        gt_dirs = torch.cat(gt_dirs_all, dim=0).to(x_2d_fpn[0].device)

        # 用 positive proposals 做 RoIAlign
        roi_2d = self.bbox_roi_extractor(x_2d_fpn, rois)
        if self.aux_detach:
            roi_2d = roi_2d.detach()

        # 时序聚合：保持你原来的逻辑不变
        if self.temporal_aggregator is not None and self._temporal_ctx is not None:
            feats2d_seq = self._temporal_ctx.get('feats2d_seq', None)
            if feats2d_seq is None:
                feats2d_seq = self._temporal_ctx.get('feat2d_seq', None)
            img_hw = self._temporal_ctx.get('img_hw', None)

            if feats2d_seq is not None and img_hw is not None:
                query_token = F.adaptive_avg_pool2d(roi_2d, 1).flatten(1)
                fused_token = self.temporal_aggregator(
                    query=query_token,
                    rois=rois,
                    feat2d_seq=feats2d_seq,
                    img_hw=img_hw
                )
                roi_agg = self._fuse_token_to_roi_map(roi_2d, fused_token)
            else:
                roi_agg = roi_2d
        else:
            roi_agg = roi_2d

        logits_action, logits_static_dir, logits_dynamic_dir = self.action_dir_head(roi_agg)

        # cache ROI context for ego stop/go training
        self._ego_train_roi_ctx = dict(
            roi_map=roi_agg.detach() if self.aux_detach else roi_agg,
            rois=rois.detach(),
            batch_size=len(data_samples),
        )

        losses = self.action_dir_head.loss(logits_action, logits_static_dir, logits_dynamic_dir, gt_actions, gt_dirs)
        return losses


    # -------------------------
    # (train) semantic loss (keep your current behavior)
    # -------------------------
    def _loss_semantic(self, x_2d_fpn: List[torch.Tensor], data_samples: List) -> Dict[str, torch.Tensor]:
        if self.semantic_head is None:
            z = x_2d_fpn[0].sum() * 0
            return dict(loss_sem_ce=z, loss_sem_dice=z)

        if hasattr(self.semantic_head, 'loss_from_batch'):
            return self.semantic_head.loss_from_batch(x_2d_fpn[:4], data_samples)


        raise NotImplementedError("semantic_head must implement loss_from_batch() in the current downstream-ego version")
    
    # -------------------------
    # (train) ego stop/go loss (same as your dualpath version)
    # -------------------------
    def _loss_ego_stopgo(self, x_2d_fpn: List[torch.Tensor], rpn_results_list, data_samples: List) -> Dict[str, torch.Tensor]:
        if self.ego_stopgo_head is None:
            z = x_2d_fpn[0].sum() * 0
            return dict(loss_stopgo=z, stopgo_acc=z)

        with torch.no_grad():
            inst_list = self._build_downstream_inst_and_seg(
                x=x_2d_fpn,
                rpn_results_list=rpn_results_list,
                data_samples=data_samples,
            )

        gt_list = []
        device = x_2d_fpn[0].device
        for ds in data_samples:
            gt = getattr(ds, 'gt_stopgo', None)
            if gt is None:
                z = x_2d_fpn[0].sum() * 0
                return dict(loss_stopgo=z, stopgo_acc=z)
            if not isinstance(gt, torch.Tensor):
                gt = torch.tensor(gt, device=device)
            gt_list.append(gt.to(device=device).view(1).long())
        gt_stopgo = torch.cat(gt_list, dim=0)

        # -------------------------
        # build padded ego inputs
        # -------------------------
        B = len(inst_list)
        K = int(self.ego_topk)
        device = x_2d_fpn[0].device

        labels = torch.full((B, K), -1, dtype=torch.long, device=device)
        actions = torch.full((B, K), -1, dtype=torch.long, device=device)
        dirs = torch.full((B, K), -1, dtype=torch.long, device=device)
        geom = torch.zeros((B, K, 5), dtype=torch.float32, device=device)
        scores = torch.zeros((B, K, 1), dtype=torch.float32, device=device)
        valid = torch.zeros((B, K), dtype=torch.bool, device=device)
        roi_feats = torch.zeros((B, K, 256, 7, 7), dtype=torch.float32, device=device)

        seg_feats = []

        for bidx, (inst, ds) in enumerate(zip(inst_list, data_samples)):
            seg_f = getattr(ds, 'seg_feat', None)
            if seg_f is None:
                seg_f = torch.zeros((256, 1, 1), dtype=torch.float32, device=device)
            seg_feats.append(seg_f.to(device=device, dtype=torch.float32))

            if inst is None:
                continue
            if (not hasattr(inst, 'bboxes')) or inst.bboxes.numel() == 0:
                continue
            if not hasattr(inst, 'roi_feats'):
                continue

            bboxes = inst.bboxes.to(device=device, dtype=torch.float32)
            cls_labels = inst.labels.to(device=device, dtype=torch.long)
            det_scores = inst.scores.to(device=device, dtype=torch.float32)
            act = inst.actions.to(device=device, dtype=torch.long)
            dr = inst.dirs.to(device=device, dtype=torch.long)
            rf = inst.roi_feats.to(device=device, dtype=torch.float32)

            keep = det_scores >= float(self.ego_roi_score_thr)
            if keep.sum() == 0:
                continue

            bboxes = bboxes[keep]
            cls_labels = cls_labels[keep]
            det_scores = det_scores[keep]
            act = act[keep]
            dr = dr[keep]
            rf = rf[keep]

            order = torch.argsort(det_scores, descending=True)
            order = order[:K]

            bboxes = bboxes[order]
            cls_labels = cls_labels[order]
            det_scores = det_scores[order]
            act = act[order]
            dr = dr[order]
            rf = rf[order]

            n = bboxes.size(0)
            if n == 0:
                continue

            labels[bidx, :n] = cls_labels
            actions[bidx, :n] = act
            dirs[bidx, :n] = dr
            scores[bidx, :n, 0] = det_scores
            valid[bidx, :n] = True
            roi_feats[bidx, :n] = rf

            img_shape = ds.metainfo.get('img_shape', None)
            if img_shape is None:
                pad_shape = ds.metainfo.get('pad_shape', (1,1))
                H = float(pad_shape[0])
                W = float(pad_shape[1])
            else:
                H = float(img_shape[0])
                W = float(img_shape[1])

            x1 = bboxes[:, 0] / W
            y1 = bboxes[:, 1] / H
            x2 = bboxes[:, 2] / W
            y2 = bboxes[:, 3] / H

            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            bw = (x2 - x1).clamp(min=1e-6)
            bh = (y2 - y1).clamp(min=1e-6)
            area = bw * bh

            geom[bidx, :n] = torch.stack([cx, cy, bw, bh, area], dim=-1)

        seg_feat = torch.stack(seg_feats, dim=0)

        losses = self.ego_stopgo_head.loss_from_features(
            labels=labels,
            actions=actions,
            dirs=dirs,
            geom=geom,
            scores=scores,
            valid=valid,
            roi_feats=roi_feats,
            seg_feat=seg_feat,
            gt_stopgo=gt_stopgo,
        )

        if 'loss_stopgo' not in losses:
            for k in ['loss_ego', 'loss_stop_go', 'loss_stop_go_ce']:
                if k in losses:
                    losses['loss_stopgo'] = losses.pop(k)
                    break
        return losses


    # -------------------------
    # total loss v2 with proposal
    # -------------------------
    def loss(self, x, rpn_results_list, data_samples, **kwargs):
        """
        New loss version:
          - detection loss still follows current bbox assign/sample flow
          - action/dir uses positive sampled proposals instead of GT boxes
          - GT is only used for action/dir target assignment
        """
        # 先保留原 detection loss 逻辑
        det_losses = super().loss(x, rpn_results_list, data_samples, **kwargs)
        self._ego_train_roi_ctx = None

        # -------------------------------------------------------
        # 重新做一遍 assign + sample，只为了拿 sampling_results
        # 不改你原有参数，不引入新 helper
        # -------------------------------------------------------
        sampling_results = []
        num_imgs = len(data_samples)

        for i in range(num_imgs):
            ds = data_samples[i]
            gt_instances = ds.gt_instances
            gt_ignore = getattr(ds, 'ignored_instances', None)

            rpn_results = rpn_results_list[i]

            # 兼容 priors / bboxes
            priors = getattr(rpn_results, 'priors', None)
            if priors is None:
                priors = getattr(rpn_results, 'bboxes', None)

            if priors is None:
                continue

            assign_result = self.bbox_assigner.assign(
                rpn_results,
                gt_instances,
                gt_ignore
            )

            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                gt_instances,
                feats=[lvl_feat[i][None] for lvl_feat in x]
            )
            sampling_results.append(sampling_result)

        # 用 proposal-based action/dir loss 替换旧的 GT-based 版本
        ad_losses = self._loss_action_dir_from_proposals(x, sampling_results, data_samples)

        # 其余保持原逻辑
        sem_losses = self._loss_semantic(x, data_samples)
        ego_losses = self._loss_ego_stopgo(x, rpn_results_list, data_samples)

        det_losses.update(ad_losses)
        det_losses.update(sem_losses)
        det_losses.update(ego_losses)
        return det_losses


    # -------------------------
    # predict (action/dir uses predicted boxes + temporal aggregator)
    # -------------------------
    def predict(self, x, rpn_results_list, data_samples, rescale: bool = False):
        logger = MMLogger.get_current_instance()

        def _get_sf4(ds, device, dtype):
            sf = ds.metainfo.get('scale_factor', None)
            if sf is not None:
                s = torch.as_tensor(sf, device=device, dtype=dtype).view(-1)
                if s.numel() >= 4:
                    return s[-4:]
            return torch.ones(4, device=device, dtype=dtype)

        inst_list = super().predict(x, rpn_results_list, data_samples, rescale=False)  # we will manually map back to ori coords
        self._ego_test_roi_ctx = None

        # seg
        seg_feat = None
        if self.semantic_head is not None:
            feats_sem = list(x[:4])

            sem_out = self.semantic_head(feats_sem)
            if isinstance(sem_out, tuple):
                logits, seg_feat = sem_out
            else:
                logits = sem_out
                seg_feat = None

            B = logits.size(0)
            for i in range(B):
                ds = data_samples[i]
                logit_i = logits[i]
                img_shape = ds.metainfo.get('img_shape', None)
                if img_shape is not None:
                    ih, iw = int(img_shape[0]), int(img_shape[1])
                    if logit_i.shape[-2:] != (ih, iw):
                        logit_i = F.interpolate(logit_i.unsqueeze(0), size=(ih, iw),
                                                mode='bilinear', align_corners=False)[0]
                mask = logit_i.argmax(dim=0).to(torch.int64)
                ds.pred_sem_seg = PixelData(data=mask.unsqueeze(0), logits=logit_i)

            if seg_feat is not None:
                for i in range(len(data_samples)):
                    data_samples[i].seg_feat = seg_feat[i].detach()


        # action/dir pred (temporal aggregator)
        if self.action_dir_head is not None:
            all_rois = []
            img_offsets = []
            n0 = 0
            for i, inst in enumerate(inst_list):
                if inst is None or (not hasattr(inst, 'bboxes')) or inst.bboxes.numel() == 0:
                    img_offsets.append((n0, n0))
                    continue
                b = inst.bboxes
                n = b.size(0)
                batch_inds = torch.full((n, 1), i, dtype=b.dtype, device=b.device)
                all_rois.append(torch.cat([batch_inds, b], dim=1))  # (n,5)
                img_offsets.append((n0, n0 + n))
                n0 += n

            if len(all_rois) > 0:
                rois_cat = torch.cat(all_rois, dim=0).contiguous()
                roi_2d = self.bbox_roi_extractor(x, rois_cat)  # (R,C,7,7)

                # temporal fuse
                roi_agg = roi_2d
                if self.temporal_aggregator is not None and self._temporal_ctx is not None:
                    feats2d_seq = self._temporal_ctx.get('feats2d_seq', None)
                    if feats2d_seq is None:
                        feats2d_seq = self._temporal_ctx.get('feat2d_seq', None)
                    img_hw = self._temporal_ctx.get('img_hw', None)
                    if feats2d_seq is not None and img_hw is not None:
                        query_token = F.adaptive_avg_pool2d(roi_2d, 1).flatten(1)
                        fused_token = self.temporal_aggregator(
                            query=query_token,
                            rois=rois_cat,
                            feat2d_seq=feats2d_seq,
                            img_hw=img_hw
                        )
                        roi_agg = self._fuse_token_to_roi_map(roi_2d, fused_token)

                with torch.no_grad():
                    pa, pd = self.action_dir_head.predict(roi_agg)  # (R,), (R,)
                    roi_feat_cache = roi_agg.detach()

                for i, inst in enumerate(inst_list):
                    s, e = img_offsets[i]
                    if s == e:
                        continue
                    inst.actions = pa[s:e].cpu()
                    inst.dirs = pd[s:e].cpu()
                    inst.roi_feats = roi_feat_cache[s:e].cpu()


        # ego pred (feature-based downstream)
        if self.ego_stopgo_head is not None:
            B = len(inst_list)
            K = int(self.ego_topk)
            device = x[0].device

            labels = torch.full((B, K), -1, dtype=torch.long, device=device)
            actions = torch.full((B, K), -1, dtype=torch.long, device=device)
            dirs = torch.full((B, K), -1, dtype=torch.long, device=device)
            geom = torch.zeros((B, K, 5), dtype=torch.float32, device=device)
            scores = torch.zeros((B, K, 1), dtype=torch.float32, device=device)
            valid = torch.zeros((B, K), dtype=torch.bool, device=device)
            roi_feats = torch.zeros((B, K, 256, 7, 7), dtype=torch.float32, device=device)

            seg_feats = []

            for bidx, (inst, ds) in enumerate(zip(inst_list, data_samples)):
                seg_f = getattr(ds, 'seg_feat', None)
                if seg_f is None:
                    seg_f = torch.zeros((256, 1, 1), dtype=torch.float32, device=device)
                seg_feats.append(seg_f.to(device=device, dtype=torch.float32))

                if inst is None:
                    continue
                if (not hasattr(inst, 'bboxes')) or inst.bboxes.numel() == 0:
                    continue
                if not hasattr(inst, 'roi_feats'):
                    continue

                bboxes = inst.bboxes.to(device=device, dtype=torch.float32)
                cls_labels = inst.labels.to(device=device, dtype=torch.long)
                det_scores = inst.scores.to(device=device, dtype=torch.float32)
                act = inst.actions.to(device=device, dtype=torch.long)
                dr = inst.dirs.to(device=device, dtype=torch.long)
                rf = inst.roi_feats.to(device=device, dtype=torch.float32)

                keep = det_scores >= float(self.ego_roi_score_thr)
                if keep.sum() == 0:
                    continue

                bboxes = bboxes[keep]
                cls_labels = cls_labels[keep]
                det_scores = det_scores[keep]
                act = act[keep]
                dr = dr[keep]
                rf = rf[keep]

                order = torch.argsort(det_scores, descending=True)
                order = order[:K]

                bboxes = bboxes[order]
                cls_labels = cls_labels[order]
                det_scores = det_scores[order]
                act = act[order]
                dr = dr[order]
                rf = rf[order]

                n = bboxes.size(0)
                if n == 0:
                    continue

                labels[bidx, :n] = cls_labels
                actions[bidx, :n] = act
                dirs[bidx, :n] = dr
                scores[bidx, :n, 0] = det_scores
                valid[bidx, :n] = True
                roi_feats[bidx, :n] = rf

                img_shape = ds.metainfo.get('img_shape', None)
                if img_shape is None:
                    pad_shape = ds.metainfo.get('pad_shape', (1, 1))
                    H = float(pad_shape[0])
                    W = float(pad_shape[1])
                else:
                    H = float(img_shape[0])
                    W = float(img_shape[1])

                x1 = bboxes[:, 0] / W
                y1 = bboxes[:, 1] / H
                x2 = bboxes[:, 2] / W
                y2 = bboxes[:, 3] / H

                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                bw = (x2 - x1).clamp(min=1e-6)
                bh = (y2 - y1).clamp(min=1e-6)
                area = bw * bh

                geom[bidx, :n] = torch.stack([cx, cy, bw, bh, area], dim=-1)

            seg_feat = torch.stack(seg_feats, dim=0)

            with torch.no_grad():
                pred, logits, prob = self.ego_stopgo_head.predict_from_features(
                    labels=labels,
                    actions=actions,
                    dirs=dirs,
                    geom=geom,
                    scores=scores,
                    valid=valid,
                    roi_feats=roi_feats,
                    seg_feat=seg_feat,
                )

            if not self._dbg_once:
                # prob = logits.softmax(dim=1)
                logger.info(
                    f"[EGO-DOWNSTREAM-DBG] p_stop={float(prob[0,0]):.4f}, "
                    f"p_go={float(prob[0,1]):.4f}, pred={int(pred[0])}"
                )
                self._dbg_once = True

            for i in range(len(data_samples)):
                data_samples[i].pred_stopgo = pred[i].detach().cpu()
                data_samples[i].pred_stopgo_logits = logits[i].detach().cpu()

        # map boxes from network input coords (after resize+center-crop/pad) back to original image coords
        # pipeline: original -> resize (scale_factor) -> center crop (crop_offset) -> network input
        # inverse: add crop_offset, then divide by scale_factor
        for i, inst in enumerate(inst_list):
            if inst is None or inst.bboxes.numel() == 0:
                continue

            ds = data_samples[i]
            device = inst.bboxes.device
            dtype = inst.bboxes.dtype

            sf4 = _get_sf4(ds, device, dtype)  # [sw, sh, sw, sh]
            sw = sf4[0].clamp(min=1e-6)
            sh = sf4[1].clamp(min=1e-6)

            # crop_offset is stored as (y0, x0) in pixels (top, left) after center-crop
            off = ds.metainfo.get('crop_offset', (0.0, 0.0))
            if off is None:
                off = (0.0, 0.0)
            # tolerate list/tuple/np/torch
            if isinstance(off, torch.Tensor):
                off = off.view(-1).detach().cpu().tolist()
            try:
                y0 = float(off[0])
                x0 = float(off[1])
            except Exception:
                y0, x0 = 0.0, 0.0
            x0 = torch.as_tensor(x0, device=device, dtype=dtype)
            y0 = torch.as_tensor(y0, device=device, dtype=dtype)

            b = inst.bboxes
            b = b.clone()
            b[:, 0] = (b[:, 0] + x0) / sw
            b[:, 2] = (b[:, 2] + x0) / sw
            b[:, 1] = (b[:, 1] + y0) / sh
            b[:, 3] = (b[:, 3] + y0) / sh
            inst.bboxes = b

        return inst_list