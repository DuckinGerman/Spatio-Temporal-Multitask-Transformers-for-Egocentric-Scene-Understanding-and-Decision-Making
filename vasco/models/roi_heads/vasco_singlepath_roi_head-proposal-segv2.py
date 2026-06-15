# projects/vasco/models/roi_heads/vasco_singlepath_roi_head.py
# -*- coding: utf-8 -*-
from typing import Dict, Optional, List, Tuple

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

    # -------------------------
    # ctx injection (2D seq)
    # -------------------------
    def set_temporal_ctx(self, ctx: Dict):
        self._temporal_ctx = ctx

    # -------------------------
    # utility: ROI token from boxes (for ego E3)
    # -------------------------
    def _roi_token_from_boxes(self,
                             x_2d_fpn: List[torch.Tensor],
                             boxes_list: List[Optional[torch.Tensor]],
                             scores_list: Optional[List[Optional[torch.Tensor]]] = None,
                             topk: int = 10,
                             score_thr: float = 0.0) -> torch.Tensor:
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
    # (train) action/dir loss using GT rois
    # -------------------------
    # def _loss_action_dir(self, x_2d_fpn: List[torch.Tensor], data_samples: List) -> Dict[str, torch.Tensor]:
    #     if self.action_dir_head is None:
    #         z = x_2d_fpn[0].sum() * 0
    #         return dict(loss_action=z, loss_dir=z, action_acc=z, dir_acc=z)

    #     # build rois from GT boxes
    #     rois_per_img, gt_actions_all, gt_dirs_all = [], [], []
    #     total = 0
    #     for ds in data_samples:
    #         gt = ds.gt_instances
    #         if not hasattr(gt, 'bboxes') or gt.bboxes.numel() == 0:
    #             continue
    #         B = gt.bboxes  # (n,4)
    #         n = B.size(0)
    #         total += n
    #         rois_per_img.append(B)

    #         ga = getattr(gt, 'actions', None) 
    #         if ga is None:
    #             ga = getattr(gt, 'gt_actions', None)
                
    #         gd = getattr(gt, 'dirs', None)
    #         if gd is None:
    #             gd = getattr(gt, 'gt_dirs', None)
    #         if gd is None:
    #             gd = getattr(gt, 'directions', None)
    #         if gd is None:
    #             gd = getattr(gt, 'gt_directions', None)


    #         # Ensure we NEVER emit -1 for action/dir.
    #         # Action raw ids are {0,1,2,3} plus fallback 4(noaction/invalid).
    #         # Direction ids are {0..4} plus fallback 5(nodirection/invalid).
    #         if ga is None:
    #             ga = torch.full((n,), 4, dtype=torch.long, device=B.device)
    #         else:
    #             ga = ga.to(device=B.device, dtype=torch.long).view(-1)
    #             # map -1 / invalid -> 4
    #             valid_ga = (ga >= 0) & (ga <= 4)
    #             ga = torch.where(valid_ga, ga, ga.new_full(ga.shape, 4))

    #         if gd is None:
    #             gd = torch.full((n,), 5, dtype=torch.long, device=B.device)
    #         else:
    #             gd = gd.to(device=B.device, dtype=torch.long).view(-1)
    #             # map -1 / invalid -> 5
    #             valid_gd = (gd >= 0) & (gd < 6)
    #             gd = torch.where(valid_gd, gd, gd.new_full(gd.shape, 5))

    #         gt_actions_all.append(ga)
    #         gt_dirs_all.append(gd)

    #     if total == 0:
    #         z = x_2d_fpn[0].sum() * 0
    #         return dict(loss_action=z, loss_dir=z, action_acc=z, dir_acc=z)

    #     rois = bbox2roi(rois_per_img).to(dtype=x_2d_fpn[0].dtype).contiguous()  # (R,5)
    #     gt_actions = torch.cat(gt_actions_all, dim=0).to(x_2d_fpn[0].device)
    #     gt_dirs = torch.cat(gt_dirs_all, dim=0).to(x_2d_fpn[0].device)

    #     # last-frame ROI feature map
    #     roi_2d = self.bbox_roi_extractor(x_2d_fpn, rois)  # (R,C,7,7)
    #     if self.aux_detach:
    #         roi_2d = roi_2d.detach()

    #     # temporal aggregation (MS-DeformAttn) => token (R,C) then fuse back
    #     if self.temporal_aggregator is not None and self._temporal_ctx is not None:
    #         feats2d_seq = self._temporal_ctx.get('feats2d_seq', None)
    #         if feats2d_seq is None:
    #             feats2d_seq = self._temporal_ctx.get('feat2d_seq', None)
    #         img_hw = self._temporal_ctx.get('img_hw', None)
    #         if feats2d_seq is not None and img_hw is not None:
    #             query_token = F.adaptive_avg_pool2d(roi_2d, 1).flatten(1)  # (R,C)
    #             fused_token = self.temporal_aggregator(
    #                 query=query_token,
    #                 rois=rois,
    #                 feat2d_seq=feats2d_seq,
    #                 img_hw=img_hw
    #             )  # (R,C)
    #             roi_agg = self._fuse_token_to_roi_map(roi_2d, fused_token)
    #         else:
    #             roi_agg = roi_2d
    #     else:
    #         roi_agg = roi_2d

    #     # action/dir head
    #     logits_action, logits_dir = self.action_dir_head(roi_agg)
    #     losses = self.action_dir_head.loss(logits_action, logits_dir, gt_actions, gt_dirs)
    #     return losses
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

        logits_action, logits_dir = self.action_dir_head(roi_agg)
        losses = self.action_dir_head.loss(logits_action, logits_dir, gt_actions, gt_dirs)
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

        # fallback (same as your current style)
        # logits = self.semantic_head(x_2d_fpn[:4])
        # gts = []
        # for ds in data_samples:
        #     gt_pd = getattr(ds, 'gt_sem_seg', None)
        #     if gt_pd is None:
        #         z = logits.sum() * 0
        #         return dict(loss_sem_ce=z, loss_sem_dice=z)
        #     gt = gt_pd.data if hasattr(gt_pd, 'data') else gt_pd
        #     if gt.dim() == 3 and gt.size(0) == 1:
        #         gt = gt[0]
        #     gts.append(gt.to(logits.device).long())
        # gt = torch.stack(gts, dim=0)
        # if gt.shape[-2:] != logits.shape[-2:]:
        #     gt = F.interpolate(gt.unsqueeze(1).float(), size=logits.shape[-2:], mode='nearest').squeeze(1).long()

        # ce = F.cross_entropy(logits, gt, ignore_index=255)
        # probs = logits.softmax(dim=1)
        # valid = (gt != 255)
        # if valid.sum() > 0:
        #     t = gt.clone(); t[~valid] = 0
        #     onehot = F.one_hot(t, num_classes=logits.size(1)).permute(0, 3, 1, 2).to(probs.dtype)
        #     probs = probs * valid.unsqueeze(1)
        #     onehot = onehot * valid.unsqueeze(1)
        #     num = 2 * (probs * onehot).sum(dim=(0, 2, 3))
        #     den = (probs * probs).sum(dim=(0, 2, 3)) + (onehot * onehot).sum(dim=(0, 2, 3)) + 1e-6
        #     dice = (1.0 - num / den).mean()
        # else:
        #     dice = ce.new_zeros(())
        return dict(loss_sem_ce=ce, loss_sem_dice=0.5 * dice)

    # -------------------------
    # (train) ego stop/go loss (same as your dualpath version)
    # -------------------------
    def _loss_ego_stopgo(self, x_2d_fpn: List[torch.Tensor], rpn_results_list, data_samples: List) -> Dict[str, torch.Tensor]:
        if self.ego_stopgo_head is None:
            z = x_2d_fpn[0].sum() * 0
            return dict(loss_stopgo=z, stopgo_acc=z)

        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in x_2d_fpn]
        feat_global = torch.cat(pooled, dim=1)  # (B, 256*num_outs)
        device = feat_global.device

        # E3: RoI token from RPN proposals during training
        if self.ego_roi_use_rpn_in_train and rpn_results_list is not None:
            boxes_list, scores_list = [], []
            for r in rpn_results_list:
                boxes_list.append(r.get('bboxes', None) if isinstance(r, dict) else getattr(r, 'bboxes', None))
                scores_list.append(r.get('scores', None) if isinstance(r, dict) else getattr(r, 'scores', None))
            roi_tok = self._roi_token_from_boxes(
                x_2d_fpn,
                boxes_list=boxes_list,
                scores_list=scores_list,
                topk=self.ego_topk,
                score_thr=self.ego_roi_score_thr,
            )
        else:
            roi_tok = x_2d_fpn[0].new_zeros((feat_global.size(0), x_2d_fpn[0].size(1)))

        feat = torch.cat([feat_global, roi_tok], dim=1)

        ego_ids, gts = [], []
        for ds in data_samples:
            ego = getattr(ds, 'ego_choice_id', None)
            gt = getattr(ds, 'gt_stopgo', None)
            if ego is None or gt is None:
                z = feat_global.sum() * 0
                return dict(loss_stopgo=z, stopgo_acc=z)
            if not isinstance(ego, torch.Tensor):
                ego = torch.tensor(ego, device=device)
            if not isinstance(gt, torch.Tensor):
                gt = torch.tensor(gt, device=device)
            ego_ids.append(ego.to(device).view(1))
            gts.append(gt.to(device).view(1))

        ego_ids = torch.cat(ego_ids, dim=0).long()
        gt_stopgo = torch.cat(gts, dim=0).long()

        logits = self.ego_stopgo_head(feat, ego_ids)
        losses = self.ego_stopgo_head.loss(logits, gt_stopgo)

        if 'loss_stopgo' not in losses:
            for k in ['loss_ego', 'loss_stop_go', 'loss_stop_go_ce']:
                if k in losses:
                    losses['loss_stopgo'] = losses.pop(k)
                    break
        if 'stopgo_acc' not in losses:
            pred = logits.argmax(dim=1)
            losses['stopgo_acc'] = (pred == gt_stopgo).float().mean()
        return losses

    # -------------------------
    # total loss v1 with gt
    # -------------------------
    # def loss(self, x, rpn_results_list, data_samples, **kwargs):
    #     det_losses = super().loss(x, rpn_results_list, data_samples, **kwargs)
    #     ad_losses = self._loss_action_dir(x, data_samples)
    #     sem_losses = self._loss_semantic(x, data_samples)
    #     ego_losses = self._loss_ego_stopgo(x, rpn_results_list, data_samples)
    #     det_losses.update(ad_losses)
    #     det_losses.update(sem_losses)
    #     det_losses.update(ego_losses)
    #     return det_losses

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

        # seg
        if self.semantic_head is not None:
            feats_sem = list(x[:4])
            logits = self.semantic_head(feats_sem)  # (B,C,H,W)
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

        # ego pred (same as your dualpath predict)
        if self.ego_stopgo_head is not None:
            pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in x]
            feat_global = torch.cat(pooled, dim=1) # (B, 256*num_outs)
            B = feat_global.size(0)

            boxes_list, scores_list = [], []
            for inst in inst_list:
                if inst is None or not hasattr(inst, 'bboxes') or inst.bboxes.numel() == 0:
                    boxes_list.append(None); scores_list.append(None)
                else:
                    boxes_list.append(inst.bboxes)
                    scores_list.append(getattr(inst, 'scores', None))

            roi_tok = self._roi_token_from_boxes(
                x, boxes_list=boxes_list, scores_list=scores_list,
                topk=self.ego_topk, score_thr=self.ego_roi_score_thr
            )
            feat = torch.cat([feat_global, roi_tok], dim=1)

            ego_ids = []
            for i in range(B):
                ego = getattr(data_samples[i], 'ego_choice_id', None)
                if ego is None:
                    ego_t = torch.zeros((1,), device=feat.device, dtype=torch.long)
                else:
                    if not isinstance(ego, torch.Tensor):
                        ego = torch.tensor(ego, device=feat.device)
                    ego_t = ego.to(feat.device).view(1).long()
                ego_ids.append(ego_t)
            ego_ids = torch.cat(ego_ids, dim=0)

            with torch.no_grad():
                logits = self.ego_stopgo_head(feat, ego_ids)
                prob = logits.softmax(dim=1)
                pred = prob.argmax(dim=1)

            # optional small debug
            if not self._dbg_once:
                logger.info(f"[EGO-PRED-DBG] p_stop={float(prob[0,0]):.4f}, p_go={float(prob[0,1]):.4f}, pred={int(pred[0])}")
                self._dbg_once = True

            for i in range(B):
                data_samples[i].pred_stopgo = pred[i].detach().cpu()
                data_samples[i].pred_stopgo_logits = logits[i].detach().cpu()

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

                for i, inst in enumerate(inst_list):
                    s, e = img_offsets[i]
                    if s == e:
                        continue
                    inst.actions = pa[s:e].cpu()
                    inst.dirs = pd[s:e].cpu()

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