# -*- coding: utf-8 -*-
# projects/vasco/models/ego_heads/ego_stopgo_head.py

from typing import Dict, Optional, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS



@MODELS.register_module()
class EgoStopGoHead(nn.Module):
    """
    Downstream ego(stop/go) head.

    Inputs:
      - labels:    (B, K) long, detection class ids padded with -1
      - actions:   (B, K) long, action ids padded with -1
      - dirs:      (B, K) long, direction ids padded with -1
      - geom:      (B, K, 5) float, normalized [cx, cy, w, h, area]
      - scores:    (B, K, 1) float, detection scores
      - valid:     (B, K) bool, valid object mask
      - roi_feats: (B, K, 256, 7, 7) or (B, K, 256), instance-level RoI features
      - seg_feat:  (B, 256, H, W) or (B, 256), semantic fusion feature

    Output:
      - logits: (B, 2) [stop, go]
    """

    def __init__(self,
                 num_det_classes: int = 29,
                 num_action_classes: int = 5,
                 num_dir_classes: int = 6,
                 ego_num_choices: int = 5,
                 class_emb_dim: int = 32,
                 action_emb_dim: int = 16,
                 dir_emb_dim: int = 16,
                 seg_in_dim: int = 32,
                 seg_proj_dim: int = 32,
                 choice_emb_dim: int = 16,
                 hidden_dim: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.0,
                 loss: Optional[Dict] = None,
                 class_weight: Optional[List[float]] = None,
                 ignore_index: int = -1,
                 label_smoothing: float = 0.0,
                 use_focal_ce: bool = False,
                 focal_gamma: float = 2.0,
                 roi_feat_dim: int = 256,
                 obj_proj_dim: int = 256,
                 seg_feat_dim: int = 256,
                 seg_feat_proj_dim: int = 256,
                 light_class_ids: Optional[List[int]] = None,
                 use_choice: bool = False,
                 use_ad: bool = False,
                 use_light_branch: bool = False,
                 use_seg_feat: bool = False,
                 ):
        super().__init__()
        self.num_det_classes = int(num_det_classes)
        self.num_action_classes = int(num_action_classes)
        self.num_dir_classes = int(num_dir_classes)
        self.ego_num_choices = int(ego_num_choices)

        self.class_emb_dim = int(class_emb_dim)
        self.action_emb_dim = int(action_emb_dim)
        self.dir_emb_dim = int(dir_emb_dim)
        self.seg_in_dim = int(seg_in_dim)
        self.seg_proj_dim = int(seg_proj_dim)
        self.choice_emb_dim = int(choice_emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        self.label_smoothing = float(label_smoothing)
        self.use_focal_ce = bool(use_focal_ce)
        self.focal_gamma = float(focal_gamma)
        self.class_weight = class_weight
        self.ignore_index = int(ignore_index)

        # Feature-fusion settings used by the feature-based ego head.
        self.roi_feat_dim = int(roi_feat_dim)
        self.obj_proj_dim = int(obj_proj_dim)
        self.seg_feat_dim = int(seg_feat_dim)
        self.seg_feat_proj_dim = int(seg_feat_proj_dim)
        self.use_choice = bool(use_choice)
        self.use_ad = bool(use_ad)
        self.use_light_branch = bool(use_light_branch)
        self.use_seg_feat = bool(use_seg_feat)

        self.light_class_ids = [] if light_class_ids is None else [int(x) for x in light_class_ids]

        # +1 because padded invalid ids are mapped to the last embedding entry
        self.class_emb = nn.Embedding(self.num_det_classes + 1, self.class_emb_dim)
        self.action_emb = nn.Embedding(self.num_action_classes + 1, self.action_emb_dim)
        self.dir_emb = nn.Embedding(self.num_dir_classes + 1, self.dir_emb_dim)
        self.choice_emb = nn.Embedding(self.ego_num_choices, self.choice_emb_dim)

        # ---- feature-based ego head ----
        # Each object token keeps high-dimensional ROI appearance features and augments them
        # with explicit instance-level metadata. Action and direction are part of the object token
        # because they are attributes of the detected instance.
        # Keep object token dimensionality fixed for ablation.
        # When use_ad=False, action/dir embeddings are zeroed in forward(),
        # but the projection layer input size remains unchanged.
        self.obj_raw_dim = (
            self.roi_feat_dim
            + self.class_emb_dim
            + self.action_emb_dim
            + self.dir_emb_dim
            + 6  # geom(5) + score(1)
        )
        self.obj_proj = nn.Sequential(
            nn.Linear(self.obj_raw_dim, self.obj_proj_dim),
            nn.LayerNorm(self.obj_proj_dim),
            nn.ReLU(inplace=True),
        )
        self.obj_attn = nn.Linear(self.obj_proj_dim, 1)

        # Traffic-light branch uses the same object token representation but only pools
        # over selected light classes. If no light is detected, it returns a zero vector
        # and has_light=0.
        self.light_attn = nn.Linear(self.obj_proj_dim, 1)

        # Segmentation branch consumes the 256-channel semantic fusion feature from SemanticHead,
        # not the final 7-class logits/mask.
        self.seg_feat_proj = nn.Sequential(
            nn.Linear(self.seg_feat_dim, self.seg_feat_proj_dim),
            nn.LayerNorm(self.seg_feat_proj_dim),
            nn.ReLU(inplace=True),
        )

        # Keep fusion dimensionality fixed for ablation.
        # Disabled branches are zeroed in forward(), not removed.
        fusion_dim = self.obj_proj_dim + self.obj_proj_dim + self.seg_feat_proj_dim + 1
        if self.use_choice:
            fusion_dim += self.choice_emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        self.loss_fn = MODELS.build(loss)


    def _safe_embed_ids(self,
                        x: torch.Tensor,
                        num_valid: int,
                        invalid_fill: int) -> torch.Tensor:
        """Map padded invalid ids (<0 or >=num_valid) to a safe embedding index."""
        x = x.long()
        valid = (x >= 0) & (x < num_valid)
        out = torch.where(valid, x, x.new_full(x.shape, invalid_fill))
        return out

    def _masked_mean(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: (B,K,C), valid: (B,K) -> (B,C)"""
        m = valid.unsqueeze(-1).to(dtype=x.dtype)
        num = (x * m).sum(dim=1)
        den = m.sum(dim=1).clamp(min=1.0)
        return num / den

    def _masked_attn_pool(self, x: torch.Tensor, valid: torch.Tensor, attn: nn.Module) -> torch.Tensor:
        """x: (B,K,C), valid: (B,K) -> (B,C). Returns zero vector if no valid token."""
        assert x.dim() == 3, f"x must be (B,K,C), got {x.shape}"
        assert valid.dim() == 2, f"valid must be (B,K), got {valid.shape}"

        B, K, C = x.shape
        valid = valid.to(device=x.device, dtype=torch.bool)
        score = attn(x).squeeze(-1)  # (B,K)
        score = score.masked_fill(~valid, -1e4)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)  # (B,K,1)
        pooled = (x * weight).sum(dim=1)

        has_any = valid.any(dim=1, keepdim=True).to(dtype=x.dtype)
        pooled = pooled * has_any
        return pooled

    def _build_light_mask(self, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Return valid mask for traffic-light tokens. If no light class ids are configured, all false."""
        valid = valid.to(dtype=torch.bool)
        if len(self.light_class_ids) == 0:
            return torch.zeros_like(valid, dtype=torch.bool)

        light_mask = torch.zeros_like(valid, dtype=torch.bool)
        for cid in self.light_class_ids:
            light_mask = light_mask | (labels == int(cid))
        return valid & light_mask
    

    @torch.no_grad()
    def _macro_f1_binary(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: (B,) long, values in {0,1}
        returns: scalar tensor
        """
        f1s = []
        eps = gt.new_tensor(1e-9, dtype=torch.float32)
        for c in (0, 1):
            tp = ((pred == c) & (gt == c)).sum().float()
            fp = ((pred == c) & (gt != c)).sum().float()
            fn = ((pred != c) & (gt == c)).sum().float()
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            f1s.append(f1)
        return (f1s[0] + f1s[1]) * 0.5

    def _weighted_ce_or_focal(self, logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Weighted CE or focal CE for binary stop/go classification."""
        weight = None
        if self.class_weight is not None:
            if isinstance(self.class_weight, (list, tuple)):
                weight = logits.new_tensor(self.class_weight, dtype=torch.float32)
            elif isinstance(self.class_weight, torch.Tensor):
                weight = self.class_weight.to(device=logits.device, dtype=torch.float32)

        ce = F.cross_entropy(
            logits,
            gt,
            weight=weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )

        if not self.use_focal_ce:
            return ce.mean()

        pt = torch.softmax(logits, dim=1).gather(1, gt.view(-1, 1)).squeeze(1)
        pt = pt.clamp(min=1e-6, max=1.0)
        focal = (1.0 - pt).pow(self.focal_gamma)
        return (focal * ce).mean()

    def loss(self, logits: torch.Tensor, gt_stopgo: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute downstream ego stop/go loss + metrics."""
        if gt_stopgo.dim() == 2 and gt_stopgo.size(1) == 1:
            gt_stopgo = gt_stopgo.view(-1)
        gt_stopgo = gt_stopgo.to(device=logits.device, dtype=torch.long)

        # mask valid labels
        valid = (gt_stopgo != self.ignore_index)
        if valid.sum() == 0:
            # keep graph valid: return zero loss and neutral metrics
            z = logits.sum() * 0.0
            return dict(loss_stopgo=z, stopgo_acc=z, stopgo_mF1=z,stopgo_f1_stop=z,
            stopgo_f1_go=z)

        logits_v = logits[valid]
        gt_v = gt_stopgo[valid]

        # ---- loss ----
        # Prefer explicit weighted CE / focal CE here so imbalance handling is fully controlled.
        loss_stopgo = self._weighted_ce_or_focal(logits_v, gt_v)

        with torch.no_grad():
            pred = logits_v.argmax(dim=1)
            acc = (pred == gt_v).float().mean()

            eps = gt_v.new_tensor(1e-9, dtype=torch.float32)

            # stop = 0
            tp0 = ((pred == 0) & (gt_v == 0)).sum().float()
            fp0 = ((pred == 0) & (gt_v != 0)).sum().float()
            fn0 = ((pred != 0) & (gt_v == 0)).sum().float()
            p0 = tp0 / (tp0 + fp0 + eps)
            r0 = tp0 / (tp0 + fn0 + eps)
            f1_stop = 2 * p0 * r0 / (p0 + r0 + eps)

            # go = 1
            tp1 = ((pred == 1) & (gt_v == 1)).sum().float()
            fp1 = ((pred == 1) & (gt_v != 1)).sum().float()
            fn1 = ((pred != 1) & (gt_v == 1)).sum().float()
            p1 = tp1 / (tp1 + fp1 + eps)
            r1 = tp1 / (tp1 + fn1 + eps)
            f1_go = 2 * p1 * r1 / (p1 + r1 + eps)

            mf1 = 0.5 * (f1_stop + f1_go)

        return dict(
            loss_stopgo=loss_stopgo,
            stopgo_acc=acc,
            stopgo_mF1=mf1,
            stopgo_f1_stop=f1_stop,
            stopgo_f1_go=f1_go,
        )
    def forward(self,
                labels: torch.Tensor,
                actions: torch.Tensor,
                dirs: torch.Tensor,
                geom: torch.Tensor,
                scores: torch.Tensor,
                valid: torch.Tensor,
                roi_feats: torch.Tensor,
                seg_feat: torch.Tensor,
                ego_choice_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Feature-based forward for downstream ego stop/go.

        Args:
            labels/actions/dirs: (B,K) padded with -1
            geom:               (B,K,5), normalized [cx,cy,w,h,area]
            scores:             (B,K,1)
            valid:              (B,K)
            roi_feats:           (B,K,256,7,7) or (B,K,256)
            seg_feat:            (B,256,H,W) or (B,256)
            ego_choice_id:       optional (B,), only used when self.use_choice=True

        Returns:
            logits: (B,2)
        """
        assert labels.dim() == 2, f"labels must be (B,K), got {labels.shape}"
        assert actions.dim() == 2, f"actions must be (B,K), got {actions.shape}"
        assert dirs.dim() == 2, f"dirs must be (B,K), got {dirs.shape}"
        assert geom.dim() == 3 and geom.size(-1) == 5, f"geom must be (B,K,5), got {geom.shape}"
        assert scores.dim() == 3 and scores.size(-1) == 1, f"scores must be (B,K,1), got {scores.shape}"
        assert valid.dim() == 2, f"valid must be (B,K), got {valid.shape}"

        device = geom.device
        labels = labels.to(device=device)
        actions = actions.to(device=device)
        dirs = dirs.to(device=device)
        geom = geom.to(device=device, dtype=torch.float32)
        scores = scores.to(device=device, dtype=torch.float32)
        valid = valid.to(device=device, dtype=torch.bool)
        roi_feats = roi_feats.to(device=device, dtype=torch.float32)
        seg_feat = seg_feat.to(device=device, dtype=torch.float32)

        if roi_feats.dim() == 5:
            # (B,K,C,7,7) -> (B,K,C)
            roi_vec = F.adaptive_avg_pool2d(roi_feats.flatten(0, 1), 1).flatten(1)
            roi_vec = roi_vec.view(labels.size(0), labels.size(1), -1)
        elif roi_feats.dim() == 3:
            roi_vec = roi_feats
        else:
            raise AssertionError(f"roi_feats must be (B,K,C,7,7) or (B,K,C), got {roi_feats.shape}")

        if roi_vec.size(-1) != self.roi_feat_dim:
            raise AssertionError(f"Expected roi feature dim {self.roi_feat_dim}, got {roi_vec.size(-1)}")

        labels_safe = self._safe_embed_ids(labels, self.num_det_classes, self.num_det_classes)
        actions_safe = self._safe_embed_ids(actions, self.num_action_classes, self.num_action_classes)
        dirs_safe = self._safe_embed_ids(dirs, self.num_dir_classes, self.num_dir_classes)

        class_tok = self.class_emb(labels_safe)
        action_tok = self.action_emb(actions_safe)
        dir_tok = self.dir_emb(dirs_safe)

        if not self.use_ad:
            action_tok = torch.zeros_like(action_tok)
            dir_tok = torch.zeros_like(dir_tok)

        obj_raw_parts = [roi_vec, class_tok, action_tok, dir_tok, geom, scores]
        obj_raw = torch.cat(obj_raw_parts, dim=-1)
        
        obj_tok = self.obj_proj(obj_raw)  # (B,K,obj_proj_dim)

        obj_feat = self._masked_attn_pool(obj_tok, valid, self.obj_attn)

        light_mask = self._build_light_mask(labels, valid)
        light_feat = self._masked_attn_pool(obj_tok, light_mask, self.light_attn)
        has_light = light_mask.any(dim=1, keepdim=True).to(dtype=obj_feat.dtype)

        if seg_feat.dim() == 4:
            seg_vec = F.adaptive_avg_pool2d(seg_feat, 1).flatten(1)
        elif seg_feat.dim() == 2:
            seg_vec = seg_feat
        else:
            raise AssertionError(f"seg_feat must be (B,C,H,W) or (B,C), got {seg_feat.shape}")

        if seg_vec.size(-1) != self.seg_feat_dim:
            raise AssertionError(f"Expected seg feature dim {self.seg_feat_dim}, got {seg_vec.size(-1)}")
        seg_vec = self.seg_feat_proj(seg_vec)

        if not self.use_light_branch:
            light_feat = torch.zeros_like(light_feat)
            has_light = torch.zeros_like(has_light)

        if not self.use_seg_feat:
            seg_vec = torch.zeros_like(seg_vec)

        # Fixed fusion layout for fair ablation:
        # [object feature, light feature, segmentation feature, has_light flag]
        parts = [obj_feat, light_feat, seg_vec, has_light]

        if self.use_choice:
            if ego_choice_id is None:
                ego_choice_id = torch.zeros((labels.size(0),), device=device, dtype=torch.long)
            if ego_choice_id.dim() == 2 and ego_choice_id.size(1) == 1:
                ego_choice_id = ego_choice_id.view(-1)
            choice_tok = self.choice_emb(ego_choice_id.to(device=device, dtype=torch.long))
            parts.append(choice_tok)

        x = torch.cat(parts, dim=-1)
        logits = self.mlp(x)
        return logits

    def loss_from_features(self,
                           labels: torch.Tensor,
                           actions: torch.Tensor,
                           dirs: torch.Tensor,
                           geom: torch.Tensor,
                           scores: torch.Tensor,
                           valid: torch.Tensor,
                           roi_feats: torch.Tensor,
                           seg_feat: torch.Tensor,
                           gt_stopgo: torch.Tensor,
                           ego_choice_id: Optional[torch.Tensor] = None):
        logits = self.forward(
            labels=labels,
            actions=actions,
            dirs=dirs,
            geom=geom,
            scores=scores,
            valid=valid,
            roi_feats=roi_feats,
            seg_feat=seg_feat,
            ego_choice_id=ego_choice_id,
        )
        return self.loss(logits, gt_stopgo)

    def predict_from_features(self,
                              labels: torch.Tensor,
                              actions: torch.Tensor,
                              dirs: torch.Tensor,
                              geom: torch.Tensor,
                              scores: torch.Tensor,
                              valid: torch.Tensor,
                              roi_feats: torch.Tensor,
                              seg_feat: torch.Tensor,
                              ego_choice_id: Optional[torch.Tensor] = None):
        logits = self.forward(
            labels=labels,
            actions=actions,
            dirs=dirs,
            geom=geom,
            scores=scores,
            valid=valid,
            roi_feats=roi_feats,
            seg_feat=seg_feat,
            ego_choice_id=ego_choice_id,
        )
        prob = logits.softmax(dim=1)
        pred = prob.argmax(dim=1)
        return pred, logits, prob