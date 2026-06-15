# projects/vasco/models/modules/temporal.py
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class VascoRoITemporalAgg(nn.Module):
    """RoI-level temporal aggregator.
    Inputs:
      - rois:  (N, 5) [batch_ind, x1, y1, x2, y2] on the LAST frame scale
      - feats_list: list[T] of tuple(levels) feature maps, each level: (B, C, H, W)
      - roi_extractor: mmdet RoI extractor (e.g., SingleRoIExtractor)
    Config:
      method: 'mean_conv1d' | 'mean'
      T: clip length
      detach_prev_frames: bool, True to stop grad from prev frames
      temporal_dropout: float in [0,1), prob to drop prev-frame tokens at train (keep last always)
    Output:
      - agg_feats: (N, C, 7, 7)
    """

    def __init__(self,
                 method: str = 'mean_conv1d',
                 T: int = 5,
                 detach_prev_frames: bool = True,
                 temporal_dropout: float = 0.2):
        super().__init__()
        assert method in ('mean_conv1d', 'mean')
        self.method = method
        self.T = T
        self.detach_prev_frames = detach_prev_frames
        self.temporal_dropout = float(temporal_dropout)
        # lightweight attention over time (per-channel scoring)
        self.score_conv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=3,
                                    padding=1, groups=1, bias=True)
        # 注意：roi_extractor 的输出通道通常为 256（FPN 统一维度）
        # 若你的 roi_extractor 输出通道不是 256，请在构造时将上面的 256 替换为实际通道数

    @torch.no_grad()
    def _make_keep_mask(self, n: int, t: int, device) -> torch.Tensor:
        """Train-time temporal dropout mask of shape (n, t). Keep last frame always."""
        if (not self.training) or self.temporal_dropout <= 0 or t <= 1:
            return torch.ones(n, t, dtype=torch.bool, device=device)
        keep = torch.rand(n, t, device=device) > self.temporal_dropout
        keep[:, -1] = True
        return keep

    def forward(self,
                rois: torch.Tensor,
                feats_list: List[Tuple[torch.Tensor, ...]],
                roi_extractor) -> torch.Tensor:
        assert isinstance(feats_list, (list, tuple)) and len(feats_list) >= 1
        T = len(feats_list)
        # per-frame RoIAlign
        roi_feats_per_t = []
        for ti, feats_t in enumerate(feats_list):
            if self.detach_prev_frames and ti < T - 1:
                feats_t = tuple(f.detach() for f in feats_t)
            roi_feat_t = roi_extractor(feats_t, rois)  # (N, C, 7, 7)
            roi_feats_per_t.append(roi_feat_t)
        # stack to (N, T, C, 7, 7)
        stacked = torch.stack(roi_feats_per_t, dim=1)
        N, T, C, H, W = stacked.shape
        # temporal dropout mask
        keep = self._make_keep_mask(N, T, stacked.device)  # (N, T)
        keep = keep.view(N, T, 1, 1, 1)

        if self.method == 'mean':
            masked = stacked * keep
            denom = keep.sum(dim=1).clamp_min(1.0)
            out = masked.sum(dim=1) / denom
            return out  # (N, C, 7, 7)

        # mean_conv1d: per-channel temporal scores -> softmax weights
        # token summary: GAP to (N, T, C)
        token = stacked.mean(dim=(3, 4))                      # (N, T, C)
        token = token.permute(0, 2, 1).contiguous()           # (N, C, T)
        # if C != 256, 重设 score_conv 通道数
        if token.size(1) != self.score_conv.in_channels:
            # 动态调整一次
            self.score_conv = nn.Conv1d(token.size(1), 1, kernel_size=3, padding=1, bias=True).to(token.device)
        score = self.score_conv(token).squeeze(1)             # (N, T)
        # -inf to dropped steps
        score = score.masked_fill(~keep.squeeze(-1).squeeze(-1).squeeze(-1), float('-inf'))
        weight = torch.softmax(score, dim=1)                  # (N, T)
        weight = weight.view(N, T, 1, 1, 1)
        out = (stacked * weight).sum(dim=1)                   # (N, C, 7, 7)
        return out
