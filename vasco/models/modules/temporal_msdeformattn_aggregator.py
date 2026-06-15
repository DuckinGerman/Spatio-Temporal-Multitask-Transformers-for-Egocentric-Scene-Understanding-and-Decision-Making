# projects/vasco/models/temporal_aggregators/temporal_msdeformattn_aggregator.py
import math
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

try:
    # true MS-DeformAttn (CUDA op) from MMCV
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except Exception as e:
    MultiScaleDeformableAttention = None
    _MSDA_IMPORT_ERR = e


def _build_level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
    # spatial_shapes: (L, 2) -> (L,) prefix sum of H*W
    hw = spatial_shapes[:, 0] * spatial_shapes[:, 1]  # (L,)
    start = torch.zeros_like(hw)
    start[1:] = torch.cumsum(hw, dim=0)[:-1]
    return start


def _roi_centers_norm(rois: torch.Tensor, img_hw: Tuple[int, int]) -> torch.Tensor:
    """rois: (N, 5) = [batch_idx, x1, y1, x2, y2] in input image pixel coords
       return: (N, 2) normalized center in [0,1] wrt img size (W,H)
    """
    _, x1, y1, x2, y2 = rois.unbind(dim=1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    H, W = img_hw
    cx = cx / max(float(W), 1.0)
    cy = cy / max(float(H), 1.0)
    return torch.stack([cx, cy], dim=-1).clamp(0, 1)


def _flatten_feats_time_as_levels(
    feat2d_seq: List[torch.Tensor],
    time_embed: Optional[nn.Embedding] = None,
    level_embed: Optional[nn.Embedding] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    feat2d_seq: list[L], each is (B, T, C, H_l, W_l)
    Treat (t,l) as a "level": total levels = T*L.
    Return:
      value: (B, sum_{t,l}(H_l*W_l), C)
      spatial_shapes: (T*L, 2) with (H_l, W_l)
      level_start_index: (T*L,)
    """
    assert isinstance(feat2d_seq, (list, tuple)) and len(feat2d_seq) > 0
    B, T, C = feat2d_seq[0].shape[:3]
    device = feat2d_seq[0].device

    spatial_shapes = []
    value_chunks = []
    lvl_id = 0

    for t in range(T):
        for l, x in enumerate(feat2d_seq):
            # x: (B,T,C,H,W) -> xt: (B,C,H,W)
            xt = x[:, t]  # (B,C,H,W)
            _, _, H, W = xt.shape
            spatial_shapes.append([H, W])

            # (B,C,H,W) -> (B, H*W, C)
            v = xt.flatten(2).transpose(1, 2).contiguous()

            # add embeddings to memory (optional but recommended)
            if level_embed is not None:
                v = v + level_embed.weight[lvl_id].view(1, 1, C)
            if time_embed is not None:
                v = v + time_embed.weight[t].view(1, 1, C)

            value_chunks.append(v)
            lvl_id += 1

    spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=device)  # (T*L,2)
    value = torch.cat(value_chunks, dim=1)  # (B, sum(HW), C)
    level_start_index = _build_level_start_index(spatial_shapes)  # (T*L,)

    return value, spatial_shapes, level_start_index


@MODELS.register_module()
class TemporalMSDeformAttnAggregator(nn.Module):
    """
    Proposal-conditioned temporal aggregator using TRUE MultiScaleDeformableAttention.

    Input:
      - query: (N, C) or (B, N, C) object tokens (typically from last-frame RoI features)
      - rois: (N, 5) [batch, x1,y1,x2,y2] in image pixel coords (last frame proposals)
      - feat2d_seq: list[L] each (B, T, C, H_l, W_l) from your 2D backbone+FPN on ALL frames

    Output:
      - fused query tokens: same shape as query (N,C) or (B,N,C)

    Notes:
      - We treat time as extra "levels": total n_levels = T*L.
      - reference_points are ROI centers normalized to [0,1], shared across all levels.
      - This module is for action/dir/ego (temporal evidence); det/seg remain last-frame.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 32,   # will be overwritten at runtime by (T*L), but used for embeddings init
        num_points: int = 4,
        dropout: float = 0.0,
        with_ffn: bool = True,
        ffn_hidden: int = 1024,
        time_max_len: int = 32,     # max T
        level_max_len: int = 64,    # max (T*L)
        use_pos_in_query: bool = True,
        query_pos_dim: int = 256,
    ):
        super().__init__()

        if MultiScaleDeformableAttention is None:
            raise ImportError(
                "mmcv.ops.MultiScaleDeformableAttention is not available. "
                "You need MMCV with compiled ops. Original import error:\n"
                f"{_MSDA_IMPORT_ERR}"
            )

        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        self.dropout = float(dropout)
        self.with_ffn = bool(with_ffn)

        # true MSDeformAttn cross-attn
        self.msda = MultiScaleDeformableAttention(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            num_levels=num_levels,      # runtime shapes must match; we ensure by not using internal level params besides projection
            num_points=self.num_points,
            dropout=self.dropout,
            batch_first=True
        )

        # memory embeddings (recommended to prevent "only last frame" collapse)
        self.time_embed = nn.Embedding(time_max_len, self.embed_dims)
        self.level_embed = nn.Embedding(level_max_len, self.embed_dims)

        # optional query positional embedding from ROI center (helps)
        self.use_pos_in_query = bool(use_pos_in_query)
        if self.use_pos_in_query:
            self.query_pos = nn.Sequential(
                nn.Linear(2, query_pos_dim),
                nn.ReLU(inplace=True),
                nn.Linear(query_pos_dim, self.embed_dims),
            )

        self.norm1 = nn.LayerNorm(self.embed_dims)
        self.drop1 = nn.Dropout(self.dropout)

        if self.with_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(self.embed_dims, ffn_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(ffn_hidden, self.embed_dims),
                nn.Dropout(self.dropout),
            )
            self.norm2 = nn.LayerNorm(self.embed_dims)

        self._dbg_once = False

    def forward(
        self,
        query: torch.Tensor,
        rois: torch.Tensor,
        feat2d_seq: List[torch.Tensor],
        img_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        query: (N,C) or (B,N,C)
        rois:  (N,5) [b,x1,y1,x2,y2]
        feat2d_seq: list[L] each (B,T,C,H,W)
        img_hw: (H_img, W_img) of the (resized/padded) network input that rois correspond to
        """
        assert rois.dim() == 2 and rois.size(1) == 5, f"rois must be (N,5), got {tuple(rois.shape)}"
        assert isinstance(feat2d_seq, (list, tuple)) and len(feat2d_seq) > 0

        B, T, C = feat2d_seq[0].shape[:3]
        assert C == self.embed_dims, f"feat channels C={C} must match embed_dims={self.embed_dims}"

        # ---- query shape -> (B, N, C) ----
        if query.dim() == 2:
            N = query.size(0)
            query_b = query.view(1, N, C).expand(B, N, C).contiguous()
        elif query.dim() == 3:
            query_b = query
            N = query.size(1)
        else:
            raise AssertionError(f"query must be (N,C) or (B,N,C), got {tuple(query.shape)}")

        # ---- build memory (B, S, C) with time-as-levels ----
        # total levels = T * L
        total_levels = T * len(feat2d_seq)
        if total_levels > self.level_embed.num_embeddings:
            raise ValueError(f"Need level_max_len >= T*L. Got T*L={total_levels}, level_max_len={self.level_embed.num_embeddings}")
        if T > self.time_embed.num_embeddings:
            raise ValueError(f"Need time_max_len >= T. Got T={T}, time_max_len={self.time_embed.num_embeddings}")

        value, spatial_shapes, level_start_index = _flatten_feats_time_as_levels(
            feat2d_seq, time_embed=self.time_embed, level_embed=self.level_embed
        )
        # value: (B,S,C), spatial_shapes: (Lv,2), level_start_index: (Lv,)

        # ---- reference points: ROI centers normalized ----
        # per-roi normalized centers in [0,1], then expand to (B,N,Lv,2)
        centers = _roi_centers_norm(rois, img_hw=img_hw).to(value.device)  # (N,2)
        # batch expand: rois contains batch_idx; we need ref per batch. simplest:
        # create (B,N,2) filled by centers where roi.batch==b, else dummy (0,0).
        ref_bn = torch.zeros((B, N, 2), device=value.device, dtype=value.dtype)
        b_inds = rois[:, 0].long().clamp(0, B - 1)
        for b in range(B):
            mask = (b_inds == b)
            if mask.any():
                ref_bn[b, mask] = centers[mask].to(value.dtype)

        reference_points = ref_bn.unsqueeze(2).repeat(1, 1, spatial_shapes.size(0), 1)  # (B,N,Lv,2)

        # ---- optional query pos from ROI center ----
        q = query_b
        if self.use_pos_in_query:
            qpos = self.query_pos(ref_bn)  # (B,N,C)
            q = q + qpos

        # ---- MSDeformAttn cross-attn ----
        # MMCV expects:
        #   query: (B, Len_q, C)
        #   value: (B, Len_v, C)
        #   reference_points: (B, Len_q, num_levels, 2)
        #   spatial_shapes: (num_levels, 2)
        #   level_start_index: (num_levels,)
        out = self.msda(
            query=q,
            value=value,
            identity=None,
            query_pos=None,
            key_padding_mask=None,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        # residual + norm
        out = self.norm1(query_b + self.drop1(out))

        # ---- FFN ----
        if self.with_ffn:
            out2 = self.ffn(out)
            out = self.norm2(out + out2)

        # ---- return in original shape ----
        if query.dim() == 2:
            # choose batch 0 (since query was expanded); caller typically uses rois per-image anyway
            # safer: gather per-roi batch
            fused = torch.zeros((N, C), device=out.device, dtype=out.dtype)
            for b in range(B):
                mask = (b_inds == b)
                if mask.any():
                    fused[mask] = out[b, mask]
            return fused
        else:
            return out

    def extra_repr(self) -> str:
        return (f"embed_dims={self.embed_dims}, heads={self.num_heads}, "
                f"points={self.num_points}, dropout={self.dropout}, with_ffn={self.with_ffn}")