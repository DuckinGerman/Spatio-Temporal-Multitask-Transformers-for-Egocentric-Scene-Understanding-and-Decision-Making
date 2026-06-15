# -*- coding: utf-8 -*-
# vasco/models/modules/temporal_agg_3d.py
from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

def sinusoidal_pe(T: int, C: int, device) -> torch.Tensor:
    pe = torch.zeros(T, C, device=device)
    pos = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, C, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(1, C)))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (T,C)

@MODELS.register_module()
class VascoRoITemporalAgg3D(nn.Module):
    """
    B‑multi + Transformer：
      - 候选层 S（near/single/all）
      - 在各层的所有时刻做 RoIAlign → 1×1(C_l→C_out)
      - Transformer 时间聚合生成 α_t
      - 层级注意力融合各层
    """
    def __init__(self,
                 multi_level: str = 'near',     # 'single' | 'near' | 'all'
                 k: int = 1,                    # near 的跨度
                 out_channels: int = 256,
                 heads: int = 4,
                 layers: int = 1,
                 level_fusion: str = 'attn',    # 'attn' | 'avg'
                 detach_prev_frames: bool = True,
                 temporal_dropout: float = 0.2,
                 featmap_strides: Tuple[int, ...] = (4, 8, 16, 32),
                 in_channels: Tuple[int,...]=(96,192,384,768)):
        super().__init__()
        self.multi_level, self.k = multi_level, k
        self.out_channels = out_channels
        self.level_fusion = level_fusion
        self.detach_prev_frames = detach_prev_frames
        self.temporal_dropout = temporal_dropout
        self.featmap_strides = featmap_strides
        
        L = len(featmap_strides)
        # 延迟构建：首次看到 C_l 再替换为 Conv2d
        self.per_level_proj = nn.ModuleList([
            nn.Conv2d(c, self.out_channels, 1) for c in in_channels
        ])

        enc = nn.TransformerEncoderLayer(
            d_model=out_channels, nhead=heads, batch_first=True, norm_first=True, dropout=0.1)
        self.temporal_encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.time_score = nn.Linear(out_channels, 1)

        self.lvl_emb = nn.Embedding(L, 16)
        self.level_mlp = nn.Sequential(
            nn.Linear(out_channels + 16 + 1, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, 1)
        )

    # --- 工具 ---
    def _map_roi_levels(self, rois: torch.Tensor, roi_extractor) -> torch.Tensor:
        # 兼容两种写法：roi_layers 或 featmap_strides
        if hasattr(roi_extractor, 'roi_layers') and isinstance(roi_extractor.roi_layers, (list, tuple)):
            num_levels = len(roi_extractor.roi_layers)
        else:
            num_levels = len(getattr(roi_extractor, 'featmap_strides', [4, 8, 16, 32]))
        return roi_extractor.map_roi_levels(rois, num_levels)

    def _cand_mask(self, l_star: torch.Tensor, L: int) -> torch.Tensor:
        if self.multi_level == 'all':
            return torch.ones((l_star.numel(), L), dtype=torch.bool, device=l_star.device)
        if self.multi_level == 'single':
            return F.one_hot(l_star, num_classes=L).bool()
        k = max(0, int(self.k))
        mask = torch.zeros((l_star.numel(), L), dtype=torch.bool, device=l_star.device)
        for s in range(-k, k + 1):
            pick = (l_star + s).clamp(0, L - 1)
            mask.scatter_(1, pick.view(-1, 1), True)
        return mask

    def _temporal_weights(self, tokens: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        # tokens: (n_l, T', C)
        Tp = tokens.size(1)
        pe = sinusoidal_pe(Tp, tokens.size(2), tokens.device)
        h = self.temporal_encoder(tokens + pe.unsqueeze(0))        # (n_l,T',C)
        s = self.time_score(h).squeeze(-1).masked_fill(~keep_mask, float('-inf'))
        return torch.softmax(s, dim=1)                             # (n_l,T')

    # --- 前向 ---
    def forward(self,
                rois: torch.Tensor,
                roi_extractor: nn.Module,
                feat3d: List[torch.Tensor]) -> torch.Tensor:
        """
        rois: (N,5) [img_idx,x1,y1,x2,y2]
        feat3d: list[L] of (B,T',C_l,H_l,W_l)
        return: (N, out_channels, H_out, W_out) 与 bbox_head 输入一致
        """
        device = rois.device
        L = len(feat3d)
        N = rois.size(0)

        l_star = self._map_roi_levels(rois, roi_extractor)         # (N,)
        cand = self._cand_mask(l_star, L)                          # (N,L)

        # 存各层输出与注意力分数
        F_store = [None for _ in range(L)]
        u_full = rois.new_full((N, L), float('-inf'))
        H_out = W_out = None

        for l in range(L):
            idx = torch.nonzero(cand[:, l], as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            rois_l = rois.index_select(0, idx)
            x3d = feat3d[l]                                        # (B,T',C_l,H_l,W_l)
            B, Tp, C_l, _, _ = x3d.shape

            # 懒构建 per-level 1×1
            # if isinstance(self.per_level_proj[l], nn.Identity):
            #     self.per_level_proj[l] = nn.Conv2d(C_l, self.out_channels, 1).to(x3d.device)
            proj = self.per_level_proj[l]

            seq_list, keep_list = [], []
            for t in range(Tp):
                xt = x3d[:, t]                                     # (B,C_l,H,W)
                if self.detach_prev_frames and t < Tp - 1:
                    xt = xt.detach()
                x_roi = roi_extractor.roi_layers[l](xt, rois_l)    # (n_l,C_l,h,w)
                seq_list.append(x_roi)
                # temporal dropout（不丢末帧）
                if self.training and self.temporal_dropout > 0 and t < Tp - 1:
                    keep = (torch.rand((x_roi.size(0),), device=device) > self.temporal_dropout)
                else:
                    keep = torch.ones((x_roi.size(0),), device=device, dtype=torch.bool)
                if t == Tp - 1:
                    keep = torch.ones_like(keep, device=device, dtype=torch.bool)
                keep_list.append(keep)

            x_seq = torch.stack(seq_list, dim=1)                   # (n_l,T',C_l,h,w)
            keep_mask = torch.stack(keep_list, dim=1)              # (n_l,T')
            n_l, Tp, C_l, h, w = x_seq.shape
            H_out = H_out or h
            W_out = W_out or w

            # 1×1 统一通道
            x_seq = proj(x_seq.view(-1, C_l, h, w)).view(n_l, Tp, self.out_channels, h, w)

            # Transformer → α_t
            tokens = x_seq.mean(dim=(3, 4))                        # (n_l,T',C)
            alpha = self._temporal_weights(tokens, keep_mask)      # (n_l,T')

            # 时间加权还原 7×7（或 h×w）
            F_l = (x_seq * alpha.view(n_l, Tp, 1, 1, 1)).sum(dim=1)  # (n_l,C,h,w)

            # 回填
            if F_store[l] is None:
                F_store[l] = torch.zeros((N, self.out_channels, h, w), device=device, dtype=F_l.dtype)
            F_store[l].index_copy_(0, idx, F_l)

            # 层级权重分数 u_l
            g_l = F.adaptive_avg_pool2d(F_l, 1).flatten(1)         # (n_l,C)
            lvl_feat = self.lvl_emb(torch.full((n_l,), l, device=device, dtype=torch.long))
            area = (rois_l[:, 3] - rois_l[:, 1]).clamp(min=0) * (rois_l[:, 4] - rois_l[:, 2]).clamp(min=0)
            u_l = self.level_mlp(torch.cat([g_l, lvl_feat, torch.log(area + 1).unsqueeze(1)], dim=1)).squeeze(1)
            u_l = u_l.to(u_full.dtype)
            u_full[idx, l] = u_l

        # 层级融合
        if self.level_fusion == 'avg':
            beta = cand.float()
            beta = beta / (beta.sum(dim=1, keepdim=True) + 1e-6)
        else:
            beta = torch.softmax(u_full, dim=1)                    # (N,L)

        out = rois.new_zeros((N, self.out_channels, H_out, W_out))
        for l in range(L):
            if F_store[l] is not None:
                out = out + F_store[l].to(out.dtype) * beta[:, l].view(N, 1, 1, 1)
        return out