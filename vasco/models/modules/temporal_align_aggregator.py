# -*- coding: utf-8 -*-
from typing import List, Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

def _sin_pos_enc(T: int, C: int, device) -> torch.Tensor:
    pe = torch.zeros(T, C, device=device)
    pos = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, C, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(1, C)))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

@MODELS.register_module()
class TemporalAlignAggregator(nn.Module):
    """RoI级时序聚合：Top‑K 选帧 → 逐bin偏移对齐 → Transformer → softmax加权 → (2D⊕γ·3D) 融合 + 一致性损失"""
    def __init__(self,
                 in_channels: int = 256,
                 roi_feat_size: int = 7,
                 k: int = 2,
                 bias_last: float = 0.5,
                 max_offset_px: float = 2.0,
                 detach_prev: bool = True,
                 gamma_max: float = 0.3,
                 gamma_scale: float = 1.0,
                 consistency_weight: float = 0.1,
                 heads: int = 4,
                 layers: int = 1):
        super().__init__()
        self.roi_feat_size = roi_feat_size
        self.k = k
        self.bias_last = bias_last
        self.max_offset_px = max_offset_px
        self.detach_prev = detach_prev
        self.gamma_max = gamma_max
        self.gamma_scale = gamma_scale
        self.consistency_weight = consistency_weight

        self.per_level_proj = None  # 回退路径用：per-level 1×1: C_l -> out_c

        # 偏移对齐：输出2通道(dx,dy)
        self.dw = nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1,
                            groups=in_channels * 2, bias=True)
        self.offset_head = nn.Conv2d(in_channels * 2, 2, kernel_size=3, padding=1, bias=True)

        # 融合分类用
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, 1, bias=True)

        # Transformer 时间建模
        enc = nn.TransformerEncoderLayer(d_model=in_channels, nhead=heads,
                                         batch_first=True, dropout=0.1, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.time_score = nn.Linear(in_channels, 1)

        # 相似度投影
        self.embed = nn.Conv2d(in_channels, 128, 1, bias=True)

        # 归一化采样网格
        u = torch.linspace(-1, 1, roi_feat_size)
        v = torch.linspace(-1, 1, roi_feat_size)
        gy, gx = torch.meshgrid(v, u, indexing='ij')
        base_grid = torch.stack([gx, gy], dim=-1)
        self.register_buffer('base_grid', base_grid)

    def _l2n(self, x, dim): return F.normalize(x, p=2, dim=dim, eps=1e-6)

    def _roi_align_on_time(self, roi_extractor, feats2d_list: List[torch.Tensor], rois: torch.Tensor):
        # feats2d_list: List[L](B,out_c,H,W)
        return roi_extractor(feats2d_list, rois)  # (R,out_c,h,w)

    def forward(self,
                rois: torch.Tensor,                       # (R,5)
                roi_extractor,
                feat3d: Optional[List[torch.Tensor]] = None,            # 回退：List[L](B,T,C,H,W)
                feats2d_seq: Optional[List[List[torch.Tensor]]] = None,  # 首选：List[T][L](B,out_c,H,W)
                roi_feats_last2d: Optional[torch.Tensor] = None,
                img_metas: Optional[List[Dict]] = None):
        device = rois.device
        R = rois.size(0)
        out_c = int(getattr(roi_extractor, 'out_channels', 768))

        # --- 统一通道并构造每帧特征 ---
        if feats2d_seq is not None:
            T = len(feats2d_seq); L = len(feats2d_seq[0])
            feats2d_by_t = [[f.to(device) for f in feats2d_seq[t]] for t in range(T)]
        else:
            assert isinstance(feat3d, (list, tuple)) and feat3d[0].dim() == 5, \
                '需提供 feats2d_seq 或 feat3d'
            # === 关键：用 3D backbone 的特征，按 t 拆成每帧的 FPN-like 特征 ===
            L = len(feat3d); T = int(feat3d[0].size(1))
            if self.per_level_proj is None:
                self.per_level_proj = nn.ModuleList(
                    [nn.Conv2d(int(feat3d[l].size(2)), out_c, 1, bias=True) for l in range(L)]
                ).to(feat3d[0].device)
            feats2d_by_t = []
            for t in range(T):
                per_t = [self.per_level_proj[l](feat3d[l][:, t].contiguous()) for l in range(L)]
                feats2d_by_t.append([p.to(device) for p in per_t])

        # 动态匹配通道（AMP 下安全）
        if self.dw.in_channels != out_c * 2:
            self.dw = nn.Conv2d(out_c * 2, out_c * 2, 3, padding=1,
                                groups=out_c * 2, bias=True).to(device)
            self.offset_head = nn.Conv2d(out_c * 2, 2, kernel_size=3, padding=1, bias=True).to(device)
            self.fuse = nn.Conv2d(out_c * 2, out_c, 1, bias=True).to(device)
            self.embed = nn.Conv2d(out_c, 128, 1, bias=True).to(device)

        # 逐时刻 RoIAlign
        X_t_list: List[torch.Tensor] = []
        for t in range(T):
            Xt = self._roi_align_on_time(roi_extractor, feats2d_by_t[t], rois)  # (R,out_c,h,w)
            if self.detach_prev and t != (T - 1):
                Xt = Xt.detach()
            X_t_list.append(Xt)
        X_T = X_t_list[T - 1]

        # 与末帧相似度（Top-K 用于“对齐选择”，Transformer 仍看全T，用 mask 控制）
        def _embed(x): return self._l2n(self.embed(x).mean(dim=(2, 3)), dim=1)
        e_T = _embed(X_T)
        sims = torch.cat([(_embed(x) * e_T).sum(dim=1, keepdim=True) for x in X_t_list], dim=1)  # (R,T)

        if T > 1 and self.k > 0:
            past = sims[:, :-1]; k = min(self.k, past.size(1))
            topk = past.topk(k=k, dim=1).indices             # (R,k)
        else:
            k = 0; topk = torch.empty(R, 0, dtype=torch.long, device=device)
        sel = torch.zeros_like(sims, dtype=torch.bool)       # (R,T)
        if k > 0: sel.scatter_(1, topk, True)
        sel[:, -1] = True

        # 可学习对齐（仅对选中的历史帧）
        h = w = self.roi_feat_size
        base_grid = self.base_grid.to(device=device, dtype=X_T.dtype).view(1, 1, h, w, 2).repeat(R, 1, 1, 1, 1)
        norm = 2.0 / (h - 1.0)
        max_off = self.max_offset_px * norm

        aligned = []
        for t in range(T):
            Xt = X_t_list[t]
            if t == T - 1 or not sel[:, t].any():
                aligned.append(Xt); continue
            pair = torch.cat([X_T, Xt], dim=1)               # (R,2C,h,w)
            feat = self.dw(pair)
            off = torch.tanh(self.offset_head(feat)) * max_off  # (R,2,h,w)
            grid = base_grid.clone()                            # dtype 已与 Xt 对齐
            grid[..., 0] += off[:, 0:1].permute(0, 1, 2, 3)
            grid[..., 1] += off[:, 1:2].permute(0, 1, 2, 3)
            grid = grid.view(R, h, w, 2)
            X_aligned = F.grid_sample(Xt, grid, mode='bilinear',
                                      padding_mode='zeros', align_corners=True)
            aligned.append(X_aligned)
        X_stack = torch.stack(aligned, dim=1)                # (R,T,C,h,w)

        # Transformer + 注意力（末帧加偏置）
        tokens = X_stack.mean(dim=(3, 4))                    # (R,T,C)
        pe = _sin_pos_enc(T, tokens.size(2), tokens.device).unsqueeze(0)
        h_enc = self.temporal_encoder(tokens + pe)           # (R,T,C)
        score = self.time_score(h_enc).squeeze(-1)           # (R,T)
        if self.bias_last != 0:
            score[:, -1] = score[:, -1] + self.bias_last     # 只加最后一列，避免广播错
        score = score.masked_fill(~sel, float('-inf'))
        alpha = torch.softmax(score, dim=1)                  # (R,T)

        X_temporal = (X_stack * alpha.view(R, T, 1, 1, 1)).sum(dim=1)  # (R,C,h,w)

        # 一致性损失
        loss_dict = {}
        if self.training and self.consistency_weight > 0:
            diff = (X_stack - X_T.unsqueeze(1).detach()) ** 2
            l2 = (diff.mean(dim=(2, 3, 4)) * sel.float()).sum(dim=1) / (sel.float().sum(dim=1) + 1e-6)
            loss_dict['loss_roi_consistency'] = l2.mean() * self.consistency_weight

        # 与 2D RoI 融合（分类使用；回归仍用 2D）
        s_mean = (alpha * sims).sum(dim=1, keepdim=True).view(R, 1, 1, 1)
        gamma = torch.clamp(self.gamma_scale * s_mean, 0.0, self.gamma_max)
        if roi_feats_last2d is None:
            roi_feats_last2d = X_T
        X_mix = self.fuse(torch.cat([roi_feats_last2d, (X_temporal.detach() if self.detach_prev else X_temporal)], dim=1))

        return (X_mix, loss_dict) if self.training else X_mix
