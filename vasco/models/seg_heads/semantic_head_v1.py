# projects/vasco/models/seg_heads/semantic_head.py
from typing import Tuple, Dict, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class SemanticHead(nn.Module):
    """Lightweight multi-scale stuff segmentation head (use FPN P2–P4).

    - Input:
        * Multi-scale: List[Tensor] / Tuple[Tensor], e.g. [P2, P3, P4], each (B, C_l, H_l, W_l)
        * Single-scale: Tensor (B, C, H, W) for backward compatibility
    - Output:
        * logits: (B, C_sem, H_out, W_out)

    Label convention:
        * foreground stuff classes: 0..5
        * background class: bg_id (default 6)
        * ignore pixels (padding/invalid): ignore_index (default 255)

    Note: This head uses an explicit background class (bg_id) but still keeps ignore_index for padding.
    """

    def __init__(self,
                 in_channels: Union[int, List[int], Tuple[int, ...]] = (256, 256, 256),
                 num_classes: int = 7,          # stuff classes: 0..5 + 6
                 ce_weight: float = 1.0,
                 dice_weight: float = 0.5,
                 ignore_index: int = 255,
                 bg_id: int = 6,
                 bg_weight: float = 0.1):      # 数据集里的 ignore 标记
        super().__init__()

        # 统一 in_channels 为 list
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        self.in_channels = list(in_channels)
        self.num_scales = len(self.in_channels)

        # 每个尺度 1×1 降到 64 通道
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(c, 64, kernel_size=1, bias=True)
            for c in self.in_channels
        ])

        # 融合: concat 后 3×3 → 128
        self.fuse = nn.Sequential(
            nn.Conv2d(64 * self.num_scales, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # Classification head: 1×1 → num_classes (no background channel)
        self.cls = nn.Conv2d(128, num_classes, kernel_size=1, bias=True)

        # Loss config
        self.num_classes = int(num_classes)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.bg_id = int(bg_id)
        self.bg_weight = float(bg_weight)
        self.ignore_index = int(ignore_index)  # =255

    # ----------------- 多尺度前向 -----------------
    def _forward_multiscale(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """多尺度融合前向，feats: List[T_l]，取第 0 层为目标尺度."""
        assert len(feats) == self.num_scales, \
            f'期望 {self.num_scales} 个尺度特征，收到 {len(feats)} 个'

        # 以第 0 个尺度（建议是 P2）为目标分辨率
        B, _, H, W = feats[0].shape
        up_feats: List[torch.Tensor] = []

        for i, x in enumerate(feats):
            x_proj = self.proj_convs[i](x)  # (B,64,H_l,W_l)
            if x_proj.shape[-2:] != (H, W):
                x_proj = F.interpolate(
                    x_proj, size=(H, W),
                    mode='bilinear', align_corners=False
                )
            up_feats.append(x_proj)

        x_cat = torch.cat(up_feats, dim=1)       # (B,64*K,H,W)
        x_fused = self.fuse(x_cat)              # (B,128,H,W)
        logits = self.cls(x_fused)              # (B,C_sem,H,W)
        return logits

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """前向:
        - x 是 list/tuple → 多尺度融合
        - x 是 Tensor → 单尺度退化（向后兼容旧调用）
        """
        if isinstance(x, (list, tuple)):
            return self._forward_multiscale(list(x))
        elif isinstance(x, torch.Tensor):
            # 旧用法: 只给 P2，则当作单尺度，走一个 proj+fuse+cls 的简化路径
            assert len(self.proj_convs) >= 1, 'proj_convs 未初始化'
            x_proj = self.proj_convs[0](x)
            x_fused = self.fuse[0](x_proj)      # Conv3×3
            x_fused = self.fuse[1](x_fused)     # ReLU
            logits = self.cls(x_fused)
            return logits
        else:
            raise TypeError(f'Unsupported input type for SemanticHead: {type(x)}')

    # ----------------- Dice 损失 -----------------
    def _dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-class soft Dice on foreground classes with ignore_index masking.

        - Foreground classes: all classes except `bg_id`.
        - Pixels with `ignore_index` are excluded from the computation.

        Args:
            logits: (B, C, H, W)
            target: (B, H, W) with values in [0..C-1] or ignore_index
        """
        B, C, H, W = logits.shape
        probs = logits.softmax(dim=1)  # (B,C,H,W)

        valid = (target != self.ignore_index)  # (B,H,W)
        if valid.sum() == 0:
            return logits.sum() * 0

        # Foreground channel indices (exclude background)
        fg_ids = [i for i in range(C) if i != self.bg_id]
        if len(fg_ids) == 0:
            return logits.sum() * 0

        # Clamp for one_hot safety; invalid pixels are masked out anyway
        t = target.clamp(0, C - 1)
        onehot = F.one_hot(t, num_classes=C).permute(0, 3, 1, 2).to(probs.dtype)  # (B,C,H,W)

        m = valid.unsqueeze(1).to(probs.dtype)  # (B,1,H,W)

        probs_fg = probs[:, fg_ids, :, :] * m
        onehot_fg = onehot[:, fg_ids, :, :] * m

        num = 2 * (probs_fg * onehot_fg).sum(dim=(0, 2, 3))
        den = (probs_fg * probs_fg).sum(dim=(0, 2, 3)) + (onehot_fg * onehot_fg).sum(dim=(0, 2, 3)) + 1e-6
        dice = 1.0 - num / den
        return dice.mean()

    # ----------------- Loss 接口 -----------------
    def loss_from_batch(self,
                        feats,
                        data_samples: List) -> Dict[str, torch.Tensor]:
        logits = self.forward(feats)  # (B,C,H,W)  C=num_classes

        # 收集 gt masks
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
        gt = torch.stack(gts, dim=0)  # (B,H,W)，取值 0..6 或 255

        # 对齐到 logits 尺度
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = F.interpolate(
                gt.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        # If the whole batch has no valid stuff pixels, return zero losses.
        valid = (gt != self.ignore_index)
        if valid.sum() == 0:
            z = logits.sum() * 0
            return dict(loss_sem_ce=z, loss_sem_dice=z)
        
        
        # Cross entropy with ignore_index + class weights (down-weight background)
        w = torch.ones((self.num_classes,), device=logits.device, dtype=logits.dtype)
        if 0 <= self.bg_id < self.num_classes:
            w[self.bg_id] = self.bg_weight
        ce = F.cross_entropy(
            logits,
            gt,
            weight=w,
            ignore_index=self.ignore_index,
        ) * self.ce_weight

        # Dice computed on valid pixels only
        dice = self._dice(logits, gt) * self.dice_weight

        return dict(loss_sem_ce=ce, loss_sem_dice=dice)