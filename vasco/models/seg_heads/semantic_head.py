import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Dict, Optional
from mmdet.registry import MODELS


def _conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Multi-class soft Dice on foreground classes with ignore_index masking.

        - Foreground classes: all classes except the last/pseudo-background class.
        - Pixels with ignore_index are excluded from computation.

        logits: [B, C, H, W]
        target: [B, H, W]
        """
        B, C, H, W = logits.shape
        probs = logits.softmax(dim=1)  # [B,C,H,W]

        valid = (target != self.ignore_index)  # [B,H,W]
        if valid.sum() == 0:
            return logits.sum() * 0.0

        # foreground ids: exclude pseudo-background class (assumed to be the last class)
        fg_ids = list(range(C - 1))
        if len(fg_ids) == 0:
            return logits.sum() * 0.0

        t = target.clamp(0, C - 1)
        onehot = F.one_hot(t, num_classes=C).permute(0, 3, 1, 2).to(probs.dtype)  # [B,C,H,W]

        m = valid.unsqueeze(1).to(probs.dtype)
        probs_fg = probs[:, fg_ids, :, :] * m
        onehot_fg = onehot[:, fg_ids, :, :] * m

        num = 2 * (probs_fg * onehot_fg).sum(dim=(0, 2, 3))
        den = (probs_fg * probs_fg).sum(dim=(0, 2, 3)) + (onehot_fg * onehot_fg).sum(dim=(0, 2, 3)) + 1e-6
        dice = 1.0 - num / den
        return dice.mean()


class PPM(nn.Module):
    """
    Pyramid Pooling Module
    """
    def __init__(
        self,
        in_channels: int,
        channels: int,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        align_corners: bool = False
    ):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners

        self.stages = nn.ModuleList()
        for scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        h, w = x.shape[2:]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(
                y, size=(h, w), mode='bilinear', align_corners=self.align_corners
            )
            out.append(y)
        return out


@MODELS.register_module()
class SemanticHead(nn.Module):
    """
    UPerNet-style semantic segmentation head for your VASCO project.

    Expected input:
        feats = [P2, P3, P4] or [P2, P3, P4, P5]
        Each feat is a Tensor [B, C_i, H_i, W_i]

    Main features:
        - PPM on highest-level feature
        - FPN-style top-down fusion
        - multi-scale feature aggregation
        - CE + Dice loss
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 256,
        num_classes: int = 7,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        dropout_ratio: float = 0.2,
        label_smoothing: float = 0.1,
        use_focal_ce: bool = False,
        focal_gamma: float = 2.0,
        ignore_index: int = 255,
        align_corners: bool = False,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        bg_id: int = 6,
        bg_weight: float = 0.1,
        return_feat: bool = False
    ):
        super().__init__()
        assert len(in_channels) >= 3, "SemanticHead expects at least 3 input feature levels."

        self.in_channels = list(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.pool_scales = pool_scales
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = float(label_smoothing)
        self.use_focal_ce = bool(use_focal_ce)
        self.focal_gamma = float(focal_gamma)
        self.bg_id = int(bg_id)
        self.bg_weight = float(bg_weight)
        self.return_feat = return_feat

        # PPM on the last/highest-level feature
        self.ppm = PPM(
            in_channels=self.in_channels[-1],
            channels=channels,
            pool_scales=pool_scales,
            align_corners=align_corners
        )
        ppm_out_channels = self.in_channels[-1] + len(pool_scales) * channels
        self.ppm_bottleneck = _conv_bn_relu(ppm_out_channels, channels, kernel_size=3, padding=1)

        # lateral convs for all lower levels except the highest one
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_ch in self.in_channels[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.fpn_convs.append(_conv_bn_relu(channels, channels, kernel_size=3, padding=1))

        # FPN bottleneck after concatenating all scales
        self.fpn_bottleneck = _conv_bn_relu(len(self.in_channels) * channels, channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        self.loss_dice = DiceLoss(ignore_index=ignore_index)

    def psp_forward(self, x: torch.Tensor) -> torch.Tensor:
        ppm_outs = [x]
        ppm_outs.extend(self.ppm(x))
        ppm_outs = torch.cat(ppm_outs, dim=1)
        output = self.ppm_bottleneck(ppm_outs)
        return output

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        feats: list of feature maps, ordered from high-resolution to low-resolution
               e.g. [P2, P3, P4] or [P2, P3, P4, P5]
        returns:
               logits [B, num_classes, H, W] at the highest resolution among inputs
        """
        assert isinstance(feats, (list, tuple)), "feats must be a list/tuple of multi-scale feature maps"
        assert len(feats) == len(self.in_channels), (
            f"Expected {len(self.in_channels)} feature levels, got {len(feats)}"
        )

        # build laterals for lower levels
        laterals = [
            lateral_conv(feats[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # top/highest-level feature with PPM
        top = self.psp_forward(feats[-1])
        laterals.append(top)

        # top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners
            )

        # apply 3x3 conv on all but the highest path from lateral branch
        fpn_outs = []
        for i in range(len(laterals) - 1):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))
        fpn_outs.append(laterals[-1])  # top feature already refined by PPM bottleneck

        # upsample all outputs to the highest resolution
        out_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=out_size, mode='bilinear', align_corners=self.align_corners
            )

        fusion = torch.cat(fpn_outs, dim=1)
        fusion = self.fpn_bottleneck(fusion)
        fusion = self.dropout(fusion)
        logits = self.cls_seg(fusion)

        if getattr(self, 'return_feat', False):
            return logits, fusion
        
        return logits

    def _stack_gt_semantic(self, data_samples: List) -> torch.Tensor:
        """
        Extract semantic GT mask from data_samples.
        Supports:
            sample.gt_sem_seg.data / tensor
            sample.sem_seg.data / tensor
            sample.seg.data / tensor
        Returns:
            gt: [B, H, W]
        """
        gt_list = []
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
                raise AttributeError('No semantic GT found in data_sample. Expected one of: gt_sem_seg, sem_seg, seg')
            if gt.dim() == 3 and gt.size(0) == 1:
                gt = gt[0]
            gt_list.append(gt.long())

        gt = torch.stack(gt_list, dim=0)  # [B, H, W]
        return gt

    def _ce_loss(self, logits: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            gt,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )

        valid = (gt != self.ignore_index)
        if self.use_focal_ce:
            pt = torch.exp(-ce)
            ce = ((1.0 - pt) ** self.focal_gamma) * ce

        ce = ce[valid]
        if ce.numel() == 0:
            return logits.sum() * 0.0
        return ce.mean()

    def _ce_loss_fg_only(self, logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            gt,
            weight=None,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )

        valid_fg = (
            (gt != self.ignore_index) &
            (gt != self.bg_id) &
            (gt >= 0) &
            (gt < self.num_classes)
        )

        ce = ce[valid_fg]
        if ce.numel() == 0:
            return logits.sum() * 0.0
        return ce.mean()

    def loss_from_batch(self, feats: List[torch.Tensor], data_samples: List) -> Dict[str, torch.Tensor]:
        out = self.forward(feats)
        if isinstance(out, tuple):
            logits, _ = out
        else:
            logits = out
        gt = self._stack_gt_semantic(data_samples).to(logits.device)

        if gt.shape[-2:] != logits.shape[-2:]:
            gt = F.interpolate(
                gt.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        valid = (gt != self.ignore_index)
        if valid.sum() == 0:
            z = logits.sum() * 0.0
            return {
                'loss_sem_ce': z,
                'loss_sem_dice': z,
                'loss_sem_ce_fg': z 
            }

        w = torch.ones((self.num_classes,), device=logits.device, dtype=logits.dtype)
        if 0 <= self.bg_id < self.num_classes:
            w[self.bg_id] = self.bg_weight

        loss_ce = self._ce_loss(logits, gt, w) * self.ce_weight
        loss_dice = self.loss_dice(logits, gt) * self.dice_weight
        loss_sem_ce_fg = self._ce_loss_fg_only(logits, gt).detach()

        return {
            'loss_sem_ce': loss_ce,
            'loss_sem_dice': loss_dice,
            'loss_sem_ce_fg': loss_sem_ce_fg 
        }
