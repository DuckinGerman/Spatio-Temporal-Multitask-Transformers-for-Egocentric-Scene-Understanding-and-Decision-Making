# mmdetection-3.3.0/projects/vasco/models/detectors/vasco_singlepath_rcnn.py
import torch
from typing import Dict

from mmdet.registry import MODELS
from mmdet.models.detectors.two_stage import TwoStageDetector


@MODELS.register_module()
class VascoSinglePathRCNN(TwoStageDetector):
    """Single-path RCNN for Vasco (2D per-frame + temporal context).

    Behaviour:
      - Detection / segmentation: use ONLY the last frame 2D FPN features.
      - Temporal heads (action/dir/ego): RoIHead consumes per-frame 2D FPN features
        via `set_temporal_ctx({'feat2d_seq': ..., 'T': T})`.

    This detector is strictly 2D-only (no 3D backbone).
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def extract_feat(self, batch_inputs: torch.Tensor):
        """Extract features.

        Args:
            batch_inputs (Tensor): (B, T, 3, H, W)

        Returns:
            list[Tensor]: last-frame 2D FPN features for RPN/RCNN.

        Side-effect:
            Calls roi_head.set_temporal_ctx({'feat2d_seq': feat2d_seq, 'T': T})
            where feat2d_seq is list[L] each of shape (B, T, C, H_l, W_l).
        """
        assert batch_inputs.dim() == 5, f"batch_inputs must be (B,T,3,H,W), got {tuple(batch_inputs.shape)}"
        B, T, C, H, W = batch_inputs.shape

        # ===== 1) 2D temporal context: frames [0..T-2] with NO grad =====
        if T > 1:
            imgs_ctx = batch_inputs[:, :-1]  # (B,T-1,3,H,W)
            imgs_ctx = imgs_ctx.reshape(B * (T - 1), C, H, W)
            with torch.no_grad():
                x2d_ctx = self.backbone(imgs_ctx)
                if self.with_neck:
                    x2d_ctx = self.neck(x2d_ctx)  # list[L](B*(T-1),256,Hl,Wl)
        else:
            x2d_ctx = None

        # ===== 2) 2D main path (and also temporal last frame): last frame WITH grad =====
        imgs_last = batch_inputs[:, -1]  # (B,3,H,W)
        x2d_last = self.backbone(imgs_last)
        if self.with_neck:
            x2d_last = self.neck(x2d_last)  # list[L](B,256,Hl,Wl)

        # ===== 3) Build temporal feature sequence list[L] -> (B,T,C2,Hl,Wl) =====
        feat2d_seq = []
        if x2d_ctx is None:
            # T == 1
            for lvl_last in x2d_last:
                _, C2, Hl, Wl = lvl_last.shape
                feat2d_seq.append(lvl_last.view(B, 1, C2, Hl, Wl).contiguous())
        else:
            for lvl_ctx, lvl_last in zip(x2d_ctx, x2d_last):
                # lvl_ctx: (B*(T-1), C2, Hl, Wl), lvl_last: (B, C2, Hl, Wl)
                _, C2, Hl, Wl = lvl_last.shape
                lvl_ctx_seq = lvl_ctx.view(B, T - 1, C2, Hl, Wl).contiguous()
                lvl_last_seq = lvl_last.view(B, 1, C2, Hl, Wl).contiguous()
                feat2d_seq.append(torch.cat([lvl_ctx_seq, lvl_last_seq], dim=1))

        # Provide temporal 2D context to RoIHead
        if hasattr(self.roi_head, 'set_temporal_ctx'):
            ctx: Dict[str, object] = {'feat2d_seq': feat2d_seq, 'T': int(T), 'img_hw': (int(H), int(W))}
            self.roi_head.set_temporal_ctx(ctx)

        # Return last-frame 2D FPN for RPN/RCNN
        return x2d_last