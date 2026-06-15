# mmdetection-3.3.0/projects/vasco/models/detectors/vasco_dualpath_rcnn.py
import numpy as np
import torch
from typing import List
from mmdet.registry import MODELS
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from mmdet.structures import DetDataSample
from mmdet.models.detectors.two_stage import TwoStageDetector
from typing import List, Dict, Optional


@MODELS.register_module()
class VascoDualPathRCNN(TwoStageDetector):
    """双路分工：
    - 2D 路：末帧 -> 2D backbone -> FPN2D  -> RPN/RCNN（主检测链）
    - 3D 路：整段 -> 3D backbone 输出 feat3d_list，供 RoI 级时序聚合使用
    """

    def __init__(self,
                 backbone,            # 2D backbone (Swin/ViTDet)，由 super() 构建为 self.backbone
                 neck=None,           # 2D FPN，     由 super() 构建为 self.neck
                 backbone_3d=None,    # 3D backbone (Video Swin) —— 额外构建
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(backbone=backbone, neck=neck,
                         rpn_head=rpn_head, roi_head=roi_head,
                         train_cfg=train_cfg, test_cfg=test_cfg,
                         data_preprocessor=data_preprocessor, 
                         init_cfg=init_cfg)
        assert backbone_3d is not None, 'backbone_3d is required'
        self.backbone_3d = MODELS.build(backbone_3d)

    def extract_feat(self, batch_inputs: torch.Tensor):
        """batch_inputs: (B,T,3,H,W)"""
        assert batch_inputs.dim() == 5
        B, T, C, H, W = batch_inputs.shape

        # 2D 主链：仅末帧
        imgs_last = batch_inputs[:, -1]             # (B,3,H,W)
        x2d = self.backbone(imgs_last)              # 2D backbone
        if self.with_neck:
            x2d = self.neck(x2d)                    # list[L](B,256,H_l,W_l) -> 供 RPN/RCNN

        # 3D 辅助链：整段
        feat3d_list = self.backbone_3d(batch_inputs)  # list[L](B,T,C_l,H_l,W_l)

        # 将 3D 上下文交给 RoIHead（时序聚合在 RoIHead 内部完成）
        if hasattr(self.roi_head, 'set_temporal_ctx'):
            self.roi_head.set_temporal_ctx({'feat3d': feat3d_list, 'T': T})

        # 返回 2D FPN（末帧）给 RPN/RCNN
        return x2d
