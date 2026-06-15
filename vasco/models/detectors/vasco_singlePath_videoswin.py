# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.structures import DetDataSample


@MODELS.register_module()
class VascoVideoRCNN(TwoStageDetector):
    """单路 Video Swin3D 两阶段检测器：
    - 继承 TwoStageDetector：RPN + RoIHead 流程用官方逻辑
    - 只重写 extract_feat：
        * backbone: VideoSwin3DBackbone，输出 list[(B,T,C,H,W)]
        * x_2d: 末帧多层特征 list[(B,C,H,W)] → 提供给 TwoStageDetector
        * 同时把 x_3d 注入 roi_head.set_temporal_ctx({'feat3d': x_3d})
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 rpn_head: Optional[dict] = None,
                 roi_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        # 注意：这里一定要有 neck 参数，即使我们用不到，也要接住
        super().__init__(
            backbone=backbone,
            neck=neck,                 # 这里会是 None（你在 config 里设的 neck=None）
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def extract_feat(self, batch_inputs: Tensor) -> List[Tensor]:
        """batch_inputs: (B,T,C,H,W)
        返回给 TwoStageDetector 的特征: list[(B,C_l,H_l,W_l)]
        """
        # 1) Video Swin3D：输出 list[(B,T',C_l,H_l,W_l)]
        x_3d = self.backbone(batch_inputs)

        # 2) 末帧 2D 特征：list[(B,C_l,H_l,W_l)]
        x_2d = [feat[:, -1] for feat in x_3d]

        # 3) 把 3D 特征传给 RoIHead 做时序聚合
        if self.roi_head is not None and hasattr(self.roi_head, 'set_temporal_ctx'):
            self.roi_head.set_temporal_ctx({'feat3d': x_3d})

        # TwoStageDetector 的 loss/predict 会直接用 x_2d 做 RPN + RoIHead
        return x_2d