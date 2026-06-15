# -*- coding: utf-8 -*-
# projects/vasco/models/backbones/swin_v2_det.py
from typing import Tuple, List, Union
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS

# 兼容不同 mmcls 版本的 import 路径
try:
    from mmcls.models.backbones import SwinTransformerV2 as _SwinV2
except Exception:
    # 有些旧版 mmcls 的路径
    from mmcls.models.backbones.swin_transformer_v2 import SwinTransformerV2 as _SwinV2


@MODELS.register_module()
class SwinTransformerV2Det(BaseModule):
    """将 mmcls 的 SwinTransformerV2 显式注册到 mmdet::MODELS 供检测使用。

    说明：
    - 完全复用 mmcls 的实现与权重加载；这里只做一层薄封装，保证与 mmdet 的期望接口对齐。
    - forward 返回 List[Tensor]（FPN 习惯用 list），如果 mmcls 返回 tuple，会转换为 list。
    - init_cfg 直接传给 mmcls 模型；本 wrapper 自身不做二次 init。
    """

    def __init__(self,
                 embed_dims: int = 96,
                 depths: Tuple[int, int, int, int] = (2, 2, 18, 2),
                 num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
                 window_size: int = 8,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_path_rate: float = 0.2,
                 use_abs_pos_embed: bool = False,
                 out_indices: Tuple[int, int, int, int] = (0, 1, 2, 3),
                 with_cp: bool = False,
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 init_cfg: Union[dict, None] = None,
                 **kwargs):
        # 不让 BaseModule 用自己的 init_cfg 触发二次 init，全部交给 mmcls backbone 处理
        super().__init__(init_cfg=None)
        # 直接构建 mmcls 的 SwinTransformerV2
        self.backbone = _SwinV2(
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            use_abs_pos_embed=use_abs_pos_embed,
            out_indices=out_indices,
            with_cp=with_cp,
            frozen_stages=frozen_stages,
            norm_eval=norm_eval,
            init_cfg=init_cfg,
            **kwargs
        )

    def init_weights(self):
        # 交给 mmcls backbone 执行权重初始化/加载
        self.backbone.init_weights()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.backbone(x)
        if isinstance(outs, tuple):
            outs = list(outs)
        return outs

    def train(self, mode: bool = True):
        # 直接沿用 mmcls 的训练/冻结/BN eval 逻辑
        self.backbone.train(mode)
        return super().train(mode)
