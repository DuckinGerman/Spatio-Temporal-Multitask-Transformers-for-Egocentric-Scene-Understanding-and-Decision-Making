# mmdetection-3.3.0/projects/vasco/models/backbones/video_swin3d.py
from typing import List, Dict, Sequence, Union
import torch
import torch.nn as nn
from mmdet.registry import MODELS

@MODELS.register_module()
class VideoSwin3DBackbone(nn.Module):
    """输入 (B,T,3,H,W)；输出各层 (B,T',C,H,W)。"""
    def __init__(self,
                 window_size=(8, 7, 7),
                 out_indices=(0, 1, 2, 3),
                 with_cp=True,
                 arch: Union[str, Dict] = dict(  # 关键：通过 arch 指定结构
                     embed_dims=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24)),
                 pretrained=None,
                 pretrained2d=False,
                 **kwargs):
        super().__init__()
        # 直接实例化类，避免 registry 作用域干扰
        from mmaction.models.backbones.swin import SwinTransformer3D

        self.backbone = SwinTransformer3D(
            arch=arch,
            pretrained=pretrained,         # 先用 None，跑通再换权重
            pretrained2d=pretrained2d,
            patch_size=(2, 4, 4),
            in_channels=3,
            window_size=window_size,       # (T_w,7,7)
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_cfg=dict(type='LN'),
            patch_norm=True,
            frozen_stages=-1,
            with_cp=with_cp,
            out_indices=out_indices,
            # 其余 kwargs 不再透传，避免未知键报错
        )

        if hasattr(self.backbone, 'init_weights'):
            try:
                self.backbone.init_weights()
            except TypeError:
                pass

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        assert x.dim() == 5
        if x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,3,T,H,W)
        feats = self.backbone(x)                       # (B,C,T',H,W)* 或 Tensor
        if isinstance(feats, torch.Tensor):
            feats = (feats,)
        return [f.permute(0, 2, 1, 3, 4).contiguous() for f in feats]  # -> (B,T',C,H,W)