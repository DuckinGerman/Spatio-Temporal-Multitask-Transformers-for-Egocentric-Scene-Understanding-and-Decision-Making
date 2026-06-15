# projects/vasco/pipelines/load_multi_images.py
import os.path as osp
from typing import Dict, List, Any
import mmcv
import numpy as np
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadMultiImagesFromFile:
    """Load T-frame clip images into `results['imgs']` and set basic metainfo.
    Expect keys in results: img_path_list(list[str]), img_path(str), height, width, frame_inds, clip_len.
    """

    def __init__(self, to_float32: bool = False, color_type: str = 'color', channel_order: str = 'bgr'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img_paths: List[str] = results['img_path_list']
        imgs: List[np.ndarray] = []
        for p in img_paths:
            img = mmcv.imread(p, flag=self.color_type, channel_order=self.channel_order)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)

        # basic shapes from original metadata
        h, w = imgs[-1].shape[:2]
        results['imgs'] = imgs
        results.setdefault('ori_shape', (h, w))
        results.setdefault('img_shape', (h, w))   # will be overwritten after aug
        results.setdefault('pad_shape', (h, w))   # will be overwritten after pad
        results.setdefault('flip', False)
        results.setdefault('flip_direction', None)
        # keep last frame path for convenience
        results.setdefault('img_path', results.get('img_path_list')[-1])
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(to_float32={self.to_float32}, '
                f'color_type={self.color_type}, channel_order={self.channel_order})')
