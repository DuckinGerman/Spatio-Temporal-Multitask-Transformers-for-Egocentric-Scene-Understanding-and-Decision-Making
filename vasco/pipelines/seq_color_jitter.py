# projects/vasco/pipelines/seq_color_jitter.py
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SeqColorJitter:
    """对一个 clip 的所有帧做一致的颜色增强（亮度 / 对比度 / 饱和度）。

    注意：
    - 同一个 clip 内所有帧用同一组 (brightness, contrast, saturation) 系数
    - 不改 hue（色相），避免影响红绿灯颜色语义
    - 输入假定是 BGR uint8（和 LoadMultiImagesFromFile 一致）
    """

    def __init__(self,
                 brightness: float = 0.2,
                 contrast: float = 0.2,
                 saturation: float = 0.1,
                 p: float = 0.8):
        """
        brightness: 亮度扰动幅度，系数范围为 [1-brightness, 1+brightness]
        contrast:   对比度扰动幅度，系数范围为 [1-contrast,   1+contrast]
        saturation: 饱和度扰动幅度，系数范围为 [1-saturation, 1+saturation]
        p:          整个 clip 做颜色增强的概率
        """
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.p = float(p)

    @staticmethod
    def _apply_jitter_single(img: np.ndarray,
                             b_factor: float,
                             c_factor: float,
                             s_factor: float) -> np.ndarray:
        """对单帧应用 BC+S 抖动，输入/输出皆为 BGR uint8。"""
        assert img.dtype == np.uint8, 'expect uint8 image'
        x = img.astype(np.float32)

        # --- 亮度：整体乘一个系数 ---
        if b_factor is not None:
            x = x * b_factor

        # --- 对比度：围绕全图均值拉伸 ---
        if c_factor is not None:
            mean = x.mean(axis=(0, 1), keepdims=True)
            x = (x - mean) * c_factor + mean

        # 先裁到合法范围
        x = np.clip(x, 0, 255).astype(np.uint8)

        # --- 饱和度：在 HSV 空间只改 S 通道 ---
        if s_factor is not None and abs(s_factor - 1.0) > 1e-3:
            hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
            # hsv[..., 1] 是 S ∈ [0,255]
            hsv[..., 1] = hsv[..., 1] * s_factor
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return x

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # 有一定概率完全不做增强
        if np.random.rand() >= self.p:
            return results

        imgs: List[np.ndarray] = results['imgs']
        if not isinstance(imgs, list) or len(imgs) == 0:
            return results

        # 为这个 clip 抽一组统一的系数
        def _sample_factor(mag: float) -> float:
            if mag <= 0:
                return 1.0
            f = 1.0 + np.random.uniform(-mag, mag)
            return max(f, 0.0)  # 防止负数

        b_factor = _sample_factor(self.brightness)
        c_factor = _sample_factor(self.contrast)
        s_factor = _sample_factor(self.saturation)

        new_imgs = []
        for img in imgs:
            # 保守检查
            if img is None:
                new_imgs.append(img)
                continue
            if img.dtype != np.uint8:
                img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img_u8 = img
            aug = self._apply_jitter_single(img_u8, b_factor, c_factor, s_factor)
            new_imgs.append(aug)

        results['imgs'] = new_imgs
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(brightness={self.brightness}, '
                f'contrast={self.contrast}, saturation={self.saturation}, p={self.p})')