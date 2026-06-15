from typing import Any, Dict, List
import numpy as np
import torch
from mmdet.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor

@MODELS.register_module()
class VascoDataPreprocessor(DetDataPreprocessor):
    """支持 [B,T,3,H,W]。5D时手动归一化并直接返回；4D沿用父类。"""

    def _to_tensor_BTCHW(self, imgs_seq: List[Any]) -> torch.Tensor:
        # imgs_seq: 长度T，每个元素是长度B的list/tuple，元素为 np.ndarray(H,W,3)
        T = len(imgs_seq)
        B = len(imgs_seq[0])
        tbchw = []
        for t in range(T):
            bt = []
            for b in range(B):
                arr = imgs_seq[t][b]
                if not isinstance(arr, np.ndarray):
                    arr = np.asarray(arr)
                t_chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # 3,H,W
                bt.append(t_chw)
            tbchw.append(torch.stack(bt, 0))  # B,3,H,W
        tbchw = torch.stack(tbchw, 0)        # T,B,3,H,W
        return tbchw.permute(1, 0, 2, 3, 4).contiguous()  # B,T,3,H,W

    def _norm_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,3,H,W  -> float32, [0,255] 到减均值/除方差；可选BGR->RGB
        x = x.to(dtype=torch.float32)
        if getattr(self, 'bgr_to_rgb', False):
            x = x[:, :, [2, 1, 0], :, :]
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype).view(1,1,3,1,1)
        std  = torch.as_tensor(self.std,  device=x.device, dtype=x.dtype).view(1,1,3,1,1)
        return (x - mean) / std

    def forward(self, data: Dict, training: bool = False) -> Dict:
        # 1) 把 collate 后的 {'inputs':{'imgs': list[T] of list/tuple[B] of np.ndarray}} 变为 B,T,3,H,W
        if isinstance(data, dict) and isinstance(data.get('inputs', None), dict):
            imgs_seq = data['inputs'].get('imgs', None)
            if imgs_seq is not None:
                btchw = self._to_tensor_BTCHW(imgs_seq)
                data = dict(inputs=btchw, data_samples=data['data_samples'])

        inputs = data['inputs']
        data_samples = data['data_samples']

        # 2) 5D 走自定义路径；4D 走父类
        if torch.is_tensor(inputs) and inputs.dim() == 5:
            device = getattr(self, 'device', None)
            if device is None:
                # 与父类一致：首次调用时设置 device
                self.device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device('cpu')
                device = self.device
            inputs = inputs.to(device=device)
            data_samples = self.cast_data(data_samples)

            # 归一化（SeqResizeFlipPad已处理尺寸/填充，此处不再pad）
            inputs = self._norm_5d(inputs)

            # 统一 meta.scale_factor 为末帧4元
            for ds in data_samples:
                sf = ds.metainfo.get('scale_factor', None)
                if sf is not None:
                    arr = np.asarray(sf, dtype=np.float32).reshape(-1)
                    if arr.size == 2:
                        sf4 = np.array([arr[0], arr[1], arr[0], arr[1]], dtype=np.float32)
                    elif arr.size == 4:
                        sf4 = arr.astype(np.float32)
                    elif arr.size % 4 == 0:
                        sf4 = arr.reshape(-1, 4)[-1].astype(np.float32)
                    else:
                        sf4 = arr[:4].astype(np.float32)
                    ds.set_metainfo(dict(scale_factor=sf4))

            # 一次性打印
            # if not getattr(self, '_printed_once', False):
            #     try:
            #         ds0 = data_samples[0]
            #         print(f'[CHK-DP] inputs.shape={tuple(inputs.shape)}')  # 期待 [B,T,3,H,W]
            #         print(f"[CHK-DP] img_shape={ds0.metainfo.get('img_shape', None)} "
            #               f"pad_shape={ds0.metainfo.get('pad_shape', None)} "
            #               f"scale_factor4={ds0.metainfo.get('scale_factor', None)}")
            #     except Exception as e:
            #         print('[CHK-DP] print error:', e)
            #     self._printed_once = True

            return dict(inputs=inputs, data_samples=data_samples)



        return super().forward(data, training)
