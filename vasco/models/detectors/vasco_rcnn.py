# mmdetection-3.3.0/projects/vasco/models/detectors/vasco_rcnn.py
import numpy as np
import torch
from typing import List
from mmdet.registry import MODELS
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from mmdet.structures import DetDataSample
from mmdet.models.detectors.two_stage import TwoStageDetector
from typing import List, Dict, Optional


@MODELS.register_module()
class VascoRCNN(FasterRCNN):
    """支持 [B,T,3,H,W]；验证期先禁 rescale 以排错。"""

    def _norm_sf4(self, sf):
        if sf is None:
            return None
        if isinstance(sf, torch.Tensor):
            sf = sf.detach().cpu().numpy()
        arr = np.asarray(sf, dtype=np.float32).reshape(-1)
        if arr.size == 2:
            return np.array([arr[0], arr[1], arr[0], arr[1]], dtype=np.float32)
        if arr.size == 4:
            return arr.astype(np.float32)
        if arr.size % 4 == 0:
            return arr.reshape(-1, 4)[-1].astype(np.float32)  # 取末帧
        return arr[:4].astype(np.float32)

    # def extract_feat(self, batch_inputs: torch.Tensor):
    #     """支持 [B,3,H,W] 与 [B,T,3,H,W]。
    #     - 4D: 清空时序上下文，走父类。
    #     - 5D: 按帧提特征，缓存 feats_list 给 RoIHead，仅返回末帧特征给 RPN/RCNN。
    #     """
    #     if batch_inputs.dim() == 4:
    #         feats = super().extract_feat(batch_inputs)  # tuple(levels)
    #         # 清空时序上下文
    #         if hasattr(self, 'roi_head') and hasattr(self.roi_head, 'set_temporal_ctx'):
    #             self.roi_head.set_temporal_ctx(None)
    #         return feats

    #     assert batch_inputs.dim() == 5, f'Expect [B,T,3,H,W], got {tuple(batch_inputs.shape)}'
    #     B, T, C, H, W = batch_inputs.shape

    #     feats_list: List[Tuple[torch.Tensor, ...]] = []
    #     # 逐帧跑 backbone+neck
    #     for t in range(T):
    #         x_t = batch_inputs[:, t, ...]                     # [B,3,H,W]
    #         feats_t = super().extract_feat(x_t)               # tuple(P2..P5)
    #         feats_list.append(feats_t)

    #     # 把整段上下文交给 RoIHead 做时序聚合
    #     if hasattr(self, 'roi_head') and hasattr(self.roi_head, 'set_temporal_ctx'):
    #         self.roi_head.set_temporal_ctx({'feats_list': feats_list})

    #     # RPN/RCNN 仍只用“末帧”特征，保持两阶段API不变
    #     return feats_list[-1]
    def extract_feat(self, imgs: torch.Tensor):
        """
        imgs: (B,T,3,H,W)
        返回：末帧 FPN (供 RPN/RCNN)，并把“每帧 FPN 序列”与 3D 特征传给 roi_head 上下文
        """
        assert imgs.dim() == 5
        feat3d_list = self.backbone(imgs)   # List[L] of (B,T,C,H,W)

        # ---- 末帧 → FPN（供 RPN/RCNN）
        feat2d_last = [x[:, -1].contiguous() for x in feat3d_list]     # (B,C_l,H,W)
        fpn_last = self.neck(feat2d_last)

        # ---- 全序列 → 每帧 FPN（供时序聚合器）
        T = int(feat3d_list[0].size(1))
        fpn_seq: List[List[torch.Tensor]] = []
        for t in range(T):
            per_t = [x[:, t].contiguous() for x in feat3d_list]        # List[L](B,C_l,H,W)
            fpn_t = self.neck(per_t)                                   # List[L](B,256,H,W)
            fpn_seq.append(fpn_t)

        # 传入 RoIHead 的时序上下文
        if hasattr(self.roi_head, 'set_temporal_ctx'):
            self.roi_head.set_temporal_ctx({'fpn_seq': fpn_seq, 'feat3d': feat3d_list, 'T': T})

        return fpn_last

    def predict(self,
                batch_inputs: torch.Tensor,
                data_samples: List[DetDataSample],
                rescale: bool = True):

        x = self.extract_feat(batch_inputs)

        # 末帧 4 元化 + 全量断言
        # bad_idx = []
        for i, ds in enumerate(data_samples):
            sf = ds.metainfo.get('scale_factor', None)
            arr = np.asarray(sf, dtype=np.float32).reshape(-1) if sf is not None else None
            if arr is None or arr.size != 4:
                bad_idx.append((i, None if arr is None else arr.shape))
            else:
                ds.set_metainfo(dict(scale_factor=arr.astype(np.float32)))
        # if bad_idx:
        #     print('[CHK-RPN-SF] bad scale_factor samples:', bad_idx)
        #     for i, _ in bad_idx:
        #         sf = data_samples[i].metainfo.get('scale_factor', None)
        #         arr = np.asarray(sf, np.float32).reshape(-1) if sf is not None else np.array([1,1,1,1], np.float32)
        #         if arr.size % 4 == 0: arr = arr.reshape(-1,4)[-1].astype(np.float32)
        #         else: arr = arr[:4].astype(np.float32) if arr.size>=4 else np.array([1,1,1,1], np.float32)
        #         data_samples[i].set_metainfo(dict(scale_factor=arr))
        # print('[CHK-RPN-SF] all scale_factors =', [np.asarray(ds.metainfo.get('scale_factor')).reshape(-1).size for ds in data_samples])

        # RPN（不 rescale，先保证链路稳定）
        rpn_results_list = self.rpn_head.predict(x, data_samples, rescale=False)
        if not hasattr(self, '_once_rpn_printed'):
            try:
                sizes = [len(p.proposals) if hasattr(p, 'proposals') else p.bboxes.shape[0]
                         for p in rpn_results_list]
                print('[CHK-rcnn-RPN] proposals_per_img=', sizes[:4])
                p0 = rpn_results_list[0]
                b = p0.proposals if hasattr(p0, 'proposals') else p0.bboxes
                s = p0.scores if hasattr(p0, 'scores') else None
                print('[CHK-rcnn-RPN] prop[:5]=', b[:5].detach().cpu().numpy())
                if s is not None:
                    print('[CHK-rcnn-RPN] prop_scores[:5]=', s[:5].detach().cpu().numpy())
            except Exception as e:
                print('[CHK-rcnn-RPN] print error:', e)
            self._once_rpn_printed = True

        # ROI（不 rescale，先让结果产出）
        results_list = self.roi_head.predict(x, rpn_results_list, data_samples, rescale=False)

        # RCNN 输出诊断
        if not hasattr(self, '_once_rcnn_printed'):
            try:
                ds0 = results_list[0]
                pred = getattr(ds0, 'pred_instances', None)
                if pred is None or pred.bboxes.numel() == 0:
                    print('[CHK-rcnn-RCNN] empty pred_instances')
                else:
                    print('[CHK-rcnn-RCNN] pred bboxes[:5]=', pred.bboxes[:5].detach().cpu().numpy())
                    if hasattr(pred, 'scores'):
                        print('[CHK-rcnn-RCNN] pred scores[:5]=', pred.scores[:5].detach().cpu().numpy())
                    print('[CHK-rcnn-RCNN] num_preds=', int(pred.bboxes.shape[0]))
            except Exception as e:
                print('[CHK-rcnn-RCNN] print error:', e)
            self._once_rcnn_printed = True

        return results_list
