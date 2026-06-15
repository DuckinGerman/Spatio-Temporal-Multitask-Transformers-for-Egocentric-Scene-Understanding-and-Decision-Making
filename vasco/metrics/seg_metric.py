# projects/vasco/metrics/seg_metric.py
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS
from mmdet.structures import DetDataSample


def _fast_hist(
    gt: np.ndarray,
    pr: np.ndarray,
    C: int,
    ignore: int = 255,
    bg_id: int = 6,
) -> np.ndarray:
    """gt, pr: (H,W) int
    C: evaluated foreground classes (0..C-1), here 6 for labels 0..5
    ignore: ignored label id (255)
    bg_id: explicit pseudo-background label id (6), excluded from evaluation

    Only valid foreground pixels with gt in [0..C-1] are evaluated.
    GT pixels equal to bg_id or ignore are excluded.
    Predictions outside [0..C-1] are ignored automatically.
    """
    mask = (gt != ignore) & (gt != bg_id)
    mask &= (gt >= 0) & (gt < C)
    mask &= (pr >= 0) & (pr < C)

    if not np.any(mask):
        return np.zeros((C, C), dtype=np.int64)

    gt_v = gt[mask].astype(np.int64)
    pr_v = pr[mask].astype(np.int64)
    hist = np.bincount(gt_v * C + pr_v, minlength=C * C).reshape(C, C)
    return hist

@METRICS.register_module()
class VascoSegMetric(BaseMetric):
    """mIoU for stuff segmentation with an explicit pseudo-background label.

    - Evaluate `num_classes` foreground categories (0..num_classes-1), here 6 classes (0..5).
    - Pseudo-background label id is `bg_id` (default 6), excluded from evaluation.
    - Ignore label is `ignore_index` (default 255).

    The metric assumes GT and prediction are label maps in the same label space.
    Any GT pixel outside [0..num_classes-1], equal to `bg_id`, or equal to `ignore_index`
    is ignored automatically. Predictions outside [0..num_classes-1] are also ignored automatically.
    """

    def __init__(self,
                 num_classes: int = 6,                 # evaluate foreground classes 0..num_classes-1
                 bg_id: int = 6,                       # explicit pseudo-background label id (excluded)
                 ignore_index: int = 255,              # ignore label id
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = 'seg'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.C = int(num_classes)                      # number of evaluated foreground classes
        self.bg_id = int(bg_id)                        # pseudo-background label id (excluded)
        self.ignore = int(ignore_index)                # ignore label id
        self._dbg_once = False

    def _get_pred_mask(self, ds: Any):
        if isinstance(ds, dict):
            return ds.get('pred_sem_seg', None)
        return getattr(ds, 'pred_sem_seg', None)

    def _get_gt_mask(self, gds: Any):
        if isinstance(gds, dict):
            return gds.get('gt_sem_seg', None)
        return getattr(gds, 'gt_sem_seg', None)

    def process(self,
                data_batch: Dict[str, Any],
                data_samples: List[DetDataSample]) -> None:
        logger = MMLogger.get_current_instance()
        gt_list = data_batch.get('data_samples') or [None] * len(data_samples)

        # 只打印一次头信息
        if not self._dbg_once and data_samples:
            ds0 = data_samples[0]
            g0 = gt_list[0]
            p0 = self._get_pred_mask(ds0)
            g0m = self._get_gt_mask(g0)
            pshape = tuple(getattr(p0, 'data', None).shape) if p0 is not None and hasattr(p0, 'data') else None
            gshape = tuple(getattr(g0m, 'data', None).shape) if g0m is not None and hasattr(g0m, 'data') else None
            logger.info(
                f"[SEG-DBG] type={type(ds0).__name__} pred_sem_seg={p0 is not None} "
                f"p_shape={pshape} gt_sem_seg={g0m is not None} g_shape={gshape}"
            )
            self._dbg_once = True

        def _to_hw_np(x) -> Optional[np.ndarray]:
            """接受 dict/PixelData/np/torch，返回 (H,W) np.int64；若无则 None。"""
            if x is None:
                return None
            # 取 data
            if isinstance(x, dict):
                x = x.get('data', None)
                if x is None:
                    return None
            elif hasattr(x, 'data'):
                x = x.data
            # 转 tensor
            t = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
            if t.ndim == 3:  # (1,H,W) -> (H,W)
                t = t[0]
            return t.long().cpu().numpy()

        for ds, gds in zip(data_samples, gt_list):
            pred_pd = ds.get('pred_sem_seg', None) if isinstance(ds, dict) else getattr(ds, 'pred_sem_seg', None)
            gt_pd   = gds.get('gt_sem_seg',  None) if isinstance(gds, dict) else getattr(gds, 'gt_sem_seg',  None)

            p = _to_hw_np(pred_pd)  # prediction labels: expected in {0..5}; any bg/out-of-range label is ignored
            g = _to_hw_np(gt_pd)    # GT labels: expected in {0..5, 6, ignore=255}

            # 始终 append 'hist'，避免 KeyError
            if p is None or g is None:
                self.results.append({'hist': np.zeros((self.C, self.C), dtype=np.int64)})
                continue

            # 尺寸不一致：把 pred 最近邻对到 GT 尺寸
            if p.shape != g.shape:
                p_t = torch.as_tensor(p)[None, None].float()
                p_t = F.interpolate(p_t, size=g.shape[-2:], mode='nearest')[0, 0].long().cpu().numpy()
                p = p_t

            hist = _fast_hist(
                g.astype(np.int64),
                p.astype(np.int64),
                self.C,
                ignore=self.ignore,
                bg_id=self.bg_id,
            )
            self.results.append({'hist': hist})

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {'mIoU': 0.0}
        agg = None
        for r in results:
            h = r.get('hist', None)
            if h is None:
                continue
            agg = h if agg is None else (agg + h)
        if agg is None:
            return {'mIoU': 0.0}
        inter = np.diag(agg)
        union = agg.sum(1) + agg.sum(0) - inter
        valid = union > 0
        iou = np.zeros(self.C, dtype=np.float64)
        iou[valid] = inter[valid] / np.clip(union[valid], 1e-6, None)
        return {'mIoU': float(np.nanmean(iou[valid]) if np.any(valid) else 0.0)}