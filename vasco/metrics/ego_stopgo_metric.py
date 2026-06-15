# -*- coding: utf-8 -*-
# projects/vasco/metrics/ego_stopgo_metric.py

from typing import Dict, List, Any, Optional
import numpy as np
import torch

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS
from mmdet.structures import DetDataSample


@METRICS.register_module()
class VascoEgoStopGoMetric(BaseMetric):
    """Frame-level ego stop/go metric (binary).

    GT is read from data_batch['data_samples'] (same style as VascoSegMetric),
    prediction is read from the model output `data_samples`.

    Expected fields:
      - GT (in gds): gt_stopgo  (0=stop, 1=go)
      - Pred (in ds): pred_stopgo (0/1) OR pred_stopgo_logits (2,)

    Reports (with prefix 'ego' by default):
      - stopgo_acc
      - stopgo_mF1
      - stopgo_f1_stop
      - stopgo_f1_go
      - stopgo_n (number of valid samples used)
      - stopgo_total (total samples seen)
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = 'ego'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self._dbg_once = False

    @staticmethod
    def _get_attr(obj: Any, key: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _to_int(x) -> Optional[int]:
        """Accept scalar tensor / numpy / python int."""
        if x is None:
            return None
        try:
            if hasattr(x, 'detach'):
                t = x.detach().cpu()
                if t.numel() == 0:
                    return None
                return int(t.reshape(-1)[0].item())
            return int(x)
        except Exception:
            return None

    def process(self,
                data_batch: Dict[str, Any],
                data_samples: List[DetDataSample]) -> None:
        logger = MMLogger.get_current_instance()
        gt_list = data_batch.get('data_samples') or [None] * len(data_samples)

        # one-time debug
        if not self._dbg_once and data_samples:
            ds0 = data_samples[0]
            g0 = gt_list[0] if gt_list else None
            p0 = self._get_attr(ds0, 'pred_stopgo', None)
            pl0 = self._get_attr(ds0, 'pred_stopgo_logits', None)
            g0s = self._get_attr(g0, 'gt_stopgo', None)

            logger.info(
                f"[EGO-DBG] type={type(ds0).__name__} "
                f"pred_stopgo={p0 is not None} pred_logits={pl0 is not None} "
                f"gt_stopgo={g0s is not None}"
            )
            self._dbg_once = True

        # We store per-sample confusion increments; invalid samples are marked valid=0
        for ds, gds in zip(data_samples, gt_list):
            # --- GT ---
            gt = self._to_int(self._get_attr(gds, 'gt_stopgo', None))

            # --- Pred ---
            pred_i = self._to_int(self._get_attr(ds, 'pred_stopgo', None))
            if pred_i is None:
                logits = self._get_attr(ds, 'pred_stopgo_logits', None)
                if logits is not None and hasattr(logits, 'detach'):
                    lg = logits.detach().cpu().reshape(-1)
                    if lg.numel() >= 2:
                        pred_i = int(torch.argmax(lg).item())

            if gt is None or pred_i is None:
                self.results.append(dict(valid=0, gt=-1, pred=-1))
                continue

            gt = 1 if gt == 1 else 0
            pred_i = 1 if pred_i == 1 else 0
            self.results.append(dict(valid=1, gt=gt, pred=pred_i))

    @staticmethod
    def _f1_for_class(tp: float, fp: float, fn: float, eps: float = 1e-9) -> float:
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        return 2.0 * prec * rec / (prec + rec + eps)

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        if not results:
            return dict(
                stopgo_acc=0.0,
                stopgo_mF1=0.0,
                stopgo_f1_stop=0.0,
                stopgo_f1_go=0.0,
                stopgo_n=0.0,
                stopgo_total=0.0,
            )

        total = len(results)
        valid_rows = [r for r in results if r.get('valid', 0) == 1]
        n = len(valid_rows)

        if n == 0:
            return dict(
                stopgo_acc=0.0,
                stopgo_mF1=0.0,
                stopgo_f1_stop=0.0,
                stopgo_f1_go=0.0,
                stopgo_n=0.0,
                stopgo_total=float(total),
            )

        gts = np.array([r['gt'] for r in valid_rows], dtype=np.int64)
        preds = np.array([r['pred'] for r in valid_rows], dtype=np.int64)

        acc = float((gts == preds).mean())

        cm = np.zeros((2, 2), dtype=np.int64)
        for gt, pr in zip(gts, preds):
            cm[int(gt), int(pr)] += 1

        # class 0 (stop)
        tp0 = float(cm[0, 0])
        fp0 = float(cm[1, 0])
        fn0 = float(cm[0, 1])
        f1_stop = self._f1_for_class(tp0, fp0, fn0)

        # class 1 (go)
        tp1 = float(cm[1, 1])
        fp1 = float(cm[0, 1])
        fn1 = float(cm[1, 0])
        f1_go = self._f1_for_class(tp1, fp1, fn1)

        mf1 = 0.5 * (f1_stop + f1_go)

        # precision / recall / count
        eps = 1e-9

        precision_stop = tp0 / (tp0 + fp0 + eps)
        recall_stop = tp0 / (tp0 + fn0 + eps)

        precision_go = tp1 / (tp1 + fp1 + eps)
        recall_go = tp1 / (tp1 + fn1 + eps)

        gt_stop = float(cm[0, :].sum())
        pred_stop = float(cm[:, 0].sum())
        gt_go = float(cm[1, :].sum())
        pred_go = float(cm[:, 1].sum())

        return dict(
            stopgo_acc=float(acc),
            stopgo_mF1=float(mf1),
            stopgo_f1_stop=float(f1_stop),
            stopgo_f1_go=float(f1_go),

            # most important for your deployment problem
            stopgo_recall_stop=float(recall_stop),
            stopgo_precision_stop=float(precision_stop),
            stopgo_gt_stop=float(gt_stop),
            stopgo_pred_stop=float(pred_stop),
            stopgo_fn_stop=float(fn0),

            # optional but useful
            stopgo_recall_go=float(recall_go),
            stopgo_precision_go=float(precision_go),
            stopgo_gt_go=float(gt_go),
            stopgo_pred_go=float(pred_go),

            stopgo_n=float(n),
            stopgo_total=float(total),
        )