from typing import List, Dict, Optional, Any
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS
from mmdet.structures import DetDataSample

def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [Na,4], b: [Nb,4]  (xyxy 同一坐标系)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:],  b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-6, None)

def _macro_f1(conf: np.ndarray) -> float:
    C = conf.shape[0]
    f1s = []
    for c in range(C):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1s.append(2 * prec * rec / (prec + rec + 1e-6))
    return float(np.mean(f1s) if len(f1s) else 0.0)

@METRICS.register_module()
class VascoActionDirMetric(BaseMetric):
    """Action/Direction metric.

    Match predictions to GT by class and IoU>=thr, then compute confusion matrices:
    - Action: 2-way (static/dynamic). GT raw ids are collapsed to binary.
    - Direction: keeps the original 6-way confusion for backward compatibility.
    - Conditional direction: evaluates state-conditioned 5+4 direction:
        static  -> front/right/left/away/nodirection
        dynamic -> right/left/near/away
    """
    def __init__(self,
                 iou_thr: float = 0.5,
                 num_actions: int = 2,
                 num_dirs: int = 6,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = 'vasco'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = float(iou_thr)
        self.num_actions = int(num_actions)
        self.num_dirs = int(num_dirs)

        self._dbg_once = False
        self._sum_once = False
        self._bbox_dbg_once = False

    def _get_pred_inst(self, ds: Any):
        # 兼容 dict / DetDataSample / InstanceData
        if isinstance(ds, dict):
            return ds.get('pred_instances', None)
        return ds.pred_instances if hasattr(ds, 'pred_instances') else ds

    def _get_gt_inst(self, gds: Any):
        if isinstance(gds, dict):
            return gds.get('gt_instances', None)
        return getattr(gds, 'gt_instances', None)

    def process(self, data_batch: Dict, data_samples: List[DetDataSample]) -> None:
        """pred 从 data_samples[i] 读（兼容 dict/DetDataSample/InstanceData）；
        gt   从 data_batch['data_samples'][i] 读（兼容 dict/对象）。
        gt 框从裁剪后坐标映射回原图： (x,y)=(x'+off)/[sw,sh] ；pred 已是原图坐标（ROI里已回映射）。
        """
        logger = MMLogger.get_current_instance()
        gt_list = data_batch.get('data_samples') or [None] * len(data_samples)

        def _get_pred(ds):
            # 兼容 dict/对象；返回一个 dict {'bboxes','labels','actions','dirs'} 都是 numpy
            if isinstance(ds, dict):
                inst = ds.get('pred_instances', None)
            else:
                inst = ds.pred_instances if hasattr(ds, 'pred_instances') else ds
            if inst is None:
                return None
            def to_np(x):
                if x is None: return None
                if isinstance(x, np.ndarray): return x
                if torch.is_tensor(x): return x.detach().cpu().numpy()
                return np.asarray(x)
            if isinstance(inst, dict):
                return {
                    'bboxes': to_np(inst.get('bboxes', None)),
                    'labels': to_np(inst.get('labels', None)),
                    'actions':to_np(inst.get('actions', None)),
                    'dirs':   to_np(inst.get('dirs', None)),
                }
            # InstanceData / DetDataSample.pred_instances
            return {
                'bboxes': to_np(getattr(inst, 'bboxes', None)),
                'labels': to_np(getattr(inst, 'labels', None)),
                'actions':to_np(getattr(inst, 'actions', None)),
                'dirs':   to_np(getattr(inst, 'dirs', None)),
            }

        def _get_gt(gds):
            # 兼容 dict/对象；返回同样结构（bboxes/labels/actions/dirs 均为 numpy）
            if isinstance(gds, dict):
                inst = gds.get('gt_instances', None)
            else:
                inst = getattr(gds, 'gt_instances', None)
            if inst is None:
                return None
            def to_np(x):
                if x is None: return None
                if isinstance(x, np.ndarray): return x
                if torch.is_tensor(x): return x.detach().cpu().numpy()
                return np.asarray(x)
            if isinstance(inst, dict):
                return {
                    'bboxes': to_np(inst.get('bboxes', None)),
                    'labels': to_np(inst.get('labels', None)),
                    'actions':to_np(inst.get('actions', None)),
                    'dirs':   to_np(inst.get('dirs', None)),
                }
            return {
                'bboxes': to_np(getattr(inst, 'bboxes', None)),
                'labels': to_np(getattr(inst, 'labels', None)),
                'actions':to_np(getattr(inst, 'actions', None)),
                'dirs':   to_np(getattr(inst, 'dirs', None)),
            }

        def _get_meta(obj):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj.get('metainfo', {}) or {}
            return getattr(obj, 'metainfo', {}) or {}

        # 批级统计一次（兼容 dict）
        if not self._sum_once:
            tot_pred = 0
            for ds in data_samples:
                p = _get_pred(ds)
                tot_pred += 0 if (p is None or p['bboxes'] is None) else int(p['bboxes'].shape[0])
            logger.info(f"[AD-DBG] batch_pred_total={tot_pred}")
            self._sum_once = True

        # 第1图头信息
        if not self._dbg_once and data_samples:
            ds0 = data_samples[0]; g0 = (data_batch.get('data_samples') or [None])[0]
            p0 = _get_pred(ds0);   gt0 = _get_gt(g0)
            pN = 0 if (p0 is None or p0['bboxes'] is None) else int(p0['bboxes'].shape[0])
            gN = 0 if (gt0 is None or gt0['bboxes'] is None) else int(gt0['bboxes'].shape[0])
            logger.info(f"[AD-DBG] type={type(ds0).__name__} predN={pN} hasA={p0 is not None and p0['actions'] is not None} hasD={p0 is not None and p0['dirs'] is not None} gtN={gN}")
            self._dbg_once = True

        # 主循环
        for ds, gds in zip(data_samples, gt_list):
            zero = {
                'conf_a': None, 'conf_d': None,
                'conf_d_static': None,
                'conf_d_dynamic': None,
                'action_samples': 0, 'dir_samples': 0,
                'static_dir_samples': 0,
                'dynamic_dir_samples': 0,
            }
            pred = _get_pred(ds)
            gt   = _get_gt(gds)
            if (pred is None or gt is None or
                pred['bboxes'] is None or pred['bboxes'].size==0 or
                gt['bboxes']   is None or gt['bboxes'].size==0 or
                pred['actions'] is None or pred['dirs'] is None or
                gt['labels'] is None or gt['actions'] is None or gt['dirs'] is None):
                self.results.append(zero); continue

            # pred 已是原图坐标（ROI 已回映射）
            pb, pl, pa, pd = pred['bboxes'], pred['labels'], pred['actions'], pred['dirs']

            # gt is usually in network-input coordinates, while pred bboxes are already restored
            # to original-image coordinates by the RoIHead. Prefer GT metainfo, then pred metainfo.
            # If scale_factor is missing or reset to [1,1,1,1], recover it from img_shape/ori_shape.
            meta = _get_meta(gds)
            if not meta:
                meta = _get_meta(ds)

            sf = meta.get('scale_factor', [1, 1, 1, 1])
            if torch.is_tensor(sf):
                sf = sf.detach().cpu().numpy()
            sf = np.asarray(sf, dtype=np.float32).reshape(-1)
            if sf.size < 2:
                sf = np.array([1, 1, 1, 1], dtype=np.float32)

            ori_shape = meta.get('ori_shape', None)
            img_shape = meta.get('img_shape', None)
            if img_shape is None:
                img_shape = meta.get('pad_shape', None)

            # Recover resize scale when scale_factor is invalid but shapes are available.
            # img_shape/ori_shape are stored as (H, W, C) or (H, W).
            try:
                if ori_shape is not None and img_shape is not None:
                    ori_h, ori_w = float(ori_shape[0]), float(ori_shape[1])
                    img_h, img_w = float(img_shape[0]), float(img_shape[1])
                    if ori_h > 0 and ori_w > 0:
                        recovered_sw = img_w / ori_w
                        recovered_sh = img_h / ori_h
                        # If scale_factor is identity but original and input sizes differ,
                        # use recovered scales for mapping gt input coords back to original coords.
                        if abs(float(sf[0]) - 1.0) < 1e-6 and abs(float(sf[1]) - 1.0) < 1e-6:
                            if abs(recovered_sw - 1.0) > 1e-3 or abs(recovered_sh - 1.0) > 1e-3:
                                sf = np.array([recovered_sw, recovered_sh, recovered_sw, recovered_sh], dtype=np.float32)
            except Exception:
                pass

            # crop_offset stored as (y0, x0) in pixels (top, left) after resize.
            off = meta.get('crop_offset', (0.0, 0.0))
            if off is None:
                off = (0.0, 0.0)
            if torch.is_tensor(off):
                off = off.detach().cpu().numpy()
            off = np.asarray(off, dtype=np.float32).reshape(-1)
            if off.size >= 2:
                y0 = float(off[0])
                x0 = float(off[1])
            else:
                y0, x0 = 0.0, 0.0

            sw, sh = float(sf[0] or 1.0), float(sf[1] or 1.0)
            gb_raw = gt['bboxes'].copy()
            gb = gb_raw.copy(); gl = gt['labels']; ga = gt['actions']; gd = gt['dirs']
            gb[:,[0,2]] = (gb[:,[0,2]] + float(x0)) / (sw if sw>0 else 1.0)
            gb[:,[1,3]] = (gb[:,[1,3]] + float(y0)) / (sh if sh>0 else 1.0)

            if not self._bbox_dbg_once:
                try:
                    logger.info("[AD-BBOX-DBG] ===== begin =====")
                    logger.info(f"[AD-BBOX-DBG] pred_boxes shape={pb.shape}, gt_boxes shape={gb.shape}")
                    logger.info(f"[AD-BBOX-DBG] scale_factor={sf.tolist()}, crop_offset_yx={[float(y0), float(x0)]}")
                    logger.info(f"[AD-BBOX-DBG] ori_shape={meta.get('ori_shape', None)}, img_shape={meta.get('img_shape', None)}, pad_shape={meta.get('pad_shape', None)}")
                    if pb.size > 0:
                        logger.info(f"[AD-BBOX-DBG] pred min xyxy={pb.min(axis=0).tolist()}")
                        logger.info(f"[AD-BBOX-DBG] pred max xyxy={pb.max(axis=0).tolist()}")
                        logger.info(f"[AD-BBOX-DBG] pred first3={pb[:min(3, len(pb))].tolist()}")
                    if gb_raw.size > 0:
                        logger.info(f"[AD-BBOX-DBG] gt_raw min xyxy={gb_raw.min(axis=0).tolist()}")
                        logger.info(f"[AD-BBOX-DBG] gt_raw max xyxy={gb_raw.max(axis=0).tolist()}")
                        logger.info(f"[AD-BBOX-DBG] gt_raw first3={gb_raw[:min(3, len(gb_raw))].tolist()}")
                    if gb.size > 0:
                        logger.info(f"[AD-BBOX-DBG] gt_mapped min xyxy={gb.min(axis=0).tolist()}")
                        logger.info(f"[AD-BBOX-DBG] gt_mapped max xyxy={gb.max(axis=0).tolist()}")
                        logger.info(f"[AD-BBOX-DBG] gt_mapped first3={gb[:min(3, len(gb))].tolist()}")
                    if pb.size > 0 and gb.size > 0:
                        iou_dbg = _pairwise_iou(pb, gb)
                        logger.info(f"[AD-BBOX-DBG] max_iou_per_gt={iou_dbg.max(axis=0).tolist()}")
                        logger.info(f"[AD-BBOX-DBG] gt_num_iou_ge_thr={(iou_dbg.max(axis=0) >= self.iou_thr).sum().item()} / {gb.shape[0]}")
                    logger.info("[AD-BBOX-DBG] ===== end =====")
                except Exception as e:
                    logger.info(f"[AD-BBOX-DBG] failed: {e}")
                self._bbox_dbg_once = True

            conf_a = np.zeros((self.num_actions, self.num_actions), np.float64)
            conf_d = np.zeros((self.num_dirs,   self.num_dirs),   np.float64)
            samples_a = 0
            samples_d = 0

            # Conditional direction confusion matrices.
            # Original DIR_CLASSES = ['front', 'right', 'left', 'near', 'away', 'nodirection']
            # Static 5-way:  front/right/left/away/nodirection -> 0/1/2/3/4
            # Dynamic 4-way: right/left/near/away         -> 0/1/2/3
            static_dir_map = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4}
            dynamic_dir_map = {1: 0, 2: 1, 3: 2, 4: 3}
            conf_d_static = np.zeros((5, 5), np.float64)
            conf_d_dynamic = np.zeros((4, 4), np.float64)
            samples_d_static = 0
            samples_d_dynamic = 0

            classes = np.unique(np.concatenate([pl, gl])) if (pl.size or gl.size) else []
            for c in classes:
                pi = np.where(pl == c)[0]; gi = np.where(gl == c)[0]
                if pi.size == 0 or gi.size == 0: continue
                iou = _pairwise_iou(pb[pi], gb[gi])
                used_p = np.zeros(len(pi), bool); used_g = np.zeros(len(gi), bool)
                while iou.size:
                    r, cj = divmod(np.argmax(iou), iou.shape[1])
                    if iou[r, cj] < self.iou_thr: break
                    if used_p[r] or used_g[cj]:
                        iou[r, :] = -1; iou[:, cj] = -1; continue
                    used_p[r] = True; used_g[cj] = True

                    # ---- action: collapse raw gt action ids -> binary (static/dynamic) ----
                    # raw dynamic: {0,3} -> 1; raw static: {1,2,4(noaction)} -> 0
                    ga_raw = int(ga[gi[cj]])
                    if ga_raw == 0 or ga_raw == 3:
                        ga_id = 1
                    else:
                        ga_id = 0

                    pa_id = int(pa[pi[r]])
                    if 0 <= ga_id < self.num_actions and 0 <= pa_id < self.num_actions:
                        conf_a[ga_id, pa_id] += 1
                        samples_a += 1

                    # ---- direction: original 6-way metric for backward compatibility ----
                    gd_id = int(gd[gi[cj]])
                    pd_id = int(pd[pi[r]])
                    if (0 <= gd_id < self.num_dirs) and (0 <= pd_id < self.num_dirs):
                        conf_d[gd_id, pd_id] += 1
                        samples_d += 1

                    # ---- conditional direction: state-conditioned 5+4 metric ----
                    if ga_id == 0:
                        # static: front/right/left/away/nodirection
                        if gd_id in static_dir_map and pd_id in static_dir_map:
                            conf_d_static[static_dir_map[gd_id], static_dir_map[pd_id]] += 1
                            samples_d_static += 1
                    else:
                        # dynamic: right/left/near/away
                        if gd_id in dynamic_dir_map and pd_id in dynamic_dir_map:
                            conf_d_dynamic[dynamic_dir_map[gd_id], dynamic_dir_map[pd_id]] += 1
                            samples_d_dynamic += 1

                    iou[r, :] = -1; iou[:, cj] = -1

            if samples_a == 0 and samples_d == 0:
                self.results.append(zero)
            else:
                self.results.append({
                    'conf_a': conf_a if samples_a > 0 else None,
                    'conf_d': conf_d if samples_d > 0 else None,
                    'conf_d_static': conf_d_static if samples_d_static > 0 else None,
                    'conf_d_dynamic': conf_d_dynamic if samples_d_dynamic > 0 else None,
                    'action_samples': int(samples_a),
                    'dir_samples': int(samples_d),
                    'static_dir_samples': int(samples_d_static),
                    'dynamic_dir_samples': int(samples_d_dynamic),
                })

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        # ---- action (aggregate confusion) ----
        conf_a_sum = None
        tot_a = 0
        for r in results:
            ca = r.get('conf_a', None)
            if ca is None:
                continue
            conf_a_sum = ca if conf_a_sum is None else (conf_a_sum + ca)
            tot_a += int(r.get('action_samples', 0))

        if conf_a_sum is not None and tot_a > 0:
            acc_a = float(np.trace(conf_a_sum) / (conf_a_sum.sum() + 1e-6))
            f1_a = float(_macro_f1(conf_a_sum))
        else:
            acc_a = 0.0
            f1_a = 0.0

        # ---- direction (aggregate confusion) ----
        conf_d_sum = None
        tot_d = 0
        for r in results:
            cd = r.get('conf_d', None)
            if cd is None:
                continue
            conf_d_sum = cd if conf_d_sum is None else (conf_d_sum + cd)
            tot_d += int(r.get('dir_samples', 0))

        if conf_d_sum is not None and tot_d > 0:
            acc_d = float(np.trace(conf_d_sum) / (conf_d_sum.sum() + 1e-6))
            f1_d = float(_macro_f1(conf_d_sum))
        else:
            acc_d = 0.0
            f1_d = 0.0

        # ---- conditional direction: static 5-way + dynamic 4-way ----
        conf_ds_sum = None
        conf_dd_sum = None
        tot_ds = 0
        tot_dd = 0
        for r in results:
            cds = r.get('conf_d_static', None)
            cdd = r.get('conf_d_dynamic', None)
            if cds is not None:
                conf_ds_sum = cds if conf_ds_sum is None else (conf_ds_sum + cds)
                tot_ds += int(r.get('static_dir_samples', 0))
            if cdd is not None:
                conf_dd_sum = cdd if conf_dd_sum is None else (conf_dd_sum + cdd)
                tot_dd += int(r.get('dynamic_dir_samples', 0))

        if conf_ds_sum is not None and tot_ds > 0:
            acc_ds = float(np.trace(conf_ds_sum) / (conf_ds_sum.sum() + 1e-6))
            f1_ds = float(_macro_f1(conf_ds_sum))
        else:
            acc_ds = 0.0
            f1_ds = 0.0

        if conf_dd_sum is not None and tot_dd > 0:
            acc_dd = float(np.trace(conf_dd_sum) / (conf_dd_sum.sum() + 1e-6))
            f1_dd = float(_macro_f1(conf_dd_sum))
        else:
            acc_dd = 0.0
            f1_dd = 0.0

        tot_cond_d = tot_ds + tot_dd
        if tot_cond_d > 0:
            correct_cond_d = 0.0
            total_cond_d = 0.0
            if conf_ds_sum is not None:
                correct_cond_d += float(np.trace(conf_ds_sum))
                total_cond_d += float(conf_ds_sum.sum())
            if conf_dd_sum is not None:
                correct_cond_d += float(np.trace(conf_dd_sum))
                total_cond_d += float(conf_dd_sum.sum())
            cond_acc_d = float(correct_cond_d / (total_cond_d + 1e-6))
            # cond_f1_d = float((f1_ds * tot_ds + f1_dd * tot_dd) / (tot_cond_d + 1e-6))
            cond_f1_d = float((f1_ds + f1_dd) / 2 )
        else:
            cond_acc_d = 0.0
            cond_f1_d = 0.0

        return {
            'action_acc': float(acc_a),
            'action_mF1': float(f1_a),
            'dir_acc': float(cond_acc_d),
            'dir_mF1': float(cond_f1_d),
            'action_samples': int(tot_a),
            'dir_samples': int(tot_cond_d),
        }