from typing import Dict, Any, List, Optional
import numpy as np
import torch
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData, PixelData

@TRANSFORMS.register_module()
class PackVascoInputs:
    def __init__(self, meta_keys: Optional[List[str]] = None):
        self.meta_keys = meta_keys or [
            'img_id','ori_shape','img_shape','pad_shape','scale_factor',
            'clip_len','img_path','frame_id',
            'crop_offset','video_id'
        ]
        self._printed_once = False
        self._printed_gt_once = False
        self._printed_keys = False

    @staticmethod
    def _sf_to4(sf):
        if sf is None: return None
        arr = np.asarray(sf, dtype=np.float32).reshape(-1)
        if arr.size == 2:  return np.array([arr[0],arr[1],arr[0],arr[1]], np.float32)
        if arr.size == 4:  return arr.astype(np.float32)
        if arr.size % 4 == 0: return arr.reshape(-1,4)[-1].astype(np.float32)
        return arr[:4].astype(np.float32)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # --- meta ---
        sf4 = self._sf_to4(results.get('scale_factor', None))
        if sf4 is not None:
            results['scale_factor'] = sf4
        # normalize crop offset (dx, dy) used for inverse mapping in RoIHead/metrics
        off = results.get('crop_offset', (0, 0))
        if off is None:
            off = (0, 0)
        if isinstance(off, (list, tuple, np.ndarray)):
            off = np.asarray(off, dtype=np.float32).reshape(-1)
            if off.size >= 2:
                off = (float(off[0]), float(off[1]))
            else:
                off = (0.0, 0.0)
        else:
            off = (0.0, 0.0)
        results['crop_offset'] = off
        meta = {
            'img_id':      results.get('img_id', -1),
            'ori_shape':   results.get('ori_shape', results.get('img_shape', None)),
            'img_shape':   results.get('img_shape', None),
            'pad_shape':   results.get('pad_shape', None),
            'scale_factor': results.get('scale_factor', None),
            'clip_len':    results.get('clip_len', None),
            'img_path':    results.get('img_path', None),
            'frame_id':    results.get('frame_id', None),
            'crop_offset': results.get('crop_offset', (0.0, 0.0)),
            'video_id':    results.get('video_id', None),
        }
        data_sample = DetDataSample(metainfo=meta)

        # --- GT（仅末帧）：始终产出 bboxes/labels/actions/dirs ---
        def _np(x): return None if x is None else np.asarray(x)
        bboxes  = _np(results.get('gt_bboxes', None))
        labels  = _np(results.get('gt_bboxes_labels', results.get('gt_labels', None)))
        actions = _np(results.get('gt_actions', None))
        dirs    = _np(results.get('gt_dirs', None))

        if bboxes is not None: n = int(bboxes.shape[0])
        elif labels is not None: n = int(labels.shape[0])
        elif actions is not None: n = int(actions.shape[0])
        elif dirs is not None: n = int(dirs.shape[0])
        else: n = 0

        if bboxes is None:  bboxes = np.zeros((n,4), np.float32)
        if labels is None:  labels = np.zeros((n,), np.int64)
        if actions is None: actions = np.zeros((n,), np.int64)
        if dirs is None:    dirs    = np.zeros((n,), np.int64)

        mlen = min(len(bboxes), len(labels), len(actions), len(dirs))
        bboxes, labels, actions, dirs = bboxes[:mlen], labels[:mlen], actions[:mlen], dirs[:mlen]

        # 裁边+去退化
        H=W=None
        if meta['img_shape'] is not None:
            H,W = int(meta['img_shape'][0]), int(meta['img_shape'][1])
        elif meta['pad_shape'] is not None:
            H,W = int(meta['pad_shape'][0]), int(meta['pad_shape'][1])
        if mlen>0:
            bboxes = bboxes.astype(np.float32, copy=True)
            if (H is not None) and (W is not None):
                bboxes[:,0::2] = np.clip(bboxes[:,0::2], 0, W)
                bboxes[:,1::2] = np.clip(bboxes[:,1::2], 0, H)
            w = bboxes[:,2]-bboxes[:,0]; h = bboxes[:,3]-bboxes[:,1]
            keep = (w>0) & (h>0)
            if keep.sum()!=keep.size:
                bboxes  = bboxes[keep]; labels=labels[keep]; actions=actions[keep]; dirs=dirs[keep]

        inst = InstanceData()
        inst.bboxes  = torch.as_tensor(bboxes, dtype=torch.float32)
        inst.labels  = torch.as_tensor(labels, dtype=torch.long)
        inst.actions = torch.as_tensor(actions, dtype=torch.long)
        inst.dirs    = torch.as_tensor(dirs, dtype=torch.long)
        data_sample.gt_instances = inst

        # ---- Ego input + stop/go GT (frame-level, from unified.json images fields) ----
        if 'ego_choice_id' in results and results['ego_choice_id'] is not None:
            data_sample.ego_choice_id = torch.as_tensor(int(results['ego_choice_id']), dtype=torch.long)
        if 'gt_stopgo' in results and results['gt_stopgo'] is not None:
            data_sample.gt_stopgo = torch.as_tensor(int(results['gt_stopgo']), dtype=torch.long)

        # 语义
        if 'gt_sem_seg' in results and results['gt_sem_seg'] is not None:
            seg = results['gt_sem_seg']
            if not torch.is_tensor(seg): seg = torch.as_tensor(seg, dtype=torch.long)
            if seg.ndim==2: seg = seg.unsqueeze(0)  # [1,H,W]
            data_sample.gt_sem_seg = PixelData(data=seg)

        data = dict(
            inputs=dict(imgs=results['imgs'], img_path_list=results['img_path_list']),
            data_samples=data_sample
        )

        # if not self._printed_keys:
        #     print('[CHK-pack-KEYS]', [k for k in results.keys() if k.startswith('gt_') or k.endswith('_bboxes')])
        #     self._printed_keys = True
        # if not self._printed_gt_once:
        #     print('[CHK-pack-GT] n_gt=', int(inst.bboxes.shape[0]),
        #           'labels[:5]=', inst.labels[:5].cpu().numpy() if inst.labels.numel() else [])
        #     self._printed_gt_once = True
        # if not self._printed_once:
        #     print('[CHK-pack-PACK] clip_len(T)=', meta.get('clip_len'),
        #           'img_shape=', meta.get('img_shape'),
        #           'pad_shape=', meta.get('pad_shape'),
        #           'scale_factor=', np.asarray(meta.get('scale_factor')).ravel() if meta.get('scale_factor') is not None else None,
        #           'img_path=', meta.get('img_path'))
        #     self._printed_once = True
        if not hasattr(self, '_gt_dbg'):
            meta = data_sample.metainfo
            gi = data_sample.gt_instances
            seg = getattr(data_sample, 'gt_sem_seg', None)
            print('[PK]', list(results.keys()))
            print(f"[PK] ori={meta.get('ori_shape')} img={meta.get('img_shape')} sf={meta.get('scale_factor')} off={meta.get('crop_offset')}")
            n = int(gi.bboxes.shape[0]) if hasattr(gi,'bboxes') else -1
            labs = gi.labels[:min(n,8)].cpu().tolist() if hasattr(gi,'labels') else []
            print(f"[PK] GT boxes={n} labels_head={labs}")
            print(f"[PK] GT seg= {seg is not None} shape={tuple(seg.data.shape) if hasattr(seg,'data') else None}")

            ego_id = getattr(data_sample, 'ego_choice_id', None)
            sg = getattr(data_sample, 'gt_stopgo', None)
            print(f"[PK] ego_choice_id={int(ego_id) if ego_id is not None else None} gt_stopgo={int(sg) if sg is not None else None}")
            self._gt_dbg = True

        return data
