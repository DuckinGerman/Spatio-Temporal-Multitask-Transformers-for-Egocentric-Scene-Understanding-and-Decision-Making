# projects/vasco/pipelines/seq_resize_flip_pad.py
from typing import Dict, Any, List, Tuple
import numpy as np
import mmcv
import cv2
from mmdet.registry import TRANSFORMS


def _compute_resize(h: int, w: int, short_choices: Tuple[int, ...], max_long: int) -> Tuple[int, int, float, float]:
    """Return new_h, new_w, scale_h, scale_w keeping ratio."""
    short_target = int(np.random.choice(short_choices))
    short = min(h, w)
    long_ = max(h, w)
    s = min(short_target / short, max_long / long_)
    new_h = int(round(h * s))
    new_w = int(round(w * s))
    return new_h, new_w, s, s


def _center_crop_or_pad(img: np.ndarray, out_h: int, out_w: int, pad_val=(0, 0, 0)) -> Tuple[np.ndarray, int, int]:
    """Center crop to (out_h,out_w). If smaller, pad to center."""
    h, w = img.shape[:2]
    y0 = max(0, (h - out_h) // 2)
    x0 = max(0, (w - out_w) // 2)
    y1 = min(h, y0 + out_h)
    x1 = min(w, x0 + out_w)
    cropped = img[y0:y1, x0:x1]
    pad_h = max(0, out_h - cropped.shape[0])
    pad_w = max(0, out_w - cropped.shape[1])
    if pad_h > 0 or pad_w > 0:
        cropped = mmcv.impad(cropped, shape=(out_h, out_w), pad_val=pad_val)
    return cropped, y0, x0


def _resize_mask_nearest(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    # mask: HxW uint8 with {0..6,255}
    return cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


@TRANSFORMS.register_module()
class SeqResizeFlipPad:
    """Apply same resize->center crop->pad to every frame in a clip.
    - No flip.
    - Resize short side in {720,864,960,1080}, long side ≤ 1600.
    - Center crop to (896,1600), then pad if needed.
    - Transform last-frame bboxes and seg accordingly.
    """

    def __init__(self,
                 short_edge_choices=(720, 864, 960, 1080),
                 max_long_edge=1600,
                 crop_size=(896, 1600),
                 pad_val=(0, 0, 0)):
        self.short_edge_choices = tuple(short_edge_choices)
        self.max_long_edge = int(max_long_edge)
        self.crop_h, self.crop_w = int(crop_size[0]), int(crop_size[1])
        self.pad_val = pad_val

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        imgs: List[np.ndarray] = results['imgs']
        ori_h, ori_w = imgs[-1].shape[:2]
        new_h, new_w, sh, sw = _compute_resize(ori_h, ori_w, self.short_edge_choices, self.max_long_edge)

        # resize all frames
        resized = [mmcv.imresize(img, (new_w, new_h)) for img in imgs]

        # center crop/pad all frames
        cropped = []
        for img in resized:
            out, y0, x0 = _center_crop_or_pad(img, self.crop_h, self.crop_w, pad_val=self.pad_val)
            cropped.append(out)
        results['imgs'] = cropped

        # transform last-frame bboxes and seg
        # bbox: xyxy -> scale, then shift by crop offset, then clip
        if 'instances' in results and len(results['instances']) > 0:
            # recompute offsets using last frame resized map
            _, y0_l, x0_l = _center_crop_or_pad(resized[-1], self.crop_h, self.crop_w, pad_val=self.pad_val)
            inst = results['instances']
            bbs, labs, acts, dirs, ign = [], [], [], [], []
            for it in inst:
                x1, y1, x2, y2 = it['bbox']
                x1 = x1 * sw - x0_l
                x2 = x2 * sw - x0_l
                y1 = y1 * sh - y0_l
                y2 = y2 * sh - y0_l
                # clip
                x1 = np.clip(x1, 0, self.crop_w - 1)
                x2 = np.clip(x2, 0, self.crop_w - 1)
                y1 = np.clip(y1, 0, self.crop_h - 1)
                y2 = np.clip(y2, 0, self.crop_h - 1)
                if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                    continue
                bbs.append([x1, y1, x2, y2])
                labs.append(int(it['bbox_label']))
                acts.append(int(it.get('action', 0)))
                dirs.append(int(it.get('dir', 0)))
                ign.append(int(it.get('ignore_flag', 0)))
            results['instances'] = [
                dict(bbox=b, bbox_label=l, action=a, dir=d, ignore_flag=g)
                for b, l, a, d, g in zip(bbs, labs, acts, dirs, ign)
            ]

        if results.get('gt_sem_seg', None) is not None:
            seg = results['gt_sem_seg'].astype(np.uint8)
            seg = _resize_mask_nearest(seg, (new_h, new_w))
            seg, y0_l, x0_l = _center_crop_or_pad(seg, self.crop_h, self.crop_w, pad_val=255)
            results['gt_sem_seg'] = seg

        if 'ori_shape' not in results or results['ori_shape'] is None:
            results['ori_shape'] = (ori_h, ori_w)
        results['scale_factor'] = np.array([sw, sh, sw, sh], dtype=np.float32)
        results['img_shape'] = (self.crop_h, self.crop_w)
        results['pad_shape'] = (self.crop_h, self.crop_w)
        results['flip'] = False
        results['flip_direction'] = None

        # === 对末帧的 GT 做几何变换（只处理 results['gt_*']） ===

        bxs = results.get('gt_bboxes', None)
        lbs = results.get('gt_bboxes_labels', None)
        acts = results.get('gt_actions', None)
        dirs = results.get('gt_dirs', None)

        if bxs is not None:
            bxs = np.asarray(bxs, dtype=np.float32).copy()
            # 1) 缩放：按你上面算好的 scale_factor（sw,sh）
            bxs[:, 0] *= sw; bxs[:, 2] *= sw
            bxs[:, 1] *= sh; bxs[:, 3] *= sh
            # 2) 平移：中心裁剪后把坐标移到裁剪坐标系
            bxs[:, [0, 2]] -= float(x0)
            bxs[:, [1, 3]] -= float(y0)
            # 3) 裁边到目标尺寸
            bxs[:, 0::2] = np.clip(bxs[:, 0::2], 0, self.crop_w)
            bxs[:, 1::2] = np.clip(bxs[:, 1::2], 0, self.crop_h)
            # 4) 去退化
            w = bxs[:, 2] - bxs[:, 0]; h = bxs[:, 3] - bxs[:, 1]
            keep = (w > 0) & (h > 0)
            if keep.sum() != keep.size:
                bxs  = bxs[keep]
                if lbs is not None:  lbs  = np.asarray(lbs,  np.int64)[keep]
                if acts is not None: acts = np.asarray(acts, np.int64)[keep]
                if dirs is not None: dirs = np.asarray(dirs, np.int64)[keep]

            # 5) 写回（即使为空也写，Pack 会兜底）
            results['gt_bboxes'] = bxs
            results['gt_bboxes_labels'] = np.asarray(lbs,  np.int64) if lbs is not None else np.zeros((0,), np.int64)
            results['gt_actions']       = np.asarray(acts, np.int64) if acts is not None else np.zeros((0,), np.int64)
            results['gt_dirs']          = np.asarray(dirs, np.int64) if dirs is not None else np.zeros((0,), np.int64)

        # 更新 meta（scale_factor/img_shape/pad_shape）
        results['scale_factor'] = np.array([sw, sh, sw, sh], dtype=np.float32)
        results['img_shape'] = (self.crop_h, self.crop_w)
        results['pad_shape'] = (self.crop_h, self.crop_w)
        results['crop_offset'] = (int(y0) if y0 is not None else 0,
                           int(x0) if x0 is not None else 0)

        # keep original (H,W) if missing
        if 'ori_shape' not in results or results['ori_shape'] is None:
            results['ori_shape'] = (int(h0), int(w0))  # 用你读入原图的变量替代 h0,w0

        # ===== OPTIONAL: 一次性打印，避免刷屏 =====
        if np.random.rand() < 0.002:
            print('[CHK-AUG] sf=', results['scale_factor'],
                'offset=', results['crop_offset'],
                'img/pad=', results['img_shape'])


        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(short_edge_choices={self.short_edge_choices}, '
                f'max_long_edge={self.max_long_edge}, crop_size=({self.crop_h},{self.crop_w}))')
