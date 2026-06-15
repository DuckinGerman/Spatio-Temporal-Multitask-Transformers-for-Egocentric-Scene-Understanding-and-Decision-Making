# projects/vasco/datasets/vasco_clip_dataset.py
# mmdetection-3.3.0 — Clip dataset: last frame has thing+bbox(+action/dir) AND stuff semantic seg
import os
import re
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils

from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset

# -----------------------------
# Label conventions (VASCO)
# -----------------------------
# Action is stored in the dataset as the ORIGINAL 4-way ids plus a fallback:
#   dynamic: {0, 3}
#   static:  {1, 2}
#   noaction / invalid / missing: 4
# Note: the training head (ActionDirHead) will later collapse these raw action ids
# into a 2-way label (static vs dynamic) for the action classification loss.
#
# Direction is stored as:
#   valid: {0, 1, 2, 3, 4}
#   nodirection / invalid / missing: 5
#
# This dataset intentionally avoids emitting -1 for action/dir to prevent
# downstream "ignore" behavior and to keep the data pipeline robust.


# -----------------------------
# Helper functions
# -----------------------------
def _xywh2xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def _instances_to_gt(instances):
    """Convert last-frame `instances` (from parse_data_info) into GT arrays.

    Returns:
        gt_bboxes: (N, 4) float32 in xyxy
        gt_labels: (N,) int64 (thing class label 0..27)
        gt_actions: (N,) int64 raw action ids in {0,1,2,3,4}
        gt_dirs: (N,) int64 direction ids in {0..5}
    """
    import numpy as np
    if not instances:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,),   np.int64),
                np.zeros((0,),   np.int64),
                np.zeros((0,),   np.int64))
    bxs, lbs, acts, dirs = [], [], [], []
    for it in instances:
        x1, y1, x2, y2 = it['bbox']
        if x2 <= x1 or y2 <= y1:
            continue
        bxs.append([x1, y1, x2, y2])
        lbs.append(int(it['bbox_label']))

        # action ids in dataset are the ORIGINAL 4-way ids:
        # dynamic: {0,3}, static: {1,2}. noaction/unknown: 4
        a_raw = int(it.get('action', -1))
        if a_raw in STATIC_ACTION_IDS or a_raw in DYNAMIC_ACTION_IDS:
            a_id = a_raw
        else:
            a_id = 4  # noaction / invalid
        acts.append(int(a_id))

        # direction ids in dataset: keep {0..4}, nodirection/invalid -> 5
        d_raw = int(it.get('dir', -1))
        if 0 <= d_raw <= 4:
            d_id = d_raw
        else:
            d_id = 5
        dirs.append(int(d_id))
    if not bxs:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,),   np.int64),
                np.zeros((0,),   np.int64),
                np.zeros((0,),   np.int64))
    return (np.asarray(bxs,  np.float32),
            np.asarray(lbs,  np.int64),
            np.asarray(acts, np.int64),
            np.asarray(dirs, np.int64))


def _load_last_frame_gt(coco, img_id, cat2label=None):
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    bxs, lbs, acts, dirs = [], [], [], []
    for a in anns:
        cid = a['category_id']
        if cat2label is not None:
            if cid not in cat2label:
                continue
            lab = int(cat2label[cid])          # 0..27
        else:
            lab = int(cid)                     # 你已确保 0..27 即 thing
        # 仅保留 thing 类
        if lab < 0 or lab > 27:
            continue
        x1, y1, x2, y2 = _xywh2xyxy(a['bbox'])
        if x2 <= x1 or y2 <= y1:
            continue
        bxs.append([x1, y1, x2, y2])
        lbs.append(lab)
        ar = int(a.get('action', -1))
        if ar in STATIC_ACTION_IDS or ar in DYNAMIC_ACTION_IDS:
            acts.append(ar)
        else:
            acts.append(4)  # noaction/invalid

        dr = int(a.get('direction', -1))
        if 0 <= dr <= 4:
            dirs.append(dr)
        else:
            dirs.append(5)  # nodirection/invalid
    import numpy as np
    if len(bxs) == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,),   np.int64),
                np.zeros((0,),   np.int64),
                np.zeros((0,),   np.int64))
    return (np.asarray(bxs,  np.float32),
            np.asarray(lbs,  np.int64),
            np.asarray(acts, np.int64),
            np.asarray(dirs, np.int64))

def _parse_index_from_path(p: str) -> int:
    stem = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r'(\d+)$', stem)
    return int(m.group(1)) if m else -1


def _join_img_path(data_root: str, data_prefix, file_name: str) -> str:
    if isinstance(data_prefix, dict):
        pref = data_prefix.get('img', '')
    else:
        pref = data_prefix or ''
    parts = [x for x in [data_root, pref, file_name] if x]
    return os.path.join(*parts)


def _sample_clip_indices(last_pos: int, video_len: int, T: int, lookback: int) -> List[int]:
    start = max(0, last_pos - lookback)
    if T <= 1:
        return [last_pos]
    lin = np.linspace(start, last_pos, num=T, endpoint=True)
    idxs = np.rint(lin).astype(int).tolist()
    out = []
    for v in idxs:
        if not out or v > out[-1]:
            out.append(v)
    while len(out) < T:
        out = [out[0]] + out
    out[-1] = last_pos
    return out[-T:]


THING_CLASSES = (
    'bicycle', 'bus', 'bus_stop_sign', 'car', 'construction_barriers',
    'danger_children_sign', 'green_light_p', 'green_light_v', 'handrail',
    'motorcycle', 'orange_light_p', 'orange_light_v', 'pedestrian_crossing_sign',
    'pedestrian_path_sign', 'pedestrian_prohibited_sign',
    'pedestrians_must_cross_road_left_sign', 'pedestrians_must_cross_road_right_sign',
    'person', 'pole', 'red_light_p', 'red_light_v', 'scooter',
    'traffic_calming_zone_sign', 'trash_bin', 'tree', 'truck', 'van', 'warning_sign'
)
STUFF_CLASSES = (
    'bike_lane', 'crosswalk', 'sidewalk',
    'stairs', 'traffic_island', 'zebra'
)
METAINFO = dict(
    classes=THING_CLASSES,
    thing_classes=THING_CLASSES,
    stuff_classes=STUFF_CLASSES,
    stuff_num_classes=len(STUFF_CLASSES)
)

STATIC_ACTION_IDS = {1, 2}    # static action ids in the original annotation space
DYNAMIC_ACTION_IDS = {0, 3}   # dynamic action ids in the original annotation space

# ===== Ego choice / stopgo =====
# ego_choice is INPUT, stopgo is GT (0=stop, 1=go)
# Keep mapping fixed for train/infer consistency.
EGO_CHOICES = ('straight', 'left', 'right', 'left-left', 'right-right')
EGO_NAME_TO_ID = {n: i for i, n in enumerate(EGO_CHOICES)}
DEFAULT_EGO_ID = EGO_NAME_TO_ID['straight']  # unknown fallback


@DATASETS.register_module()
class VascoClipDataset(CocoDataset):
    """Clip dataset for VASCO.

    - Build clips of length T (temporal_window). All T frames are loaded as input.
    - Only the LAST frame provides supervision (bboxes/labels/action/dir and semantic stuff).
    - By default we keep the historical constraint that the last frame must contain
      both thing instances and stuff segmentation (controlled by ensure_last_has_thing_and_stuff).

    Action/Direction note:
      This dataset emits raw action ids (0/1/2/3/4) and direction ids (0..5).
      The model head can apply additional mappings (e.g., collapsing action to 2-way).
    """

    METAINFO = METAINFO

    def __init__(self,
                 temporal_window: int = 5,
                 temporal_span: int = 20,
                 ensure_last_has_thing_and_stuff: bool = True,
                 **kwargs) -> None:
        self.temporal_window = int(temporal_window)
        self.temporal_span = int(temporal_span)
        self.ensure_last = bool(ensure_last_has_thing_and_stuff)
        super().__init__(**kwargs)

        

    # ---------- NO parent call here ----------
    def parse_data_info(self, raw_data_info: Dict[str, Any]) -> Dict[str, Any]:
        """raw_data_info keys from CocoDataset.load_data_list():
           {'img_id', 'img_info', 'ann_info'}"""

        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info.get('raw_ann_info', [])

        img_id = img_info.get('id', -1)
        file_name = img_info.get('file_name', '')
        height = int(img_info.get('height', 0))
        width = int(img_info.get('width', 0))
        img_path = _join_img_path(self.data_root, self.data_prefix, file_name)

        # ===== ego choice (input) + stopgo (gt) from images[] =====
        ego_raw = img_info.get('ego_choice', '')
        if isinstance(ego_raw, str):
            ego_raw = ego_raw.strip().lower()
        else:
            ego_raw = ''

        # NOTE: Deploy-time Google navigation does NOT provide a 'stop' command.
        # Any legacy 'stop' stored in ego_choice is treated as 'straight' intent.
        if ego_raw == 'stop':
            ego_raw = 'straight'

        ego_choice_id = EGO_NAME_TO_ID.get(ego_raw, DEFAULT_EGO_ID)

        sg_raw = img_info.get('stopgo', '')
        if isinstance(sg_raw, str):
            sg_raw = sg_raw.strip().lower()
        else:
            sg_raw = ''
        gt_stopgo = 1 if sg_raw == 'go' else 0   # stop=0, go=1

        # thing instances with attributes; use self.cat2label (only thing classes are mapped)
        instances = []
        has_thing = False
        cat2label = getattr(self, 'cat2label', {})
        for an in ann_info:
            cid = an.get('category_id', -1)
            if cid in cat2label:
                bbox = an.get('bbox', None)
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                x, y, w, h = bbox
                if w <= 1 or h <= 1:
                    continue
                attrs = an.get('attributes', {}) or {}
                action = int(attrs.get('action_id', attrs.get('action', -1)))
                # map noaction/invalid -> 4
                if action not in STATIC_ACTION_IDS and action not in DYNAMIC_ACTION_IDS:
                    action = 4

                direction = int(attrs.get('direction_id', attrs.get('dir', -1)))
                # map nodirection/invalid -> 5
                if not (0 <= direction <= 4):
                    direction = 5

                instances.append(dict(
                    bbox=[x, y, x + w, y + h],
                    bbox_label=cat2label[cid],
                    ignore_flag=0,
                    action=action,
                    dir=direction,
                ))
                has_thing = True

        # stuff semantic map: map stuff category_id 28..33 -> 0..5 (6 foreground classes)
        # pixels without any stuff annotation are set to BG_ID=6 (explicit pseudo-background class)
        # ignore pixels (padding/invalid) remain 255 (ignore_index)
        sem_label = None  # np.uint8 [H,W] or None
        BG_ID = 6
        for an in ann_info:
            cid = an.get('category_id', -1)
            if 28 <= int(cid) <= 33:
                seg = an.get('segmentation', None)
                if not seg:
                    continue
                try:
                    if isinstance(seg, list):  # polygons
                        rles = mask_utils.frPyObjects(seg, height, width)
                        rle = mask_utils.merge(rles)
                        m = mask_utils.decode(rle)
                    elif isinstance(seg, dict):  # RLE
                        m = mask_utils.decode(seg)
                    else:
                        m = None
                    if m is None:
                        continue
                    if m.ndim == 3:
                        m = (m.sum(axis=2) > 0).astype(np.uint8)
                    else:
                        m = (m > 0).astype(np.uint8)
                    if m.sum() == 0:
                        continue
                    if sem_label is None:
                        sem_label = np.full((height, width), BG_ID, dtype=np.uint8)
                    cls = int(cid) - 28  # [0..5]
                    sem_label[m.astype(bool)] = cls
                except Exception:
                    continue

        # video/frame ids
        video_id = img_info.get('video_id', os.path.basename(os.path.dirname(file_name)))
        frame_id = img_info.get('frame_id', _parse_index_from_path(file_name))
        try:
            frame_id = int(frame_id)
        except Exception:
            frame_id = _parse_index_from_path(file_name)

        data = dict(
            img_id=img_id,
            img_path=img_path,
            height=height,
            width=width,
            instances=instances,
            gt_sem_seg=sem_label,
            video_id=str(video_id),
            frame_id=frame_id,
            # ego
            ego_choice_id=int(ego_choice_id),
            gt_stopgo=int(gt_stopgo),
        )
        data['__has_thing__'] = has_thing
        # "has_stuff" means the last frame has at least one FOREGROUND stuff pixel (0..5).
        # Note: sem_label contains BG_ID=6 for all non-stuff pixels once it is created.
        data['__has_stuff__'] = (sem_label is not None) and np.any((sem_label >= 0) & (sem_label < BG_ID))
        return data



    # ---------- build clips by grouping per video ----------
    def load_data_list(self) -> List[Dict[str, Any]]:
        # Let parent assemble raw_data_info per image, but it will call OUR parse_data_info above
        singles = super().load_data_list()

        by_vid = defaultdict(list)
        for di in singles:
            by_vid[di['video_id']].append(di)
        for vid in by_vid:
            by_vid[vid].sort(key=lambda d: d.get('frame_id', 0))

        T = self.temporal_window
        span = self.temporal_span
        clip_list: List[Dict[str, Any]] = []

        for vid, frames in by_vid.items():
            n = len(frames)
            for j in range(n):
                last = frames[j]
                has_thing = bool(last.get('__has_thing__', False) or len(last.get('instances', [])) > 0)
                gt = last.get('gt_sem_seg', None)
                # "has_stuff" means at least one FOREGROUND stuff pixel (0..5) exists in the last frame.
                # Background pixels are encoded as 6, and ignore pixels remain 255.
                has_stuff = (gt is not None) and np.any((gt >= 0) & (gt < 6))
                if self.ensure_last and not (has_thing and has_stuff):
                    continue

                idxs = _sample_clip_indices(j, n, T, span)
                ctx_frames = [frames[idx] for idx in idxs]

                clip = dict(last)
                clip['img_path_list'] = [f['img_path'] for f in ctx_frames]
                clip['frame_inds'] = [f.get('frame_id', -1) for f in ctx_frames]
                clip['clip_len'] = T
                clip['video_id'] = vid
                
                # Use ONLY last-frame instances to generate GT arrays (bbox/label/action/dir)
                gb, gl, ga, gd = _instances_to_gt(last.get('instances', []))
                clip['gt_bboxes'] = gb
                clip['gt_bboxes_labels'] = gl
                clip['gt_actions'] = ga
                clip['gt_dirs'] = gd

                # ego input + stopgo gt (use LAST frame only, consistent with your supervision rule)
                clip['ego_choice_id'] = int(last.get('ego_choice_id', DEFAULT_EGO_ID))
                clip['gt_stopgo'] = int(last.get('gt_stopgo', 0))

                # 末帧原始尺寸（给 CocoMetric）
                if 'ori_shape' not in clip or clip['ori_shape'] is None:
                    h0 = int(last.get('height', 0))
                    w0 = int(last.get('width', 0))
                    clip['ori_shape'] = (h0, w0) if h0 and w0 else last.get('img_shape', None)

                clip_list.append(clip)

                # Optional debug: set VASCO_DEBUG=1 to print a few sample clips
                if os.environ.get('VASCO_DEBUG', '0') == '1' and len(clip_list) <= 5:
                    uniq = sorted(set(ga.tolist())) if ga.size > 0 else []
                    ego_id = int(clip.get('ego_choice_id', DEFAULT_EGO_ID))
                    sg = int(clip.get('gt_stopgo', 0))
                    print(
                        f"[VASCO DEBUG] clip#{len(clip_list)} "
                        f"video={vid}, last_frame={last.get('frame_id', -1)}, "
                        f"ego_choice_id={ego_id}, gt_stopgo={sg}, "
                        f"gt_actions={ga[:10]} (unique={uniq})"
                    )
                
        print(f"[VASCO] Built {len(clip_list)} clips (last frame must have thing + stuff). T={T}, span={span}")
        assert len(clip_list) > 0, "No clips satisfy: last frame has BOTH thing and stuff."

        return clip_list
