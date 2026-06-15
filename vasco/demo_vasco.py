# -*- coding: utf-8 -*-
"""
视频滑窗推理+可视化（bbox+动作/方向+语义+后置Ego）
- 与训练/val 完全共享 dataset.pipeline 与 data_preprocessor
- 适配当前 two-stage/post-ego 结构：
  - 使用 --perception-only 可只运行前置 perception，不显示/调用后置 Ego
  1) 前置感知模型输出 detection / segmentation / action / direction
  2) 后置 ego head 基于前置预测 + seg summary + ego_choice 输出 stop/go
- action/direction:
  - action 对外仍为 2 类：static / dynamic
  - direction 对外仍为原始 6 类：front / right / left / near / away / nodirection
  - 当前模型内部可以是 state-conditioned 5+4 或 MoE，但 RoIHead.predict 会映射回上述外部输出
- seg: 当前为 6 类前景（0..5）+ 背景类 id=6；可视化按 logits 处理，并忽略背景类显示
- 输出 2 个视频：
  1) bbox 颜色=action，文本=object+action
  2) bbox 颜色=direction，文本=object+direction
- Ego:
  - 仅把 COCO images[] 里的 ego_choice 注入 metainfo 作为 ego 头输入之一
  - stop/go 仅显示模型预测，不从 COCO 读取/覆盖（训练集里的 stopgo 是 GT，不需要）
"""

import os
import cv2
import argparse
import numpy as np
import json
from collections import deque

import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.dataset import Compose
from mmengine.structures import PixelData
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample
from torchvision.ops import nms


# -------------------------
# Constants / Class Names
# -------------------------
# External deploy-time outputs. Keep these unchanged even if the internal AD head is 5+4/MoE.
ACTION_CLASSES = ["static", "dynamic"]
DIR_CLASSES = ["front", "right", "left", "near", "away", "nodirection"]
EGO_STOPGO_NAMES = ["stop", "go"]

# Deploy-time navigation intent (5-class, no 'stop')
EGO_CHOICE_NAMES = ["straight", "left", "right", "left-left", "right-right"]
EGO_CHOICE_TO_ID = {n: i for i, n in enumerate(EGO_CHOICE_NAMES)}

IGNORE_INDEX = 255
SEG_BG_ID = 6  # background id in training/inference label space (not visualized / not evaluated)

# seg 类别名称：6类（0..5 为前景），背景类 id=6（不在这里显示，不参与评估）
SEG_CLASSES = ["bike_lane", "crosswalk", "sidewalk", "stairs", "traffic_island", "zebra"]

# det 类别名称：28类
THING_CLASSES = [
    "bicycle",                         # 0
    "bus",                             # 1
    "bus_stop_sign",                   # 2
    "car",                             # 3
    "construction_barriers",           # 4
    "danger_children_sign",            # 5
    "green_light_p",                   # 6
    "green_light_v",                   # 7
    "handrail",                        # 8
    "motorcycle",                      # 9
    "orange_light_p",                  # 10
    "orange_light_v",                  # 11
    "pedestrian_crossing_sign",        # 12
    "pedestrian_path_sign",            # 13
    "pedestrian_prohibited_sign",      # 14
    "pedestrians_must_cross_road_left_sign",   # 15
    "pedestrians_must_cross_road_right_sign",  # 16
    "person",                          # 17
    "pole",                            # 18
    "red_light_p",                     # 19
    "red_light_v",                     # 20
    "scooter",                         # 21
    "traffic_calming_zone_sign",       # 22
    "trash_bin",                       # 23
    "tree",                            # 24
    "truck",                           # 25
    "van",                             # 26
    "warning_sign",                    # 27
]


# -------------------------
# Palettes (BGR for cv2)
# -------------------------
# 语义分割 palette：6类（0..5）
PALETTE = np.array([
    [0,   0,   0],    # 0
    [0, 114, 189],    # 1
    [217, 83,  25],   # 2
    [237, 177, 32],   # 3
    [126, 47,  142],  # 4
    [119, 172, 48],   # 5
], dtype=np.uint8)

# 检测 palette：28类
DET_PALETTE = np.array([
    [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0],
    [0, 255, 255], [255, 0, 255], [128, 0, 0], [0, 128, 0],
    [0, 0, 128], [128, 128, 0], [0, 128, 128], [128, 0, 128],
    [64, 0, 0], [0, 64, 0], [0, 0, 64], [64, 64, 0],
    [0, 64, 64], [64, 0, 64], [192, 0, 0], [0, 192, 0],
    [0, 0, 192], [192, 192, 0], [0, 192, 192], [192, 0, 192],
    [96, 96, 255], [96, 255, 96], [255, 96, 96], [200, 200, 200],
], dtype=np.uint8)

# action palette：2类
ACTION_PALETTE = np.array([
    [255, 140, 0],   # static
    [0, 165, 255],   # dynamic
], dtype=np.uint8)

# direction palette：6类
DIR_PALETTE = np.array([
    [255, 0, 0],     # front
    [0, 0, 255],     # right
    [0, 255, 0],     # left
    [255, 255, 0],   # near
    [255, 0, 255],   # away
    [0, 255, 255],   # nodirection
], dtype=np.uint8)


# -------------------------
# Helpers
# -------------------------
def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def load_coco_video_frames(ann_json: str, img_root: str, video_id: int = None, video_name: str = None):
    """Load single video's frames and ego_choice from COCO json.
    Only reads ego_choice / ego_choice_id from images[].

    Returns:
      frames: list[dict] with keys: path, frame_id, choice_str, choice_id
      vid_name: str
      vid_id: int
    """
    with open(ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    videos = coco.get("videos", None)
    images = coco.get("images", [])

    resolved_vid_id = None
    resolved_vid_name = None

    if video_id is not None:
        resolved_vid_id = int(video_id)
        if videos:
            for v in videos:
                if _safe_int(v.get("id")) == resolved_vid_id:
                    resolved_vid_name = v.get("name", v.get("file_name", None))
                    break
    elif video_name is not None:
        if videos:
            for v in videos:
                n = v.get("name", v.get("file_name", ""))
                if str(n) == str(video_name):
                    resolved_vid_id = _safe_int(v.get("id"))
                    resolved_vid_name = str(n)
                    break
        if resolved_vid_id is None:
            vid_try = _safe_int(video_name, default=None)
            if vid_try is not None:
                resolved_vid_id = vid_try

    if resolved_vid_id is None:
        vid_ids = [_safe_int(im.get("video_id")) for im in images if "video_id" in im]
        vid_ids = [v for v in vid_ids if v is not None]
        if not vid_ids:
            raise ValueError("COCO json has no video_id in images and no videos[]; cannot select video.")
        resolved_vid_id = sorted(set(vid_ids))[0]

    if resolved_vid_name is None:
        if videos:
            for v in videos:
                if _safe_int(v.get("id")) == resolved_vid_id:
                    resolved_vid_name = v.get("name", v.get("file_name", f"video_{resolved_vid_id}"))
                    break
        if resolved_vid_name is None:
            resolved_vid_name = f"video_{resolved_vid_id}"

    frames = []
    for im in images:
        if _safe_int(im.get("video_id")) != resolved_vid_id:
            continue

        file_name = im.get("file_name") or im.get("path")
        if not file_name:
            continue
        full_path = os.path.join(img_root, file_name)

        fid = im.get("frame_id", im.get("frame_index", im.get("id")))
        fid = _safe_int(fid, default=None)

        choice_str = None
        if im.get("ego_choice", None) is not None:
            choice_str = str(im["ego_choice"]).strip().lower()

        choice_id = None
        if im.get("ego_choice_id", None) is not None:
            choice_id = _safe_int(im["ego_choice_id"], default=None)

        frames.append(dict(path=full_path, frame_id=fid, choice_str=choice_str, choice_id=choice_id))

    if not frames:
        raise ValueError(f"No images found for video_id={resolved_vid_id} in COCO json.")

    if all(f["frame_id"] is not None for f in frames):
        frames.sort(key=lambda x: x["frame_id"])
    else:
        frames.sort(key=lambda x: x["path"])

    return frames, resolved_vid_name, resolved_vid_id


class ChoiceState:
    """Persist last known ego_choice and update only when a new one is observed."""
    def __init__(self, default: str = "straight"):
        self.choice_str = default
        self.choice_id = int(EGO_CHOICE_TO_ID.get(default, 0))

    def update_from_sparse(self, choice_str=None, choice_id=None):
        if choice_str is not None:
            c = str(choice_str).strip().lower()
            if c in EGO_CHOICE_TO_ID:
                self.choice_str = c
                self.choice_id = int(EGO_CHOICE_TO_ID[c])
                return
        if choice_id is not None:
            cid = _safe_int(choice_id, default=None)
            if cid is not None and 0 <= cid < len(EGO_CHOICE_NAMES):
                self.choice_id = int(cid)
                self.choice_str = EGO_CHOICE_NAMES[int(cid)]

    def update_fallback_by_second(self, sec: int) -> None:
        c = get_choice_by_second(sec)
        self.choice_str = c
        self.choice_id = int(EGO_CHOICE_TO_ID.get(c, 0))


def get_choice_by_second(sec: int) -> str:
    """ONLY for raw video mode when no COCO annotations are provided."""
    if 0 <= sec <= 6:
        return "straight"
    if 7 <= sec <= 9:
        return "right"
    if 10 <= sec <= 22:
        return "straight"
    if 23 <= sec <= 24:
        return "left"
    return "straight"


def draw_choice_banner(img_bgr, choice: str):
    """Draw CHOICE text at bottom-left."""
    if not choice:
        return img_bgr
    h, _ = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    text = f"CHOICE: {choice.upper()}"
    x = 20
    y = h - 20
    cv2.putText(img_bgr, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img_bgr


def draw_stopgo_banner(img_bgr, stopgo_id, score=None):
    """Draw STOP/GO text at bottom center. stop:red, go:green."""
    if stopgo_id is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3

    stopgo_id = int(stopgo_id)
    stopgo_id = 1 if stopgo_id == 1 else 0
    name = EGO_STOPGO_NAMES[stopgo_id].upper()
    color = (0, 255, 0) if stopgo_id == 1 else (0, 0, 255)

    text = f"{name} {score:.2f}" if score is not None else name
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, (w - tw) // 2)
    y = h - 20

    cv2.putText(img_bgr, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img_bgr


def draw_seg_legend(img_bgr,
                    class_names=SEG_CLASSES,
                    palette=PALETTE,
                    font_scale=0.6,
                    thickness=1):
    """Left-top segmentation legend."""
    x0, y0 = 8, 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_gap = 4

    for cid, name in enumerate(class_names):
        if cid >= len(palette):
            break
        color = tuple(int(c) for c in palette[cid])
        (tw, th), _ = cv2.getTextSize(name, font, font_scale, thickness)
        y = y0 + th
        cv2.putText(img_bgr, name, (x0, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y0 += th + line_gap
    return img_bgr


def draw_action_dir_legend(img_bgr,
                           action_names=ACTION_CLASSES,
                           dir_names=DIR_CLASSES,
                           action_palette=ACTION_PALETTE,
                           dir_palette=DIR_PALETTE,
                           font_scale=0.6,
                           thickness=1):
    """Right-top action/direction legend."""
    h, w = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 8
    line_gap = 4

    y0 = 8
    for aid, name in enumerate(action_names):
        if aid >= len(action_palette):
            break
        color = tuple(int(c) for c in action_palette[aid])
        text = f"A: {name}"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = w - margin - tw
        y = y0 + th
        cv2.putText(img_bgr, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y0 += th + line_gap

    y0 += 4
    for did, name in enumerate(dir_names):
        if did >= len(dir_palette):
            break
        color = tuple(int(c) for c in dir_palette[did])
        text = f"D: {name}"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = w - margin - tw
        y = y0 + th
        cv2.putText(img_bgr, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y0 += th + line_gap

    return img_bgr


def draw_dets_obj_action(img_bgr, bboxes, labels, scores, actions=None, score_thr=0.30):
    """Video-A: bbox color follows ACTION; text shows 'object action' only."""
    if bboxes is None or len(bboxes) == 0:
        return img_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    _, w_img = img_bgr.shape[:2]

    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        s = float(scores[i]) if scores is not None else 1.0
        if s < score_thr:
            continue

        cls_id = int(labels[i]) if labels is not None else -1
        act_id = int(actions[i]) if (actions is not None and actions[i] is not None) else -1

        if 0 <= act_id < len(ACTION_PALETTE):
            box_color = tuple(int(c) for c in ACTION_PALETTE[act_id])
        else:
            box_color = (0, 255, 0)

        cls_name = THING_CLASSES[cls_id] if 0 <= cls_id < len(THING_CLASSES) else f"c{cls_id}"
        act_name = ACTION_CLASSES[act_id] if 0 <= act_id < len(ACTION_CLASSES) else f"a{act_id}"
        text = f"{cls_name} {act_name}"

        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = int(x1)
        y = int(y1) - 6
        if y - th - 4 < 0:
            y = int(y1) + th + 6
        if x + tw + 6 > w_img:
            x = max(0, w_img - tw - 6)

        cv2.putText(img_bgr, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img_bgr, text, (x, y), font, font_scale, box_color, thickness, cv2.LINE_AA)

    return img_bgr


def draw_dets_obj_dir(img_bgr, bboxes, labels, scores, dirs=None, score_thr=0.30):
    """Video-B: bbox color follows DIRECTION; text shows 'object direction' only."""
    if bboxes is None or len(bboxes) == 0:
        return img_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    _, w_img = img_bgr.shape[:2]

    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        s = float(scores[i]) if scores is not None else 1.0
        if s < score_thr:
            continue

        cls_id = int(labels[i]) if labels is not None else -1
        dir_id = int(dirs[i]) if (dirs is not None and dirs[i] is not None) else -1

        if 0 <= dir_id < len(DIR_PALETTE):
            box_color = tuple(int(c) for c in DIR_PALETTE[dir_id])
        else:
            box_color = (0, 255, 0)

        cls_name = THING_CLASSES[cls_id] if 0 <= cls_id < len(THING_CLASSES) else f"c{cls_id}"
        dir_name = DIR_CLASSES[dir_id] if 0 <= dir_id < len(DIR_CLASSES) else f"d{dir_id}"
        text = f"{cls_name} {dir_name}"

        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = int(x1)
        y = int(y1) - 6
        if y - th - 4 < 0:
            y = int(y1) + th + 6
        if x + tw + 6 > w_img:
            x = max(0, w_img - tw - 6)

        cv2.putText(img_bgr, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img_bgr, text, (x, y), font, font_scale, box_color, thickness, cv2.LINE_AA)

    return img_bgr


def _compute_iou_matrix(boxes1, boxes2):
    """boxes1: (M,4), boxes2: (N,4) -> IoU matrix (M,N)"""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xI1 = np.maximum(x11, x21.T)
    yI1 = np.maximum(y11, y21.T)
    xI2 = np.minimum(x12, x22.T)
    yI2 = np.minimum(y12, y22.T)

    inter_w = np.clip(xI2 - xI1, a_min=0, a_max=None)
    inter_h = np.clip(yI2 - yI1, a_min=0, a_max=None)
    inter = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2.T - inter
    union = np.clip(union, a_min=1e-6, a_max=None)

    return (inter / union).astype(np.float32)


def temporal_smooth_boxes(
    prev_state,
    bboxes, labels, scores, actions, dirs,
    iou_thr=0.3,
    alpha=0.5,
    min_age=1,
    lock_thr=0.8,
    post_nms_iou=0.4,
):
    """Frame-to-frame smoothing + second NMS."""
    if bboxes is None or len(bboxes) == 0:
        return None, None, None, None, None, prev_state

    bboxes = np.asarray(bboxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32) if scores is not None else None
    labels = np.asarray(labels, dtype=np.int32) if labels is not None else None
    actions = np.asarray(actions, dtype=np.int32) if actions is not None else None
    dirs = np.asarray(dirs, dtype=np.int32) if dirs is not None else None

    N = bboxes.shape[0]
    new_b = np.zeros_like(bboxes)
    new_s = scores.copy() if scores is not None else None
    new_l = labels.copy() if labels is not None else None
    new_a = actions.copy() if actions is not None else None
    new_d = dirs.copy() if dirs is not None else None
    new_age = np.zeros((N,), dtype=np.int32)

    if prev_state is None or prev_state.get("bboxes", None) is None or len(prev_state["bboxes"]) == 0:
        new_b[:] = bboxes
        new_age[:] = 1
    else:
        prev_b = prev_state["bboxes"]
        prev_l = prev_state["labels"]
        prev_s = prev_state["scores"]
        prev_age = prev_state["ages"]
        M = prev_b.shape[0]

        iou_mat = _compute_iou_matrix(prev_b, bboxes)

        if labels is not None and prev_l is not None:
            for i in range(M):
                for j in range(N):
                    if prev_l[i] != labels[j]:
                        iou_mat[i, j] = -1.0

        matches_prev = -np.ones((N,), dtype=np.int32)
        if M > 0 and N > 0:
            flat_idx = np.argsort(-iou_mat.reshape(-1))
            used_prev = set()
            used_cur = set()
            for idx in flat_idx:
                i = idx // N
                j = idx % N
                if iou_mat[i, j] < iou_thr:
                    break
                if (i in used_prev) or (j in used_cur):
                    continue
                matches_prev[j] = i
                used_prev.add(i)
                used_cur.add(j)

        for j in range(N):
            i = matches_prev[j]
            if i >= 0:
                iou_ij = iou_mat[i, j]
                if iou_ij >= lock_thr:
                    new_b[j] = prev_b[i]
                else:
                    new_b[j] = alpha * bboxes[j] + (1.0 - alpha) * prev_b[i]

                if scores is not None and prev_s is not None:
                    new_s[j] = alpha * scores[j] + (1.0 - alpha) * prev_s[i]
                new_age[j] = prev_age[i] + 1
            else:
                new_b[j] = bboxes[j]
                new_age[j] = 1

    if post_nms_iou is not None and new_s is not None and len(new_b) > 0:
        tb = torch.from_numpy(new_b)
        ts = torch.from_numpy(new_s)
        keep2 = nms(tb, ts, post_nms_iou).cpu().numpy()

        new_b = new_b[keep2]
        new_s = new_s[keep2]
        new_l = new_l[keep2] if new_l is not None else None
        new_a = new_a[keep2] if new_a is not None else None
        new_d = new_d[keep2] if new_d is not None else None
        new_age = new_age[keep2]

    new_state = dict(
        bboxes=new_b.copy(),
        labels=new_l.copy() if new_l is not None else None,
        scores=new_s.copy() if new_s is not None else None,
        actions=new_a.copy() if new_a is not None else None,
        dirs=new_d.copy() if new_d is not None else None,
        ages=new_age,
    )

    if min_age > 1:
        keep = new_age >= int(min_age)
        if not keep.any():
            return None, None, None, None, None, new_state
        draw_b = new_b[keep]
        draw_l = new_l[keep] if new_l is not None else None
        draw_s = new_s[keep] if new_s is not None else None
        draw_a = new_a[keep] if new_a is not None else None
        draw_d = new_d[keep] if new_d is not None else None
    else:
        draw_b, draw_l, draw_s, draw_a, draw_d = new_b, new_l, new_s, new_a, new_d

    return draw_b, draw_l, draw_s, draw_a, draw_d, new_state


def warp_seg_to_ori(seg_hw, meta):
    """Warp seg map from resized/crop space back to original image space."""
    ori_shape = meta.get("ori_shape", (None, None))
    if isinstance(ori_shape, (list, tuple, np.ndarray)):
        H0 = int(np.array(ori_shape).ravel()[0])
        W0 = int(np.array(ori_shape).ravel()[1])
    else:
        H0, W0 = ori_shape
    assert H0 and W0, "meta.ori_shape missing"

    img_shape = meta.get("img_shape", (None, None))
    if isinstance(img_shape, (list, tuple, np.ndarray)):
        arr = np.array(img_shape).ravel()
        ih = int(arr[0]); iw = int(arr[1])
    else:
        ih, iw = img_shape
    assert ih and iw, "meta.img_shape missing"

    sf = meta.get("scale_factor", 1.0)
    if isinstance(sf, (list, tuple, np.ndarray)):
        arr = np.array(sf, dtype=np.float32).ravel()
        if arr.size == 1:
            sx = sy = float(arr[0])
        else:
            sx = float(arr[0]); sy = float(arr[1])
    else:
        sx = sy = float(sf)

    co = meta.get("crop_offset", (0, 0))
    if isinstance(co, (list, tuple, np.ndarray)):
        arr = np.array(co, dtype=np.float32).ravel()
        y0 = float(arr[0]) if arr.size >= 1 else 0.0
        x0 = float(arr[1]) if arr.size >= 2 else 0.0
    else:
        y0, x0 = 0.0, 0.0

    if seg_hw.shape[:2] != (ih, iw):
        seg_hw = cv2.resize(seg_hw, (iw, ih), interpolation=cv2.INTER_NEAREST)

    M = np.array([[1.0 / sx, 0.0, x0 / sx],
                  [0.0, 1.0 / sy, y0 / sy]], dtype=np.float32)

    border_value = IGNORE_INDEX if np.issubdtype(seg_hw.dtype, np.integer) else 0.0

    seg_ori = cv2.warpAffine(
        seg_hw, M, (W0, H0),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return seg_ori


def extract_stopgo(sample: DetDataSample):
    """Return (stopgo_id, stopgo_score) id: 0=stop, 1=go. Prefer model outputs."""
    if not isinstance(sample, DetDataSample):
        return None, None

    # New post-ego path: RoIHead writes both scalar prediction and logits.
    # Prefer logits first so that the displayed confidence score is available.
    for key in ["pred_stopgo_logits", "pred_ego_stopgo_logits", "pred_stop_go_logits"]:
        if hasattr(sample, key):
            v = getattr(sample, key)
            if v is None:
                continue
            try:
                if hasattr(v, "data"):
                    v = v.data
                if isinstance(v, np.ndarray):
                    t = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    t = v
                else:
                    t = torch.as_tensor(v)
                t = t.detach().float().cpu().view(-1)
                if t.numel() >= 2:
                    p = torch.softmax(t[:2], dim=0)
                    sid = int(torch.argmax(p).item())
                    score = float(p[sid].item())
                    return sid, score
            except Exception:
                pass

    for key in ["pred_stopgo", "pred_ego_stopgo", "pred_stop_go"]:
        if hasattr(sample, key):
            v = getattr(sample, key)
            if v is None:
                continue
            try:
                if hasattr(v, "data"):
                    v = v.data
                if isinstance(v, np.ndarray):
                    t = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    t = v
                else:
                    t = torch.as_tensor(v)
                t = t.detach().float().cpu()
                if t.numel() == 1:
                    return int(t.item()), None
                if t.ndim > 1:
                    t = t.view(-1)
                p = torch.softmax(t, dim=0) if t.numel() == 2 else t
                sid = int(torch.argmax(p).item())
                score = float(p[sid].item()) if p.numel() >= 2 else None
                return sid, score
            except Exception:
                pass

    # fallback: metainfo (only if your head writes it)
    for k in ["pred_stopgo", "ego_stopgo"]:
        if k in sample.metainfo:
            v = sample.metainfo.get(k)
            if v is None:
                continue
            if isinstance(v, str):
                vv = v.strip().lower()
                if vv == "go":
                    return 1, None
                if vv == "stop":
                    return 0, None
            try:
                return int(v), None
            except Exception:
                pass

    if hasattr(sample, "pred_ego"):
        v = getattr(sample, "pred_ego")
        if isinstance(v, dict):
            vv = v.get("stopgo", v.get("pred_stopgo", None))
            if vv is not None:
                if isinstance(vv, str):
                    s = vv.strip().lower()
                    return (1, None) if s == "go" else (0, None)
                try:
                    return int(vv), None
                except Exception:
                    pass

    return None, None


def extract_pred(sample: DetDataSample):
    """Return bboxes, labels, scores, actions, dirs, seg_label(or None), seg_logits(or None)."""
    seg = None
    seg_logits = None
    if not isinstance(sample, DetDataSample):
        return None, None, None, None, None, seg, seg_logits

    pss = getattr(sample, "pred_sem_seg", None)
    if isinstance(pss, PixelData):
        t = getattr(pss, "data", None)
        if isinstance(t, torch.Tensor):
            if t.ndim == 3 and t.size(0) == 1:
                t = t[0]
            if t.ndim == 2:
                seg = t.detach().cpu().to(torch.int64).numpy()
        elif isinstance(t, np.ndarray):
            if t.ndim == 3 and t.shape[0] == 1:
                t = t[0]
            if t.ndim == 2:
                seg = t.astype(np.int64)

        lg = getattr(pss, "logits", None)
        if lg is not None and isinstance(lg, torch.Tensor) and lg.ndim == 3 and lg.size(0) > 1:
            seg_logits = lg.detach().cpu()  # (C,H,W)

    if seg_logits is None:
        lg_np = sample.metainfo.get("seg_logits", None)
        if lg_np is not None:
            seg_logits = torch.from_numpy(lg_np)

    if seg_logits is None and isinstance(pss, PixelData):
        t = getattr(pss, "data", None)
        if isinstance(t, torch.Tensor) and t.ndim == 3 and t.size(0) > 1:
            seg_logits = t.detach().cpu()
            seg = seg_logits.argmax(dim=0).to(torch.int64).numpy()
        elif isinstance(t, np.ndarray) and t.ndim == 3 and t.shape[0] > 1:
            seg_logits = torch.from_numpy(t)
            seg = seg_logits.argmax(dim=0).to(torch.int64).numpy()

    inst = getattr(sample, "pred_instances", None)
    if inst is None or not hasattr(inst, "bboxes") or inst.bboxes.numel() == 0:
        return None, None, None, None, None, seg, seg_logits

    b = inst.bboxes.detach().cpu().numpy()
    s = inst.scores.detach().cpu().numpy() if hasattr(inst, "scores") else None
    l = inst.labels.detach().cpu().numpy() if hasattr(inst, "labels") else None

    a = None
    # Current RoIHead writes final action ids to `actions`.
    # Keep fallbacks for older checkpoints that may store logits/probs.
    if hasattr(inst, "actions") and inst.actions is not None:
        A = inst.actions
        a = (A.detach().cpu().numpy() if A.ndim == 1 else A.argmax(-1).detach().cpu().numpy())
    elif hasattr(inst, "action_logits") and inst.action_logits is not None:
        A = inst.action_logits
        a = A.argmax(-1).detach().cpu().numpy()
    elif hasattr(inst, "action_scores") and inst.action_scores is not None:
        A = inst.action_scores
        a = A.argmax(-1).detach().cpu().numpy()

    d = None
    # Current AD head may be internally state-conditioned 5+4, but RoIHead.predict
    # should expose final original 6-way direction ids through `dirs`.
    # Keep fallbacks for older checkpoints that may store logits/probs.
    if hasattr(inst, "dirs") and inst.dirs is not None:
        D = inst.dirs
        d = (D.detach().cpu().numpy() if D.ndim == 1 else D.argmax(-1).detach().cpu().numpy())
    elif hasattr(inst, "dir_logits") and inst.dir_logits is not None:
        D = inst.dir_logits
        d = D.argmax(-1).detach().cpu().numpy()
    elif hasattr(inst, "dir_scores") and inst.dir_scores is not None:
        D = inst.dir_scores
        d = D.argmax(-1).detach().cpu().numpy()

    return b, l, s, a, d, seg, seg_logits


def build_pipeline(cfg):
    if "test_dataloader" in cfg and "dataset" in cfg.test_dataloader:
        pipe_cfg = cfg.test_dataloader.dataset.pipeline
    else:
        pipe_cfg = cfg.val_dataloader.dataset.pipeline
    return Compose(pipe_cfg)


def make_sample(clip_paths, video_id, frame_id):
    return dict(img_path_list=clip_paths, video_id=video_id, frame_id=frame_id)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)

    ap.add_argument("--input", default=None, help="Video path (used when --ann-json is not set)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--score-thr", type=float, default=0.50)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tempdir", default="./_frames_tmp")

    # seg logits post-process (only when seg_logits exists)
    ap.add_argument("--seg-thr", type=float, default=0.7, help="pixel confidence threshold (logits only)")
    ap.add_argument("--seg-ema", type=float, default=0.8, help="EMA factor for prob map (logits only)")
    ap.add_argument("--min-area", type=float, default=0.02, help="min connected area ratio (logits only)")

    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--nms-iou", type=float, default=0.4, help="extra NMS IoU on demo side")

    ap.add_argument("--smooth-alpha", type=float, default=0.5, help="EMA smoothing alpha for bbox")
    ap.add_argument("--smooth-iou", type=float, default=0.3, help="IoU match threshold for smoothing")
    ap.add_argument("--smooth-min-age", type=int, default=3, help="min consecutive frames to show bbox")

    # COCO dataset mode
    ap.add_argument("--ann-json", default=None, help="COCO json (train/val) containing ego_choice in images[]")
    ap.add_argument("--img-root", default=None, help="Root folder for COCO image file_name")
    ap.add_argument("--video-id", type=int, default=None, help="Select video id from COCO json")
    ap.add_argument("--video-name", default=None, help="Select video name from COCO json (optional)")
    ap.add_argument("--force-choice", default=None, choices=EGO_CHOICE_NAMES, help="Force a fixed ego choice for all frames (e.g. straight)")
    ap.add_argument(
        "--perception-only",
        action="store_true",
        help="Run only perception outputs: detection/segmentation/action/direction. Disable post-ego stop/go prediction and visualization.",
    )
    

    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()
    if args.perception_only:
        # Disable post-ego head at deploy time. This keeps the same perception model/checkpoint
        # but skips stop/go prediction and prevents post-ego visualization.
        try:
            if hasattr(model, "roi_head") and hasattr(model.roi_head, "ego_stopgo_head"):
                model.roi_head.ego_stopgo_head = None
            print("[DEMO] perception-only mode: ego_stopgo_head disabled.")
        except Exception as e:
            print(f"[DEMO] warning: failed to disable ego_stopgo_head: {e}")

    T = args.T
    pipeline = build_pipeline(cfg)

    os.makedirs(args.tempdir, exist_ok=True)

    use_coco = args.ann_json is not None

    cap = None
    frames = None

    if use_coco:
        assert args.img_root is not None, "--img-root is required when using --ann-json"
        frames, vid_name, _ = load_coco_video_frames(
            args.ann_json, args.img_root,
            video_id=args.video_id,
            video_name=args.video_name
        )
        first = cv2.imread(frames[0]["path"])
        assert first is not None, f"Failed to read first image: {frames[0]['path']}"
        H, W = first.shape[:2]
        fps = 25.0
        video_id = vid_name
    else:
        assert args.input is not None, "Either provide --input video OR use --ann-json/--img-root"
        cap = cv2.VideoCapture(args.input)
        assert cap.isOpened(), f"打开视频失败: {args.input}"
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_id = os.path.splitext(os.path.basename(args.input))[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_base, out_ext = os.path.splitext(args.output)
    out_action = out_base + "-bbox_action" + out_ext
    out_dir = out_base + "-bbox_dir" + out_ext

    vw_action = cv2.VideoWriter(out_action, fourcc, fps, (W, H))
    vw_dir = cv2.VideoWriter(out_dir, fourcc, fps, (W, H))

    buf = deque(maxlen=T)
    paths = deque(maxlen=T)

    frame_idx = 0
    tmp_dir = None
    if cap is not None:
        tmp_dir = os.path.join(args.tempdir, video_id)
        os.makedirs(tmp_dir, exist_ok=True)

    choice_state = ChoiceState(default="straight")
    prob_ema = None  # (C,H0,W0) EMA probability map
    prev_state = None  # bbox smoothing state

    while True:
        if frames is not None:
            if frame_idx >= len(frames):
                break
            frame_path = frames[frame_idx]["path"]
            frame = cv2.imread(frame_path)
            if frame is None:
                raise RuntimeError(f"Failed to read image: {frame_path}")

            # COCO mode: update choice from sparse annotations (interval repeat)
            choice_state.update_from_sparse(
                frames[frame_idx].get("choice_str", None),
                frames[frame_idx].get("choice_id", None),
            )

            buf.append(frame)
            paths.append(frame_path)
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmp_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            buf.append(frame)
            paths.append(frame_path)

        vis_img = buf[-1].copy()

        # sliding window inference (last frame)
        if len(paths) == T and (frame_idx % args.stride == T - 1 or args.stride == 1):
            sample = make_sample(list(paths), video_id, frame_idx)
            packed = pipeline(sample)

            if isinstance(packed, (tuple, list)) and len(packed) == 2:
                inputs, data_samples = packed
            elif isinstance(packed, dict):
                inputs = packed["inputs"]
                data_samples = packed["data_samples"]
            else:
                raise TypeError(f"pipeline 输出类型不支持: {type(packed)}")

            # imgs -> List[T][B=1]
            if isinstance(inputs, dict) and "imgs" in inputs:
                imgs_T = inputs["imgs"]
                if len(imgs_T) > 0 and not isinstance(imgs_T[0], (list, tuple)):
                    imgs_seq = [[img] for img in imgs_T]
                else:
                    imgs_seq = imgs_T
                inputs = {"imgs": imgs_seq}

            if isinstance(data_samples, DetDataSample):
                data_samples = [data_samples]
            elif not isinstance(data_samples, (list, tuple)):
                raise TypeError(f"data_samples 类型不支持: {type(data_samples)}")

            # Ego choice injection (ONLY ego_choice as input to ego head)
            if args.force_choice is not None:
                choice_str = str(args.force_choice).strip().lower()
                choice_id = int(EGO_CHOICE_TO_ID[choice_str])
            elif frames is not None:
                choice_str = choice_state.choice_str
                choice_id = int(choice_state.choice_id)
            else:
                sec = int(frame_idx // fps)
                choice_state.update_fallback_by_second(sec)
                choice_str = choice_state.choice_str
                choice_id = int(choice_state.choice_id)

            for ds in data_samples:
                if isinstance(ds, DetDataSample):
                    ds.set_metainfo({"ego_choice": choice_str, "ego_choice_id": choice_id})
                elif isinstance(ds, dict):
                    mi = ds.get("metainfo", {}) or {}
                    mi.update({"ego_choice": choice_str, "ego_choice_id": choice_id})
                    ds["metainfo"] = mi

            pred_input = dict(inputs=inputs, data_samples=data_samples)

            with torch.no_grad():
                pred_list = model.test_step(pred_input)

            pred_ds = pred_list[0] if isinstance(pred_list, (list, tuple)) else pred_list

            bboxes, labels, scores, actions, dirs, seg, seg_logits = extract_pred(pred_ds)
            if args.perception_only:
                stopgo_id, stopgo_score = None, None
            else:
                stopgo_id, stopgo_score = extract_stopgo(pred_ds)

            # extra NMS (demo side)
            if bboxes is not None and scores is not None and len(bboxes) > 0:
                b = torch.as_tensor(bboxes, dtype=torch.float32)
                s = torch.as_tensor(scores, dtype=torch.float32)
                keep = nms(b, s, iou_threshold=args.nms_iou).cpu().numpy()

                bboxes = bboxes[keep]
                scores = scores[keep]
                labels = labels[keep] if labels is not None else None
                actions = actions[keep] if actions is not None else None
                dirs = dirs[keep] if dirs is not None else None

            # bbox smoothing
            bboxes_s, labels_s, scores_s, actions_s, dirs_s, prev_state = temporal_smooth_boxes(
                prev_state,
                bboxes, labels, scores, actions, dirs,
                iou_thr=args.smooth_iou,
                alpha=args.smooth_alpha,
                min_age=args.smooth_min_age,
            )

            # -------------------------
            # Segmentation visualization
            # Prefer logits path for smoother / stricter visualization.
            # Fallback to raw seg label map only if logits are unavailable.
            # -------------------------
            meta = pred_ds.metainfo

            if seg_logits is not None:
                # logits path: use softmax + EMA + confidence threshold + area cleanup
                ih, iw = meta.get("img_shape", (None, None))
                logit = seg_logits  # (C,h,w)

                if ih and iw and (logit.shape[-2:] != (ih, iw)):
                    logit = F.interpolate(
                        logit.unsqueeze(0),
                        size=(ih, iw),
                        mode="bilinear",
                        align_corners=False,
                    )[0]  # (C, ih, iw)

                C = int(logit.shape[0])

                # warp each channel back to original image space
                logit_up = []
                for c in range(C):
                    ch = warp_seg_to_ori(logit[c].numpy(), meta)
                    logit_up.append(torch.from_numpy(ch)[None])
                logit_up = torch.cat(logit_up, dim=0)  # (C, H0, W0)

                prob = torch.softmax(logit_up, dim=0)  # (C,H0,W0)
                if (prob_ema is None) or (prob_ema.shape != prob.shape):
                    prob_ema = prob.clone()
                else:
                    prob_ema = args.seg_ema * prob_ema + (1.0 - args.seg_ema) * prob

                conf, lab = prob_ema.max(dim=0)  # lab ∈ [0..C-1]
                seg_pred = lab.numpy().astype(np.int32)
                conf_np = conf.numpy()

                # If the model includes an explicit background channel (id=6), do not visualize it.
                if C > SEG_BG_ID:
                    seg_pred[seg_pred == SEG_BG_ID] = IGNORE_INDEX

                # confidence mask: low-confidence pixels are suppressed
                seg_pred[conf_np < args.seg_thr] = IGNORE_INDEX

                # area cleanup on foreground classes only
                H0, W0 = seg_pred.shape
                min_pixels = int(args.min_area * H0 * W0)
                if min_pixels > 0:
                    seg_vis = seg_pred.copy()
                    for k in range(min(6, C)):
                        mk = (seg_vis == k)
                        if not mk.any():
                            continue
                        mk_u8 = (mk.astype(np.uint8) * 255)
                        num, comp, stats, _ = cv2.connectedComponentsWithStats(mk_u8, connectivity=4)
                        for i_cc in range(1, num):
                            if stats[i_cc, cv2.CC_STAT_AREA] < min_pixels:
                                seg_vis[comp == i_cc] = IGNORE_INDEX

                        # extra morphology for stricter, cleaner boundaries
                        cls_mask = (seg_vis == k).astype(np.uint8) * 255
                        if cls_mask.any():
                            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_OPEN, ker, iterations=1)
                            cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, ker, iterations=1)
                            seg_vis[seg_vis == k] = IGNORE_INDEX
                            seg_vis[cls_mask > 0] = k
                    seg_pred = seg_vis

                # resize to vis img
                if seg_pred.shape[:2] != vis_img.shape[:2]:
                    seg_pred = cv2.resize(seg_pred, (vis_img.shape[1], vis_img.shape[0]), interpolation=cv2.INTER_NEAREST)

                mask2 = (seg_pred != IGNORE_INDEX) & (seg_pred >= 0) & (seg_pred < 6) & (seg_pred != SEG_BG_ID)
                if mask2.any():
                    seg_clamped = np.clip(seg_pred, 0, 5)
                    color_img = PALETTE[seg_clamped]
                    alpha = 0.30
                    blended = (vis_img.astype(np.float32) * (1.0 - alpha) + color_img.astype(np.float32) * alpha).astype(np.uint8)
                    vis_img[mask2] = blended[mask2]

            elif seg is not None:
                # fallback path: raw seg label map (no logits available)
                seg_ori = warp_seg_to_ori(seg.astype(np.int32), meta)  # (H0,W0)

                if seg_ori.shape[:2] != vis_img.shape[:2]:
                    seg_ori = cv2.resize(seg_ori, (vis_img.shape[1], vis_img.shape[0]), interpolation=cv2.INTER_NEAREST)

                mask = (seg_ori != IGNORE_INDEX) & (seg_ori >= 0) & (seg_ori < 6) & (seg_ori != SEG_BG_ID)
                if mask.any():
                    seg_clamped = np.clip(seg_ori, 0, 5)
                    color_img = PALETTE[seg_clamped]
                    alpha = 0.30
                    blended = (vis_img.astype(np.float32) * (1.0 - alpha) + color_img.astype(np.float32) * alpha).astype(np.uint8)
                    vis_img[mask] = blended[mask]

            # -------------------------
            # Build two output videos
            # -------------------------
            vis_img_action = vis_img.copy()
            vis_img_dir = vis_img.copy()

            vis_img_action = draw_seg_legend(vis_img_action)
            vis_img_dir = draw_seg_legend(vis_img_dir)

            vis_img_action = draw_action_dir_legend(vis_img_action)
            vis_img_dir = draw_action_dir_legend(vis_img_dir)

            vis_img_action = draw_dets_obj_action(
                vis_img_action, bboxes_s, labels_s, scores_s, actions=actions_s, score_thr=args.score_thr
            )
            vis_img_dir = draw_dets_obj_dir(
                vis_img_dir, bboxes_s, labels_s, scores_s, dirs=dirs_s, score_thr=args.score_thr
            )

            if not args.perception_only:
                vis_img_action = draw_stopgo_banner(vis_img_action, stopgo_id, stopgo_score)
                vis_img_dir = draw_stopgo_banner(vis_img_dir, stopgo_id, stopgo_score)

            vis_img_action = draw_choice_banner(vis_img_action, choice_str)
            vis_img_dir = draw_choice_banner(vis_img_dir, choice_str)

            vw_action.write(vis_img_action)
            vw_dir.write(vis_img_dir)
        else:
            # before enough frames, write raw frame
            vw_action.write(vis_img)
            vw_dir.write(vis_img)

        frame_idx += 1

    vw_action.release()
    vw_dir.release()
    if cap is not None:
        cap.release()

    print(f"保存到: {out_action}")
    print(f"保存到: {out_dir}")


if __name__ == "__main__":
    main()


"""
# Example (video mode):
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python projects/vasco/demo_vasco.py \
  --config /home/tili/masterwork/transformer/mmdetection-3.3.0/projects/vasco/configs/vasco_swin2d_temporal.py \
  --checkpoint /home/tili/masterwork/transformer/mmdetection-3.3.0/work_dirs/vasco_swin_2d-ade20kbase-proposalfreeze-05dloss-topk2ad/best_coco_bbox_mAP_epoch_18.pth \
  --input /home/tili/masterwork/transformer/mmdetection-3.3.0/GX010102.MP4 \
  --output /home/tili/masterwork/transformer/mmdetection-3.3.0/GX010102-swinbase.MP4 \
  --T 8 
  --stride 1


# Example (COCO dataset mode, uses images[].ego_choice; supports post-ego checkpoint):
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python projects/vasco/demo_vasco.py \
  --config /home/tili/masterwork/transformer/mmdetection-3.3.0/work_dirs/vasco_swin_2d-ade20klarge-proposalfreeze-segv2-allweight1-topk10adclassweightdir-2s-det+adv4+seg-ego-t5/vasco_swin2d_temporal.py \
  --checkpoint /home/tili/masterwork/transformer/mmdetection-3.3.0/work_dirs/vasco_swin_2d-ade20klarge-proposalfreeze-segv2-allweight1-topk10adclassweightdir-2s-det+adv4+seg-ego-t5/epoch_10.pth \
  --ann-json /home/tili/masterwork/transformer/VASCO/vasco_train/train.json \
  --img-root /home/tili/masterwork/transformer/VASCO/vasco_train \
  --video-id 19 \
  --output /home/tili/masterwork/transformer/mmdetection-3.3.0/work_dirs/vasco_swin_2d-ade20klarge-proposalfreeze-segv2-allweight1-topk10adclassweightdir-2s-det+adv4+seg-ego-t5/vasco_post_ego_video19.MP4 \
  --T 5 \
  --stride 1

# Example (COCO dataset mode, perception-only; no post-ego STOP/GO):
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python projects/vasco/demo_vasco.py \
  --config /home/tili/masterwork/transformer/mmdetection-3.3.0/work_dirs/vasco_swin_2d-ade20klarge-proposalfreeze-segv2-allweight1-topk10adclassweightdir-2s-det+adv4+seg-egonone/vasco_swin2d_temporal.py \
  --checkpoint /home/tili/masterwork/transformer/mmdetection-3.3.0/work_dirs/vasco_swin_2d-ade20klarge-proposalfreeze-segv2-allweight1-topk10adclassweightdir-2s-det+adv4+seg-egonone/epoch_20.pth \
  --ann-json /home/tili/masterwork/transformer/VASCO/vasco_train/train.json \
  --img-root /home/tili/masterwork/transformer/VASCO/vasco_train \
  --video-id 19 \
  --output /home/tili/masterwork/transformer/mmdetection-3.3.0/work_dirs/vasco_swin_2d-ade20klarge-proposalfreeze-segv2-allweight1-topk10adclassweightdir-2s-det+adv4+seg-egonone/vasco_perception_only_video19.MP4 \
  --T 8 \
  --stride 1 \
  --perception-only

"""