import json
from collections import defaultdict

# ----------------------------
# Config
# ----------------------------
ANN_FILE = "/home/tili/masterwork/transformer/VASCO/vasco_train/train.json"

# If your thing classes are exactly 0..27
THING_MAX_ID = 27

# ----------------------------
# Your label maps (optional fallback)
# ----------------------------
THING_NAME_TO_ID = {
    "bicycle": 0, "bus": 1, "bus_stop_sign": 2, "car": 3, "construction_barriers": 4,
    "danger_children_sign": 5, "green_light_p": 6, "green_light_v": 7, "handrail": 8,
    "motorcycle": 9, "orange_light_p": 10, "orange_light_v": 11, "pedestrian_crossing_sign": 12,
    "pedestrian_path_sign": 13, "pedestrian_prohibited_sign": 14,
    "pedestrians_must_cross_road_left_sign": 15, "pedestrians_must_cross_road_right_sign": 16,
    "person": 17, "pole": 18, "red_light_p": 19, "red_light_v": 20, "scooter": 21,
    "traffic_calming_zone_sign": 22, "trash_bin": 23, "tree": 24, "truck": 25,
    "van": 26, "warning_sign": 27,
}

ACTION_KEY_CANDIDATES = ("action_id", "action")
DIR_KEY_CANDIDATES = ("direction_id", "dir")

DEFAULT_ACTION_ID = -1
DEFAULT_DIR_ID = -1


def load_coco(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cat_maps(coco):
    # prefer COCO categories
    cat_id_to_name = {}
    for c in coco.get("categories", []):
        cid = int(c["id"])
        cat_id_to_name[cid] = c.get("name", str(cid))

    # fallback if categories missing
    if not cat_id_to_name and THING_NAME_TO_ID:
        cat_id_to_name = {v: k for k, v in THING_NAME_TO_ID.items()}

    cat_name_to_id = {n: i for i, n in cat_id_to_name.items()}
    return cat_id_to_name, cat_name_to_id


def pick_attr(attrs: dict, keys, default=-1):
    if not isinstance(attrs, dict):
        return default
    for k in keys:
        if k in attrs:
            return attrs.get(k, default)
    return default


def to_int(x, default=-1):
    try:
        return int(x)
    except Exception:
        return default


def main():
    coco = load_coco(ANN_FILE)
    cat_id_to_name, _ = build_cat_maps(coco)

    anns = coco.get("annotations", [])
    if not anns:
        raise RuntimeError("No annotations found in json.")

    # counters: which categories have action=-1 / dir=-1
    action_neg1_by_cat = defaultdict(int)
    dir_neg1_by_cat = defaultdict(int)

    # totals for sanity
    total_thing_anns = 0
    total_action_neg1 = 0
    total_dir_neg1 = 0

    # also count per category total things to compute ratio
    total_by_cat = defaultdict(int)

    for a in anns:
        cid = to_int(a.get("category_id", -999), -999)
        if cid < 0 or cid > THING_MAX_ID:
            continue  # ignore stuff or invalid
        total_thing_anns += 1
        total_by_cat[cid] += 1

        attrs = a.get("attributes", {}) or {}

        action_id = to_int(pick_attr(attrs, ACTION_KEY_CANDIDATES, DEFAULT_ACTION_ID), DEFAULT_ACTION_ID)
        dir_id = to_int(pick_attr(attrs, DIR_KEY_CANDIDATES, DEFAULT_DIR_ID), DEFAULT_DIR_ID)

        if action_id == -1:
            total_action_neg1 += 1
            action_neg1_by_cat[cid] += 1

        if dir_id == -1:
            total_dir_neg1 += 1
            dir_neg1_by_cat[cid] += 1

    # sort by count desc
    action_sorted = sorted(action_neg1_by_cat.items(), key=lambda x: x[1], reverse=True)
    dir_sorted = sorted(dir_neg1_by_cat.items(), key=lambda x: x[1], reverse=True)

    print("========== SUMMARY ==========")
    print(f"Total thing annotations: {total_thing_anns}")
    print(f"Total action == -1:      {total_action_neg1}  ({(total_action_neg1 / max(1,total_thing_anns))*100:.2f}%)")
    print(f"Total direction == -1:   {total_dir_neg1}  ({(total_dir_neg1 / max(1,total_thing_anns))*100:.2f}%)")
    print()

    print("========== ACTION == -1 by category ==========")
    if not action_sorted:
        print("No action == -1 found.")
    else:
        for cid, cnt in action_sorted:
            name = cat_id_to_name.get(cid, str(cid))
            tot = total_by_cat.get(cid, 0)
            ratio = (cnt / tot * 100.0) if tot else 0.0
            print(f"[{cid:>2}] {name:<40}  neg1={cnt:<8}  total={tot:<8}  ratio={ratio:6.2f}%")
    print()

    print("========== DIRECTION == -1 by category ==========")
    if not dir_sorted:
        print("No direction == -1 found.")
    else:
        for cid, cnt in dir_sorted:
            name = cat_id_to_name.get(cid, str(cid))
            tot = total_by_cat.get(cid, 0)
            ratio = (cnt / tot * 100.0) if tot else 0.0
            print(f"[{cid:>2}] {name:<40}  neg1={cnt:<8}  total={tot:<8}  ratio={ratio:6.2f}%")
    print()

    # Optional: categories where all are -1 (useful to catch systematic missing labels)
    all_action_neg1 = [cid for cid, tot in total_by_cat.items() if tot > 0 and action_neg1_by_cat.get(cid, 0) == tot]
    all_dir_neg1 = [cid for cid, tot in total_by_cat.items() if tot > 0 and dir_neg1_by_cat.get(cid, 0) == tot]

    print("========== ALL -1 categories (diagnostic) ==========")
    print("Action all -1 categories:")
    if not all_action_neg1:
        print("  None")
    else:
        for cid in sorted(all_action_neg1):
            print(f"  [{cid}] {cat_id_to_name.get(cid, str(cid))}")
    print("Direction all -1 categories:")
    if not all_dir_neg1:
        print("  None")
    else:
        for cid in sorted(all_dir_neg1):
            print(f"  [{cid}] {cat_id_to_name.get(cid, str(cid))}")


if __name__ == "__main__":
    main()