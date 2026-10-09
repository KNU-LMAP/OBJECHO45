"""Generate the OBJECHO45 metadata tables from the clean 4 ms recordings.

Writes three CSV files to metadata/:
  clips.csv    one row per recording (13,419), parsed from the filename
  classes.csv  the 45 object-angle classes, with the class index the paper's models use
  folds_10.csv the 10-fold split used in the paper

Only filenames are read, so this runs in seconds. The fold split reproduces the paper's
protocol exactly: clips sorted by filename, classes sorted by name, then
StratifiedKFold(n_splits=10, shuffle=True, random_state=1).

Usage:
  python scripts/make_metadata.py --data-dir data/clean_4ms --out-dir metadata
"""
import argparse
import csv
import os
from collections import Counter

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold

K_FOLDS = 10
SEED = 1


def is_excluded(obj, angle):
    """Classes left out of OBJECHO45 because a symmetry makes them duplicates of another."""
    # sphere is rotationally symmetric: every angle presents the same shape
    if obj == "sphere" and angle != "00":
        return True
    # cube is four-fold symmetric: cube_90 presents the same face as cube_00
    if obj == "cube" and angle == "90":
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/clean_4ms")
    parser.add_argument("--out-dir", default="metadata")
    args = parser.parse_args()

    # Filename convention: <object>_<angle>_<distance>_<size>_<index>.wav
    filenames = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".wav"))
    clips = []
    for filename in filenames:
        clip_id = os.path.splitext(filename)[0]
        obj, angle, distance, size, idx = clip_id.split("_")
        if is_excluded(obj, angle):
            continue
        clips.append({
            "clip_id": clip_id,
            "object": obj,
            "angle_deg": int(angle),
            "distance_m": float(distance.rstrip("m")),
            "size_cm": int(size),
            "index": int(idx),
            "class": f"{obj}_{angle}",
        })

    # Class indices follow sorted class names -- the order the paper's checkpoints use
    counts = Counter(c["class"] for c in clips)
    classes = sorted(counts)
    class_index = {name: i for i, name in enumerate(classes)}

    # Stratified split over all clips, in filename order
    targets = np.array([class_index[c["class"]] for c in clips])
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_of = {}
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        for i in val_idx:
            fold_of[clips[i]["clip_id"]] = k

    # Sanity checks against the numbers reported in the paper
    assert len(clips) == 13419, len(clips)
    assert len(classes) == 45, len(classes)
    assert len(fold_of) == len(clips)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "clips.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(clips[0]))
        writer.writeheader()
        writer.writerows(clips)

    with open(os.path.join(args.out_dir, "classes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class", "object", "angle_deg", "n_clips"])
        for name in classes:
            obj, angle = name.split("_")
            writer.writerow([class_index[name], name, obj, int(angle), counts[name]])

    with open(os.path.join(args.out_dir, "folds_10.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "fold"])
        for c in clips:
            writer.writerow([c["clip_id"], fold_of[c["clip_id"]]])

    fold_sizes = Counter(fold_of.values())
    print(f"{len(clips)} clips, {len(classes)} classes "
          f"({min(counts.values())}-{max(counts.values())} clips per class)")
    print(f"fold sizes: {[fold_sizes[k] for k in range(1, K_FOLDS + 1)]}")
    print(f"written to {args.out_dir}/ (scikit-learn {sklearn.__version__})")


if __name__ == "__main__":
    main()
