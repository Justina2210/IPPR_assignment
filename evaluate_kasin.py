"""
evaluate_kasin.py
-----------------
Member-level evaluation for Ka Sin's three detectors (dirty, stain,
spotting).

This sits alongside the group's shared evaluate.py rather than
replacing it. evaluate.py answers "does each detector fire on its own
labelled folder?", which is the number the group reports. This script
answers two further questions that the shared script cannot, because it
only ever runs one detector per image:

1. Can the three detectors tell each other apart? The GUI has to name
   the defect, so a detector that fires correctly but is out-scored by
   a sibling is still a problem.

2. What do they do on images that are NOT their defect? Specificity is
   split into two groups, because lumping them together hides the
   finding:

   - geometry defects (tearing, touching, folds, beading, oversize,
     finger-not-enough) have no colour anomaly by definition, so any
     response here is a genuine false positive, usually a fold shadow.

   - colour defects (discoloration, plastic contamination) genuinely
     do contain colour anomalies. A colour-based detector responding to
     them is an expected and arguably correct low-level response that
     the system should resolve by comparing detector scores, not a bug.

Run from the project root:  python evaluate_kasin.py
"""

import glob
import os

import numpy as np

from preprocessing import load_image, preprocess_image
from segmentation import segment_glove
from detectors.dirty import detect_dirty
from detectors.spotting import detect_spotting
from detectors.stain import detect_stain

DETECTORS = {
    "dirty": detect_dirty,
    "stain": detect_stain,
    "spotting": detect_spotting,
}

MINE = ("dirty", "stain", "spotting")

GEOMETRY_DEFECTS = {
    "tearing", "tearing_fingertip", "finger_not_enough", "touching",
    "damaged_by_fold", "fold", "incomplete_beading", "oversize",
}
COLOUR_DEFECTS = {"discoloration", "plastic_contamination"}

THRESHOLD = 0.5
DATASET_ROOT = "datasets" if os.path.isdir("datasets") else "dataset"


def score_all(image_path):
    """Run all three of this member's detectors on one image."""
    processed = preprocess_image(load_image(image_path))
    segmentation = segment_glove(processed)
    results = {}
    for name, function in DETECTORS.items():
        outcome = function(processed, segmentation)
        results[name] = outcome["detection_score"]
    return results


def main():
    print("=" * 78)
    print("PART 1  Detection on own folders, and mutual discrimination")
    print("=" * 78)
    print(f"{'image':32s} {'dirty':>7s} {'stain':>7s} {'spot':>7s}  {'hit':>4s}  top")

    detected = {d: 0 for d in MINE}
    total = {d: 0 for d in MINE}
    top_correct = {d: 0 for d in MINE}

    for defect in MINE:
        print(f"--- true label: {defect} ---")
        for path in sorted(glob.glob(f"{DATASET_ROOT}/*/{defect}/*")):
            scores = score_all(path)
            winner = max(scores, key=scores.get)
            total[defect] += 1
            if scores[defect] >= THRESHOLD:
                detected[defect] += 1
            if winner == defect:
                top_correct[defect] += 1
            print(f"{os.path.basename(path)[:32]:32s} {scores['dirty']:7.3f} "
                  f"{scores['stain']:7.3f} {scores['spotting']:7.3f}  "
                  f"{'yes' if scores[defect] >= THRESHOLD else 'NO':>4s}  {winner}")

    print("\nDetection rate (own detector, threshold 0.5):")
    for defect in MINE:
        print(f"   {defect:9s} {detected[defect]}/{total[defect]}")
    print("Highest-scoring detector is the correct one:")
    for defect in MINE:
        print(f"   {defect:9s} {top_correct[defect]}/{total[defect]}")

    print()
    print("=" * 78)
    print("PART 2  Response on images that are not this member's defects")
    print("=" * 78)

    groups = {"geometry": [], "colour": []}
    for path in sorted(glob.glob(f"{DATASET_ROOT}/*/*/*")):
        defect = path.split(os.sep)[-2] if os.sep in path else path.split("/")[-2]
        if defect in MINE:
            continue
        if defect in GEOMETRY_DEFECTS:
            groups["geometry"].append(path)
        elif defect in COLOUR_DEFECTS:
            groups["colour"].append(path)

    for group_name, paths in groups.items():
        if not paths:
            continue
        print(f"\n--- {group_name} defects ({len(paths)} images) ---")
        fired = {d: 0 for d in MINE}
        peaks = {d: (0.0, "") for d in MINE}
        for path in paths:
            scores = score_all(path)
            for defect in MINE:
                if scores[defect] >= THRESHOLD:
                    fired[defect] += 1
                if scores[defect] > peaks[defect][0]:
                    peaks[defect] = (scores[defect], os.path.basename(path))
        for defect in MINE:
            rate = 100.0 * fired[defect] / len(paths)
            print(f"   {defect:9s} fired on {fired[defect]:2d}/{len(paths)} "
                  f"({rate:5.1f}%)   highest: {peaks[defect][1]} ({peaks[defect][0]:.3f})")


if __name__ == "__main__":
    main()
