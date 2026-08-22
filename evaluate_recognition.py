"""
evaluate_recognition.py
-----------------------
Recognition evaluation for the Glove Defect Detection System.

This is the third of the project's three team-wide evaluation scripts.
Each answers a different question about the same 12 detectors:

    evaluate.py               Sensitivity.  Given that an image really
                              contains defect X, does detector X fire?

    fp_sweep.py               Specificity.  Given an image that is NOT
                              defect X, does detector X wrongly fire?

    evaluate_recognition.py   Recognition.  Given an unlabelled image,
                              does the SYSTEM name the right defect?

The third question is the one the assignment brief actually asks
("segment & recognize the defects"), and it is the one the GUI has to
answer in auto-detect mode, where no folder label is available to pick
a detector for it.

Neither of the other two scripts can answer it, because both are
organised around one detector at a time. This script runs every
implemented detector on every image and asks which one scores highest.

Method
------
The per-image ranking approach is Ka Sin's, first written as
evaluate_kasin.py for her own three detectors (dirty, stain,
spotting). That script found that a detector can fire correctly and
still be out-scored by a sibling, which counts as a pass under
evaluate.py but produces a wrong answer in the GUI. This script
generalises that method to all 12 detectors and adds the confusion
matrix. evaluate_kasin.py is superseded by this file.

Ka Sin's geometry/colour grouping is also kept, because it changes how
a confusion should be read:

  - geometry defects have no colour anomaly by definition, so a
    colour-based detector winning on one is a genuine error.

  - colour defects really do contain colour anomalies, so one colour
    detector out-scoring another is a much softer failure - the
    evidence is real, the label is just wrong.

Run from the project root:

    python evaluate_recognition.py
    python evaluate_recognition.py --material latex
    python evaluate_recognition.py --no-csv

Output
------
Prints top-1 accuracy overall, per defect and per material; a confusion
matrix; the rank the correct detector achieved when it did not win; and
a geometry/colour breakdown of the errors. Writes every per-image,
per-detector score to outputs/recognition_results.csv.
"""

import argparse
import csv
import os
import traceback

from evaluate import (
    DETECTOR_REGISTRY,
    DETECTION_THRESHOLD,
    MIN_GLOVE_AREA,
    OUTPUT_ROOT,
    discover_images,
    load_detector,
    validate_result,
)
from preprocessing import load_image, preprocess_image
from segmentation import segment_glove

RECOGNITION_CSV = os.path.join(OUTPUT_ROOT, "recognition_results.csv")

# Label used when no detector at all clears DETECTION_THRESHOLD on an
# image. Treated as a wrong answer, not as an excluded image: a system
# that says nothing has still failed to recognise the defect.
NO_PREDICTION = "(none)"

# Defect grouping, carried over from evaluate_kasin.py. Every name here
# must match a datasets/ folder name and a DETECTOR_REGISTRY key.
GEOMETRY_DEFECTS = {
    "tearing",
    "tearing_fingertip",
    "finger_not_enough",
    "touching",
    "damaged_by_fold",
    "incomplete_beading",
    "oversize",
}
COLOUR_DEFECTS = {
    "dirty",
    "stain",
    "spotting",
    "discoloration",
    "plastic_contamination",
}


# ============================================================
# DETECTOR SELECTION
# ============================================================

def implemented_detectors():
    """Every DETECTOR_REGISTRY entry that currently imports successfully."""
    return sorted(name for name in DETECTOR_REGISTRY if load_detector(name) is not None)


def defect_group(defect_name):
    """geometry / colour / unclassified, for reading confusions."""
    if defect_name in GEOMETRY_DEFECTS:
        return "geometry"
    if defect_name in COLOUR_DEFECTS:
        return "colour"
    return "unclassified"


# ============================================================
# SCORING
# ============================================================

def score_all_detectors(processed, segmentation, detectors):
    """
    Run every detector on one already-preprocessed image.

    Returns (scores, errors) where scores maps defect name to float and
    errors maps defect name to a one-line reason. A detector that
    raises is recorded at score 0.0 rather than removed, so one broken
    detector cannot silently shrink another image's candidate list and
    hand the win to a detector that would otherwise have lost.
    """
    scores = {}
    errors = {}

    for name in detectors:
        detector_func = load_detector(name)
        try:
            result = detector_func(processed, segmentation)
            validate_result(result)
            scores[name] = float(result["detection_score"])
        except Exception as exc:
            scores[name] = 0.0
            errors[name] = f"{type(exc).__name__}: {exc}"
            traceback.format_exc(limit=1)

    return scores, errors


def evaluate_one_image(material, true_defect, image_path, detectors):
    """
    Full recognition record for one image.

    predicted is the highest-scoring detector, but only if it clears
    DETECTION_THRESHOLD. If nothing clears it the system has abstained
    and predicted is NO_PREDICTION.

    correct_rank is where the CORRECT detector placed once all scores
    are sorted high to low, counting from 1. Rank 1 means the system
    got it right. A near miss at rank 2 and a total miss at rank 11 are
    both simply "wrong" in the accuracy figure, so the rank is kept
    separately - it is the difference between a detector that needs
    retuning and one that has no signal at all.
    """
    record = {
        "material": material,
        "true_defect": true_defect,
        "image_path": image_path,
        "status": "success",
        "predicted": NO_PREDICTION,
        "top1_correct": False,
        "correct_score": None,
        "correct_fired": False,
        "correct_rank": None,
        "winner_score": None,
        "margin": None,
        "scores": {},
        "errors": {},
    }

    try:
        image = load_image(image_path)
        processed = preprocess_image(image)
        segmentation = segment_glove(processed)
    except Exception as exc:
        record["status"] = "segmentation_failure"
        record["errors"]["_pipeline"] = f"{type(exc).__name__}: {exc}"
        return record

    if segmentation["glove_area"] < MIN_GLOVE_AREA:
        record["status"] = "segmentation_failure"
        record["errors"]["_pipeline"] = "glove_area below MIN_GLOVE_AREA"
        return record

    scores, errors = score_all_detectors(processed, segmentation, detectors)
    record["scores"] = scores
    record["errors"] = errors

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    winner_name, winner_score = ranked[0]
    record["winner_score"] = winner_score

    if winner_score >= DETECTION_THRESHOLD:
        record["predicted"] = winner_name

    if true_defect in scores:
        record["correct_score"] = scores[true_defect]
        record["correct_fired"] = scores[true_defect] >= DETECTION_THRESHOLD
        record["correct_rank"] = [name for name, _ in ranked].index(true_defect) + 1

    record["top1_correct"] = record["predicted"] == true_defect

    # How far ahead the winner was. A tiny margin means the ranking is
    # fragile even when it happens to be right.
    if len(ranked) > 1:
        record["margin"] = round(ranked[0][1] - ranked[1][1], 3)

    return record


# ============================================================
# REPORTING
# ============================================================

def print_overall(records):
    usable = [r for r in records if r["status"] == "success"]
    skipped = len(records) - len(usable)

    correct = [r for r in usable if r["top1_correct"]]
    abstained = [r for r in usable if r["predicted"] == NO_PREDICTION]
    fired_but_lost = [
        r for r in usable
        if r["correct_fired"] and not r["top1_correct"]
    ]

    print("=" * 74)
    print("RECOGNITION SUMMARY")
    print("=" * 74)
    print(f"Images evaluated: {len(usable)}"
          + (f"  ({skipped} skipped, segmentation failure)" if skipped else ""))
    print(f"Top-1 accuracy: {len(correct)}/{len(usable)} "
          f"({100.0 * len(correct) / len(usable):.1f}%)" if usable else "no images")
    print(f"System named no defect at all: {len(abstained)}")
    print(f"Correct detector fired but was out-scored: {len(fired_but_lost)}")
    print()
    print("Read together with evaluate.py's detection rate: an image in")
    print("that last group passes evaluate.py and still shows the wrong")
    print("defect name in the GUI.")


def print_per_defect(records, detectors):
    usable = [r for r in records if r["status"] == "success"]
    print()
    print("=" * 74)
    print("TOP-1 ACCURACY BY TRUE DEFECT")
    print("=" * 74)
    print(f"{'defect':24s} {'top-1':>9s} {'fired':>9s} {'mean rank':>10s}")

    for defect in sorted(set(r["true_defect"] for r in usable)):
        subset = [r for r in usable if r["true_defect"] == defect]
        wins = sum(1 for r in subset if r["top1_correct"])
        fired = sum(1 for r in subset if r["correct_fired"])
        ranks = [r["correct_rank"] for r in subset if r["correct_rank"]]
        mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        print(f"{defect:24s} {wins:4d}/{len(subset):<4d} {fired:4d}/{len(subset):<4d} "
              f"{mean_rank:10.1f}")


def print_per_material(records):
    usable = [r for r in records if r["status"] == "success"]
    print()
    print("=" * 74)
    print("TOP-1 ACCURACY BY MATERIAL")
    print("=" * 74)
    for material in sorted(set(r["material"] for r in usable)):
        subset = [r for r in usable if r["material"] == material]
        wins = sum(1 for r in subset if r["top1_correct"])
        print(f"{material:12s} {wins:3d}/{len(subset):<3d} "
              f"({100.0 * wins / len(subset):5.1f}%)")


def print_confusion(records, detectors):
    """
    Confusion matrix, true defect down the side, prediction across.

    Printed as a list of rows rather than a wide grid, because 12
    columns of counts does not fit a terminal legibly and this form
    pastes into a report table more cleanly.
    """
    usable = [r for r in records if r["status"] == "success"]
    print()
    print("=" * 74)
    print("CONFUSION: what the system said instead")
    print("=" * 74)

    for defect in sorted(set(r["true_defect"] for r in usable)):
        subset = [r for r in usable if r["true_defect"] == defect]
        counts = {}
        for r in subset:
            counts[r["predicted"]] = counts.get(r["predicted"], 0) + 1

        wins = counts.get(defect, 0)
        wrong = sorted(
            ((name, n) for name, n in counts.items() if name != defect),
            key=lambda pair: pair[1],
            reverse=True,
        )

        print(f"\n{defect}  ({wins}/{len(subset)} correct)")
        if not wrong:
            print("   no confusions")
            continue
        for name, n in wrong:
            if name == NO_PREDICTION:
                print(f"   {n}x nothing named")
                continue
            note = ""
            if defect_group(defect) == "geometry" and defect_group(name) == "colour":
                note = "   <- colour detector won on a geometry defect"
            elif defect_group(defect) == defect_group(name):
                note = f"   ({defect_group(name)} vs {defect_group(name)})"
            print(f"   {n}x {name}{note}")


def print_group_breakdown(records):
    usable = [r for r in records if r["status"] == "success"]
    errors = [r for r in usable if not r["top1_correct"]
              and r["predicted"] != NO_PREDICTION]

    print()
    print("=" * 74)
    print("ERROR TYPES BY DEFECT GROUP")
    print("=" * 74)

    buckets = {}
    for r in errors:
        key = (defect_group(r["true_defect"]), defect_group(r["predicted"]))
        buckets[key] = buckets.get(key, 0) + 1

    if not buckets:
        print("No wrong-label errors.")
        return

    for (true_group, pred_group), n in sorted(
        buckets.items(), key=lambda pair: pair[1], reverse=True
    ):
        if true_group == "geometry" and pred_group == "colour":
            reading = "hard error, no colour anomaly should exist"
        elif true_group == "colour" and pred_group == "colour":
            reading = "soft error, real anomaly, wrong label"
        elif true_group == pred_group:
            reading = "same group, wrong label"
        else:
            reading = "cross-group"
        print(f"   {n:3d}  true {true_group:12s} -> said {pred_group:12s}  {reading}")


def print_fragile(records, limit=10):
    """Correct answers that only just won. Right today, fragile tomorrow."""
    usable = [
        r for r in records
        if r["status"] == "success" and r["top1_correct"] and r["margin"] is not None
    ]
    tight = sorted(usable, key=lambda r: r["margin"])[:limit]

    print()
    print("=" * 74)
    print("NARROWEST CORRECT WINS")
    print("=" * 74)
    if not tight:
        print("None.")
        return
    for r in tight:
        print(f"   margin {r['margin']:.3f}  {r['material']}/{r['true_defect']}/"
              f"{os.path.basename(r['image_path'])}")


def write_csv(records, detectors):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fieldnames = [
        "material", "true_defect", "image_path", "status",
        "predicted", "top1_correct", "correct_score", "correct_fired",
        "correct_rank", "winner_score", "margin",
    ] + [f"score_{name}" for name in detectors]

    with open(RECOGNITION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in fieldnames if not k.startswith("score_")}
            for name in detectors:
                row[f"score_{name}"] = r["scores"].get(name)
            writer.writerow(row)

    print(f"\nPer-image scores: {RECOGNITION_CSV}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run every implemented detector on every image and measure "
                    "whether the highest-scoring one is the correct one.",
    )
    parser.add_argument(
        "--material", "-m", default=None,
        help="Only evaluate this material, e.g. --material latex",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Print the report without writing outputs/recognition_results.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    detectors = implemented_detectors()
    if not detectors:
        raise SystemExit("No detectors in DETECTOR_REGISTRY import successfully.")

    images = [
        (material, defect, path)
        for (material, defect, path) in discover_images()
        if args.material is None or material == args.material
    ]
    if not images:
        raise SystemExit("No images matched. Check --material.")

    print(f"Detectors: {len(detectors)}  ({', '.join(detectors)})")
    print(f"Images: {len(images)}")
    print(f"Detector runs: {len(detectors) * len(images)}")
    print("This runs every detector on every image, so it takes noticeably")
    print("longer than evaluate.py. Preprocessing is done once per image.")
    print()

    records = []
    for index, (material, defect, path) in enumerate(images, start=1):
        record = evaluate_one_image(material, defect, path, detectors)
        records.append(record)
        mark = "ok " if record["top1_correct"] else "MISS"
        print(f"[{index:3d}/{len(images)}] {mark} {material}/{defect}/"
              f"{os.path.basename(path)[:34]:34s} said: {record['predicted']}")

    print()
    print_overall(records)
    print_per_defect(records, detectors)
    print_per_material(records)
    print_confusion(records, detectors)
    print_group_breakdown(records)
    print_fragile(records)

    if not args.no_csv:
        write_csv(records, detectors)


if __name__ == "__main__":
    main()
