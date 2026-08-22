"""
fp_sweep.py
-----------
False-positive sweep for the Glove Defect Detection System.

evaluate.py's own run only ever tests a detector against the ONE
folder matching its name (each labelled image is only ever tested
against its own matching detector - see evaluate.py's closing
docstring), so it has no false-positive metric. This script fills that
gap: for a given detector, it runs that detector against every image
whose folder is NOT its own defect, using the exact same detector-
loading path and detected/threshold rule evaluate.py uses, so the
numbers here are directly comparable to evaluate.py's output.

Does not modify evaluate.py or any detector - it only imports from
evaluate.py, preprocessing.py and segmentation.py.

Usage
-----
    python fp_sweep.py tearing              # one detector
    python fp_sweep.py tearing_fingertip
    python fp_sweep.py finger_not_enough
    python fp_sweep.py all                  # every implemented detector

Works for any detector name in evaluate.DETECTOR_REGISTRY, not just
the three currently implemented ones - a teammate's newly registered
detector is picked up automatically once it imports successfully.

Output
------
Prints, per detector: how many non-matching images were tested, the
false-positive count, and the list of wrongly-flagged files with their
scores. Also writes every tested (detector, image) pair to
outputs/fp_sweep.csv.
"""

import argparse
import csv
import os
import sys
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

FP_SWEEP_CSV = os.path.join(OUTPUT_ROOT, "fp_sweep.csv")


# ============================================================
# DETECTOR SELECTION
# ============================================================

def implemented_detectors():
    """Every DETECTOR_REGISTRY entry that currently imports successfully."""
    return sorted(name for name in DETECTOR_REGISTRY if load_detector(name) is not None)


def resolve_targets(requested):
    """
    Turn the command-line argument into a list of defect names to sweep.

    "all" -> every currently-implemented detector.
    A specific name -> that one, if it's registered and importable
    (fails loudly and clearly otherwise, rather than silently skipping
    it the way evaluate.py's own loader does for a whole-dataset run).
    """
    if requested == "all":
        targets = implemented_detectors()
        if not targets:
            sys.exit("No detectors in DETECTOR_REGISTRY currently import successfully.")
        return targets

    if requested not in DETECTOR_REGISTRY:
        sys.exit(
            f"'{requested}' is not a key in evaluate.DETECTOR_REGISTRY. "
            f"Known defect names: {', '.join(sorted(DETECTOR_REGISTRY))}"
        )
    if load_detector(requested) is None:
        sys.exit(f"'{requested}' is registered but its module/function failed to import.")
    return [requested]


# ============================================================
# PREPROCESS/SEGMENT CACHE
# ============================================================

def build_image_cache(images):
    """
    Preprocessing + segmentation is identical no matter which detector
    runs next, so it's computed once per image here and reused for
    every detector in this sweep, instead of redoing it once per
    detector (mirrors app.py's run_full_evaluation()).
    """
    cache = {}
    for material, defect_folder, image_path in images:
        try:
            image = load_image(image_path)
            processed = preprocess_image(image)
            segmentation = segment_glove(processed)
        except Exception:
            cache[image_path] = (None, None)
            continue
        cache[image_path] = (processed, segmentation)
    return cache


# ============================================================
# SWEEP
# ============================================================

def sweep_one_detector(defect_name, other_images, cache):
    """
    Run `defect_name`'s detector against every image in other_images
    (all images whose own folder label is NOT defect_name), using the
    same detected/threshold rule as evaluate.evaluate_image():
        detected = bool(result["detected"]) and result["detection_score"] >= DETECTION_THRESHOLD

    Returns a list of per-image record dicts (one per image tested).
    """
    detector_func = load_detector(defect_name)
    records = []

    for material, defect_folder, image_path in other_images:
        record = {
            "detector": defect_name,
            "material": material,
            "defect_folder": defect_folder,
            "image_path": image_path,
            "status": None,
            "detection_score": 0.0,
            "detected": False,
            "false_positive": False,
            "error": None,
        }

        processed, segmentation = cache.get(image_path, (None, None))
        if processed is None or segmentation is None:
            record["status"] = "segmentation_failure"
            records.append(record)
            continue

        if segmentation["glove_area"] < MIN_GLOVE_AREA:
            record["status"] = "segmentation_failure"
            records.append(record)
            continue

        try:
            result = detector_func(processed, segmentation)
            validate_result(result)
        except Exception as exc:
            record["status"] = "detector_failure"
            record["error"] = f"{exc}\n{traceback.format_exc(limit=2)}"
            records.append(record)
            continue

        detected = bool(result["detected"]) and float(result["detection_score"]) >= DETECTION_THRESHOLD
        record["status"] = "success"
        record["detection_score"] = float(result["detection_score"])
        record["detected"] = detected
        record["false_positive"] = detected  # detected on a non-matching image = false positive
        records.append(record)

    return records


# ============================================================
# REPORTING
# ============================================================

def print_report(defect_name, records):
    # This script's own denominator: only images the detector actually
    # RAN on successfully. Any segmentation_failure or detector_failure
    # is excluded from both the numerator and the denominator - it's
    # neither a false positive nor evidence of a true negative, just a
    # run that didn't produce a usable result.
    tested = [r for r in records if r["status"] == "success"]
    false_positives = [r for r in records if r["false_positive"]]
    detector_failures = [r for r in records if r["status"] == "detector_failure"]
    seg_failures = [r for r in records if r["status"] == "segmentation_failure"]
    excluded = detector_failures + seg_failures

    print(f"\n=== {defect_name} ===")
    print(f"Non-matching images tested (status == success): {len(tested)}"
          + (f"  (+{len(excluded)} excluded: {len(seg_failures)} segmentation_failure, "
             f"{len(detector_failures)} detector_failure)" if excluded else ""))
    print(f"False positives: {len(false_positives)}/{len(tested)}  "
          f"[this script's denominator: successfully-run non-matching images only]")

    # app.py's run_full_evaluation() never excludes anything: every
    # non-matching image gets bucketed into FP or TN via
    # _run_cached_detector(), which returns False (i.e. TN, not
    # excluded) on a segmentation_failure or detector_failure exactly
    # like it does on a genuine "not detected". So its FP denominator
    # for this detector is always every non-matching image, full stop.
    # Printed here as its own line - not folded into the count above -
    # so both counting rules stay visible and comparable rather than
    # silently picking one. They agree only when there are zero
    # failures; they will diverge the moment one occurs.
    print(f"App.py-equivalent denominator (run_full_evaluation counts every "
          f"non-matching image - including any failure - as FP or TN, "
          f"never excluded): {len(false_positives)}/{len(records)}")

    if detector_failures:
        print(f"Detector failures (excluded from this script's FP count/denominator above): {len(detector_failures)}")
        for r in detector_failures:
            print(f"   FAILED: {r['material']}/{r['defect_folder']}/{os.path.basename(r['image_path'])}"
                  f"  {r['error'].splitlines()[0]}")
    for r in false_positives:
        print(f"   FP: {r['material']}/{r['defect_folder']}/{os.path.basename(r['image_path'])}"
              f"  score={r['detection_score']:.2f}")


def write_csv(all_records):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fieldnames = [
        "detector", "material", "defect_folder", "image_path",
        "status", "detection_score", "detected", "false_positive", "error",
    ]
    with open(FP_SWEEP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nFull sweep log: {FP_SWEEP_CSV}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run a detector (or every implemented detector) against every "
                    "image outside its own folder and report false positives.",
    )
    parser.add_argument(
        "detector",
        help="A defect name from evaluate.DETECTOR_REGISTRY, or 'all' to sweep "
             "every currently-implemented detector.",
    )
    args = parser.parse_args()

    targets = resolve_targets(args.detector)
    all_images = list(discover_images())
    print(f"Total images discovered: {len(all_images)}")
    print(f"Detectors to sweep: {', '.join(targets)}")

    cache = build_image_cache(all_images)

    all_records = []
    for defect_name in targets:
        other_images = [(m, d, p) for (m, d, p) in all_images if d != defect_name]
        records = sweep_one_detector(defect_name, other_images, cache)
        all_records.extend(records)
        print_report(defect_name, records)

    write_csv(all_records)


if __name__ == "__main__":
    main()
