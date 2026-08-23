import os
import csv
import json
import time
import shutil
import argparse
import importlib
import statistics
import traceback

import cv2
import numpy as np

from preprocessing import load_image, preprocess_image
from segmentation import segment_glove

DATASET_ROOT = "datasets"
OUTPUT_ROOT = "outputs"
OVERLAY_DIR = os.path.join(OUTPUT_ROOT, "overlays")
MASK_DIR = os.path.join(OUTPUT_ROOT, "masks")
SUCCESS_DIR = os.path.join(OUTPUT_ROOT, "successful_cases")
FAILURE_DIR = os.path.join(OUTPUT_ROOT, "failure_cases")
SEG_FAILURE_DIR = os.path.join(FAILURE_DIR, "segmentation")
DET_FAILURE_DIR = os.path.join(FAILURE_DIR, "detector")
RESULTS_CSV = os.path.join(OUTPUT_ROOT, "results.csv")
SUMMARY_JSON = os.path.join(OUTPUT_ROOT, "summary.json")
DETECTION_RESULTS_JSON = os.path.join(OUTPUT_ROOT, "detection_results.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Below this glove-mask pixel count, segmentation is treated as failed.
MIN_GLOVE_AREA = 500

# Single cutoff for detected/not-detected, comparable across all 12 defects.
DETECTION_THRESHOLD = 0.5

# Magenta: doesn't occur naturally in glove colours or backgrounds.
BOX_COLOR = (255, 0, 255)   # BGR
BOX_THICKNESS = 3
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.6
LABEL_THICKNESS = 2
MASK_FILL_COLOR = (0, 255, 255)   # BGR, translucent fill over flagged pixels
MASK_FILL_ALPHA = 0.45

# Oversize-only visual measurement colours; do not affect any other defect.
OVERSIZE_PALM_COLOR = (255, 255, 0)   # BGR: cyan
OVERSIZE_CUFF_COLOR = (0, 165, 255)   # BGR: orange
OVERSIZE_LINE_THICKNESS = 3

# defect folder name -> "module.function"; missing/unimportable entries are skipped.
DETECTOR_REGISTRY = {
    "tearing": "detectors.tearing.detect_tearing",
    "tearing_fingertip": "detectors.tearing_fingertip.detect_tearing_fingertip",
    "finger_not_enough": "detectors.finger_not_enough.detect_finger_not_enough",

    "dirty": "detectors.dirty.detect_dirty",
    "stain": "detectors.stain.detect_stain",
    "spotting": "detectors.spotting.detect_spotting",

    "discoloration": "detectors.discoloration.detect_discoloration",
    "plastic_contamination": "detectors.plastic_contamination.detect_plastic_contamination",
    "oversize": "detectors.oversize.detect_oversize",

    "touching": "detectors.touching.detect_touching",
    "damaged_by_fold": "detectors.damaged_by_fold.detect_damaged_by_fold",
    "incomplete_beading": "detectors.incomplete_beading.detect_incomplete_beading",
}

REQUIRED_KEYS = {
    "defect_name": str,
    "detected": bool,
    "detection_score": (int, float),
    "algorithm": str,
    "bounding_box": (tuple, list, type(None)),
    "measurements": dict,
}

_detector_cache = {}


def load_detector(defect_name):
    """Import and return the detector function for a defect name; returns None (not a raise) if unregistered/unimportable, so the loop keeps going."""
    if defect_name in _detector_cache:
        return _detector_cache[defect_name]

    path = DETECTOR_REGISTRY.get(defect_name)
    if path is None:
        _detector_cache[defect_name] = None
        return None

    module_path, func_name = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
    except Exception:
        _detector_cache[defect_name] = None
        return None

    _detector_cache[defect_name] = func
    return func


def validate_result(result):
    """Validate a detector's return dict against the shared schema; raises ValueError with a specific message on mismatch."""
    if not isinstance(result, dict):
        raise ValueError("Detector must return a dict.")

    for key, expected_type in REQUIRED_KEYS.items():
        if key not in result:
            raise ValueError(f"Detector result missing required key: '{key}'")
        if not isinstance(result[key], expected_type):
            raise ValueError(
                f"Detector result key '{key}' has wrong type: "
                f"expected {expected_type}, got {type(result[key])}"
            )

    score = result["detection_score"]
    if not (0.0 <= float(score) <= 1.0):
        raise ValueError(f"detection_score must be between 0.0 and 1.0, got {score}")

    return True


def _bbox_from_mask(mask):
    """Derive a bounding box (x, y, w, h) from a binary defect mask."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
    return (x, y, w, h)


def build_segmented_view(original_bgr, glove_mask):
    """Dim everything outside glove_mask to greyscale so overlays draw the eye to the glove, not the background."""
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    dimmed_background = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 0.35).astype(np.uint8)
    mask_bool = glove_mask > 0
    segmented = np.where(mask_bool[:, :, None], original_bgr, dimmed_background)
    return segmented.astype(np.uint8)


def draw_overlay(base_bgr, defect_name, detected, score, bounding_box, defect_mask=None):
    """Draw the translucent defect-mask fill, bounding box, and label onto a copy of the base image."""
    overlay = base_bgr.copy()

    if defect_mask is not None and np.any(defect_mask):
        fill = overlay.copy()
        fill[defect_mask > 0] = MASK_FILL_COLOR
        overlay = cv2.addWeighted(fill, MASK_FILL_ALPHA, overlay, 1 - MASK_FILL_ALPHA, 0)

    if bounding_box is not None:
        x, y, w, h = bounding_box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

    label = f"{defect_name}: {'DETECTED' if detected else 'NOT DETECTED'} ({score:.1%})"
    text_origin = (10, 25)
    cv2.putText(overlay, label, text_origin, LABEL_FONT, LABEL_SCALE,
                (0, 0, 0), LABEL_THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(overlay, label, text_origin, LABEL_FONT, LABEL_SCALE,
                BOX_COLOR, LABEL_THICKNESS, cv2.LINE_AA)

    return overlay


def draw_oversize_overlay(
    base_bgr,
    detected,
    score,
    bounding_box,
    evidence_mask=None,
    measurements=None,
):
    """Oversize-only visual overlay (yellow=evidence, cyan=palm line, orange=cuff line, magenta=box); explanatory text lives in measurements['prototype_details'], not on the image."""
    overlay = base_bgr.copy()
    measurements = measurements or {}

    if evidence_mask is not None and np.any(evidence_mask):
        fill = overlay.copy()
        fill[evidence_mask > 0] = MASK_FILL_COLOR
        overlay = cv2.addWeighted(
            fill,
            MASK_FILL_ALPHA,
            overlay,
            1 - MASK_FILL_ALPHA,
            0,
        )

    if bounding_box is not None:
        x, y, w, h = [int(v) for v in bounding_box]
        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            BOX_COLOR,
            BOX_THICKNESS,
        )

    palm_line = measurements.get("palm_measurement_line")
    if palm_line is not None and len(palm_line) == 4:
        x1, y1, x2, y2 = [int(v) for v in palm_line]
        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            OVERSIZE_PALM_COLOR,
            OVERSIZE_LINE_THICKNESS,
            cv2.LINE_AA,
        )

    cuff_line = measurements.get("cuff_measurement_line")
    if cuff_line is not None and len(cuff_line) == 4:
        x1, y1, x2, y2 = [int(v) for v in cuff_line]
        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            OVERSIZE_CUFF_COLOR,
            OVERSIZE_LINE_THICKNESS,
            cv2.LINE_AA,
        )

    return overlay


def evaluate_image(image_path, defect_name):
    """Run preprocessing -> segmentation -> the matching detector on one image; status is one of success/segmentation_failure/detector_failure/not_implemented."""
    record = {
        "image_path": image_path,
        "defect_name": defect_name,
        "status": None,
        "detected": False,
        "detection_score": 0.0,
        "algorithm": None,
        "bounding_box": None,
        "measurements": {},
        "glove_area": None,
        "processing_time_ms": None,
        "overlay_path": None,
        "mask_path": None,
        "error": None,
    }

    start = time.time()

    try:
        image = load_image(image_path)
        processed = preprocess_image(image)
        segmentation = segment_glove(processed)
    except Exception as exc:
        record["status"] = "segmentation_failure"
        record["error"] = f"preprocessing/segmentation crashed: {exc}"
        record["processing_time_ms"] = (time.time() - start) * 1000
        return record, None, None

    record["glove_area"] = segmentation["glove_area"]
    segmented_view = build_segmented_view(processed["original"], segmentation["glove_mask"])

    if segmentation["glove_area"] < MIN_GLOVE_AREA:
        record["status"] = "segmentation_failure"
        record["error"] = "glove mask area below MIN_GLOVE_AREA threshold"
        record["processing_time_ms"] = (time.time() - start) * 1000
        return record, segmented_view, None

    detector_func = load_detector(defect_name)
    if detector_func is None:
        record["status"] = "not_implemented"
        record["processing_time_ms"] = (time.time() - start) * 1000
        return record, segmented_view, None

    try:
        result = detector_func(processed, segmentation)
        validate_result(result)
    except Exception as exc:
        record["status"] = "detector_failure"
        record["error"] = f"{exc}\n{traceback.format_exc(limit=2)}"
        record["processing_time_ms"] = (time.time() - start) * 1000
        return record, segmented_view, None

    defect_mask = result.get("mask")
    bounding_box = result.get("bounding_box") or _bbox_from_mask(defect_mask)

    record["status"] = "success"
    record["detected"] = bool(result["detected"]) and result["detection_score"] >= DETECTION_THRESHOLD
    record["detection_score"] = float(result["detection_score"])
    record["algorithm"] = result["algorithm"]
    record["bounding_box"] = bounding_box
    record["measurements"] = result.get("measurements", {})
    record["processing_time_ms"] = (time.time() - start) * 1000

    return record, segmented_view, defect_mask


def discover_images(defect_filter=None, material_filter=None):
    """Yield (material, defect, image_path) for every dataset image whose parent folder matches a registered defect, optionally filtered."""
    for material in sorted(os.listdir(DATASET_ROOT)):
        if material_filter and material.lower() != material_filter.lower():
            continue

        material_dir = os.path.join(DATASET_ROOT, material)
        if not os.path.isdir(material_dir):
            continue

        for defect_name in sorted(os.listdir(material_dir)):
            if defect_filter and defect_name.lower() != defect_filter.lower():
                continue

            defect_dir = os.path.join(material_dir, defect_name)
            if not os.path.isdir(defect_dir):
                continue

            for filename in sorted(os.listdir(defect_dir)):
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                    yield material, defect_name, os.path.join(defect_dir, filename)


def run_evaluation(defect_filter=None, material_filter=None):
    for folder in (OVERLAY_DIR, MASK_DIR, SUCCESS_DIR, SEG_FAILURE_DIR, DET_FAILURE_DIR):
        os.makedirs(folder, exist_ok=True)

    all_records = []

    for material, defect_name, image_path in discover_images(defect_filter, material_filter):
        record, segmented_view, defect_mask = evaluate_image(image_path, defect_name)
        record["material"] = material
        all_records.append(record)

        base_name = f"{material}_{defect_name}_{os.path.splitext(os.path.basename(image_path))[0]}"

        if segmented_view is not None:
            # Oversize needs its own visual style; other defects use draw_overlay().
            if defect_name == "oversize":
                overlay = draw_oversize_overlay(
                    segmented_view,
                    record["detected"],
                    record["detection_score"],
                    record["bounding_box"],
                    defect_mask,
                    record.get("measurements", {}),
                )
            else:
                overlay = draw_overlay(
                    segmented_view,
                    defect_name,
                    record["detected"],
                    record["detection_score"],
                    record["bounding_box"],
                    defect_mask,
                )

            overlay_path = os.path.join(OVERLAY_DIR, f"{base_name}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)
            record["overlay_path"] = overlay_path

            if defect_mask is not None and np.any(defect_mask):
                mask_path = os.path.join(MASK_DIR, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, defect_mask)
                record["mask_path"] = mask_path

            if record["status"] == "success" and record["detected"]:
                shutil.copy(overlay_path, os.path.join(SUCCESS_DIR, f"{base_name}_overlay.jpg"))
            elif record["status"] == "segmentation_failure":
                shutil.copy(overlay_path, os.path.join(SEG_FAILURE_DIR, f"{base_name}_overlay.jpg"))
            elif record["status"] in ("detector_failure", "success"):
                # "success" but not detected = a missed detection -> detector failure bucket
                shutil.copy(overlay_path, os.path.join(DET_FAILURE_DIR, f"{base_name}_overlay.jpg"))

        status_flag = "OK" if record["status"] == "success" else record["status"].upper()
        print(f"[{status_flag}] {material}/{defect_name}: {os.path.basename(image_path)} "
              f"(score={record['detection_score']:.2f})")

    write_results_csv(all_records)
    write_detection_results_json(all_records)

    summary = build_summary(all_records)
    write_summary_json(summary)
    print_summary(summary)


def _json_safe(value):
    """Convert NumPy/OpenCV/Python values into JSON-safe types, so detectors don't need to know about JSON themselves."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item)
            for item in value
        ]

    return str(value)


def write_detection_results_json(records):
    """Write full per-image records (including each detector's own measurements) to detection_results.json for result inspection."""
    payload = {
        "format_version": 1,
        "record_count": len(records),
        "results": [
            _json_safe(record)
            for record in records
        ],
    }

    with open(
        DETECTION_RESULTS_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_results_csv(records):
    fieldnames = [
        "material", "defect_name", "image_path", "status", "detected",
        "detection_score", "algorithm", "bounding_box", "glove_area",
        "defect_area_pct", "processing_time_ms", "error",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in fieldnames}
            # defect_area_pct comes from measurements, not a top-level field.
            row["defect_area_pct"] = r.get("measurements", {}).get("area_pct")
            writer.writerow(row)


def build_summary(records):
    """Compute detection rate and other report metrics, overall / per-defect / per-material."""
    summary = {"overall": {}, "by_defect": {}, "by_material": {}}

    def _stats_block(recs):
        total = len(recs)
        attempted = [r for r in recs if r["status"] in ("success", "detector_failure")]
        successful_runs = [r for r in recs if r["status"] == "success"]
        detected = [r for r in successful_runs if r["detected"]]
        seg_failures = [r for r in recs if r["status"] == "segmentation_failure"]
        det_failures = [r for r in recs if r["status"] == "detector_failure"]
        not_impl = [r for r in recs if r["status"] == "not_implemented"]

        scores_all = [r["detection_score"] for r in successful_runs]
        scores_detected = [r["detection_score"] for r in detected]
        times = [r["processing_time_ms"] for r in recs if r["processing_time_ms"] is not None]
        areas = [
            r["measurements"]["area_pct"] for r in detected
            if r.get("measurements", {}).get("area_pct") is not None
        ]

        return {
            "total_images": total,
            "detected": len(detected),
            "detection_rate_pct": round(100 * len(detected) / total, 1) if total else None,
            "segmentation_failures": len(seg_failures),
            "segmentation_failure_rate_pct": round(100 * len(seg_failures) / total, 1) if total else None,
            "detector_failures": len(det_failures),
            "detector_failure_rate_pct": round(100 * len(det_failures) / total, 1) if total else None,
            "not_implemented": len(not_impl),
            "mean_detection_score": round(statistics.mean(scores_all), 3) if scores_all else None,
            "stdev_detection_score": round(statistics.pstdev(scores_all), 3) if len(scores_all) > 1 else None,
            "mean_score_when_detected": round(statistics.mean(scores_detected), 3) if scores_detected else None,
            "mean_processing_time_ms": round(statistics.mean(times), 1) if times else None,
            "mean_defect_area_pct": round(statistics.mean(areas), 2) if areas else None,
            "algorithm": next((r["algorithm"] for r in successful_runs if r["algorithm"]), None),
        }

    summary["overall"] = _stats_block(records)

    defects = sorted(set(r["defect_name"] for r in records))
    for defect in defects:
        summary["by_defect"][defect] = _stats_block([r for r in records if r["defect_name"] == defect])

    materials = sorted(set(r["material"] for r in records))
    for material in materials:
        summary["by_material"][material] = _stats_block([r for r in records if r["material"] == material])

    return summary


def write_summary_json(summary):
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def print_summary(summary):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall detection rate: {summary['overall']['detection_rate_pct']}% "
          f"({summary['overall']['detected']}/{summary['overall']['total_images']})")
    print("-" * 60)
    for defect, stats in summary["by_defect"].items():
        print(f"{defect:24s} "
              f"rate={stats['detection_rate_pct']}%  "
              f"mean_score={stats['mean_detection_score']}  "
              f"seg_fail={stats['segmentation_failures']}  "
              f"det_fail={stats['detector_failures']}  "
              f"algo={stats['algorithm']}")
    print("=" * 60)
    print(f"Full log: {RESULTS_CSV}")
    print(f"Per-image JSON: {DETECTION_RESULTS_JSON}")
    print(f"Summary:  {SUMMARY_JSON}")
    print(f"Overlays: {OVERLAY_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the glove defect detection evaluation pipeline."
    )
    parser.add_argument(
        "--defect", "-d", default=None,
        help="Only evaluate this defect type, e.g. --defect discoloration",
    )
    parser.add_argument(
        "--material", "-m", default=None,
        help="Only evaluate this material, e.g. --material nitrile",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(defect_filter=args.defect, material_filter=args.material)
