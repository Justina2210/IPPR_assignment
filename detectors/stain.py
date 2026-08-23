import numpy as np

from . import _anomaly

# Minimum region coverage considered a stain mark.
MIN_MARK_AREA_FRAC = 0.0012

# Maximum circularity accepted for a stroke-like mark.
MAX_MARK_CIRCULARITY = 0.90

# Largest-mark score ramp.
LARGEST_MARK_RAMP = (0.004, 0.030)
# Elongation score ramp.
ELONGATION_RAMP = (1.25, 2.10)
# Isolation score ramp.
ISOLATION_RAMP = (4.0, 7.5)
# Coverage score ramp.
COVERAGE_RAMP = (0.015, 0.090)
# Circularity score ramp; lower circularity indicates a stroke.
SHAPE_RAMP = (0.75, 0.35)           # reversed: lower circularity is more stroke-like

# Minimum combined score for detection.
DECISION_THRESHOLD = 0.50


def detect_stain(processed: dict, segmentation: dict) -> dict:
    """Detect liquid staining on a segmented glove."""
    result = {
        "defect_name": "stain",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": "Illumination-normalised LAB anomaly + connected-component "
                     "geometry (area/elongation/circularity) + edge-sharpness test",
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }

    fields, candidate, blobs, interior = _anomaly.prepare(processed, segmentation)
    if fields is None:
        result["measurements"] = {"note": "glove interior too small to analyse"}
        return result

    surrounded = _anomaly.accept_blobs(blobs)
    marks = [
        b for b in surrounded
        if b["area_frac"] >= MIN_MARK_AREA_FRAC
        and b["circularity"] <= MAX_MARK_CIRCULARITY
    ]

    stain_mask = _anomaly.blobs_to_mask(marks, candidate, interior.shape)
    stats = _anomaly.cloud_statistics(marks, stain_mask, fields, interior)

    if stats["count"] == 0:
        result["measurements"] = {"mark_count": 0, "area_pct": 0.0}
        return result

    # Isolation carries the most weight: a stain's darkening is 5.5-17.7x its surroundings vs 3.3-4.6x for soiling's halo; size/shape then confirm a stroke, not a speck.
    evidence = _anomaly.combine([
        (0.38, _anomaly.ramp(stats["isolation"], *ISOLATION_RAMP)),
        (0.20, _anomaly.ramp(stats["max_area_frac"], *LARGEST_MARK_RAMP)),
        (0.18, _anomaly.ramp(stats["elongation"], *ELONGATION_RAMP)),
        (0.12, _anomaly.ramp(stats["circularity"], *SHAPE_RAMP)),
        (0.12, _anomaly.ramp(stats["area_frac"], *COVERAGE_RAMP)),
    ])

    # Shadow rejection: a deep crease in a loose glove is long, narrow, dark and sharply bounded - the same geometry as a stain stroke.
    confidence = _anomaly.chromatic_confidence(stats)
    score = evidence * confidence

    result["detected"] = bool(score >= DECISION_THRESHOLD)
    result["detection_score"] = round(float(score), 4)
    result["bounding_box"] = _anomaly.dominant_bbox(
        _anomaly.localisation_blobs(marks), stats["equivalent_radius"])
    result["mask"] = stain_mask
    result["measurements"] = {
        "mark_count": int(stats["count"]),
        "area_pct": round(100.0 * stats["area_frac"], 3),
        "largest_mark_pct": round(100.0 * stats["max_area_frac"], 3),
        "median_mark_area_px": round(stats["median_area"], 1),
        "median_elongation": round(stats["elongation"], 3),
        "median_circularity": round(stats["circularity"], 3),
        "isolation_ratio": round(stats["isolation"], 2),
        "edge_sharpness": round(stats["edge_sharp"], 3),
        "chromatic_fraction": round(stats["chromatic_fraction"], 3),
        "shape_evidence": round(float(evidence), 4),
        "chromatic_confidence": round(float(confidence), 3),
    }
    return result
