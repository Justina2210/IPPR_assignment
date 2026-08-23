import numpy as np

from . import _anomaly

# Maximum region coverage accepted for a speck.
MAX_SPECK_AREA_FRAC = 0.010

# Minimum circularity accepted for a speck.
MIN_SPECK_CIRCULARITY = 0.55
# Maximum elongation accepted for a speck.
MAX_SPECK_ELONGATION = 2.2

# Count score ramp.
COUNT_RAMP = (12.0, 55.0)
# Circularity score ramp.
CIRCULARITY_RAMP = (0.70, 0.95)
# Isolation score ramp.
ISOLATION_RAMP = (4.0, 9.0)
# Coverage score ramp.
COVERAGE_RAMP = (0.010, 0.060)
# Median-area score ramp; smaller specks score higher.
SMALLNESS_RAMP = (200.0, 60.0)      # reversed: smaller median area is better

# Minimum combined score for detection.
DECISION_THRESHOLD = 0.50


def detect_spotting(processed: dict, segmentation: dict) -> dict:
    """Detect scattered spotting on a segmented glove."""
    result = {
        "defect_name": "spotting",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": "Illumination-normalised LAB anomaly + connected-component "
                     "shape filtering (circularity/size) + speck isolation ratio",
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }

    fields, candidate, blobs, interior = _anomaly.prepare(processed, segmentation)
    if fields is None:
        result["measurements"] = {"note": "glove interior too small to analyse"}
        return result

    surrounded = _anomaly.accept_blobs(blobs)
    specks = [
        b for b in surrounded
        if b["area_frac"] <= MAX_SPECK_AREA_FRAC
        and b["circularity"] >= MIN_SPECK_CIRCULARITY
        and b["elongation"] <= MAX_SPECK_ELONGATION
    ]

    speck_mask = _anomaly.blobs_to_mask(specks, candidate, interior.shape)
    stats = _anomaly.cloud_statistics(specks, speck_mask, fields, interior)

    if stats["count"] == 0:
        result["measurements"] = {"speck_count": 0, "area_pct": 0.0}
        return result

    # Count and isolation carry the most weight - they're what distinguish spotting from a dirt smear; shape/size just confirm the regions are specks.
    evidence = _anomaly.combine([
        (0.30, _anomaly.ramp(stats["count"], *COUNT_RAMP)),
        (0.25, _anomaly.ramp(stats["isolation"], *ISOLATION_RAMP)),
        (0.20, _anomaly.ramp(stats["circularity"], *CIRCULARITY_RAMP)),
        (0.15, _anomaly.ramp(stats["median_area"], *SMALLNESS_RAMP)),
        (0.10, _anomaly.ramp(stats["area_frac"], *COVERAGE_RAMP)),
    ])

    # Apply the shared colour gate for consistent shadow rejection.
    confidence = _anomaly.chromatic_confidence(stats)
    score = evidence * confidence

    result["detected"] = bool(score >= DECISION_THRESHOLD)
    result["detection_score"] = round(float(score), 4)
    result["bounding_box"] = _anomaly.dominant_bbox(
        _anomaly.localisation_blobs(specks), stats["equivalent_radius"])
    result["mask"] = speck_mask
    result["measurements"] = {
        "speck_count": int(stats["count"]),
        "area_pct": round(100.0 * stats["area_frac"], 3),
        "median_speck_area_px": round(stats["median_area"], 1),
        "median_circularity": round(stats["circularity"], 3),
        "median_elongation": round(stats["elongation"], 3),
        "isolation_ratio": round(stats["isolation"], 2),
        "surrounding_darkening": round(stats["ring_dark"], 4),
        "dispersion": round(stats["dispersion"], 3),
        "chromatic_fraction": round(stats["chromatic_fraction"], 3),
        "shape_evidence": round(float(evidence), 4),
        "chromatic_confidence": round(float(confidence), 3),
    }
    return result
