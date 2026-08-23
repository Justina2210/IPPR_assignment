import numpy as np

from . import _anomaly

# Minimum fragment coverage considered soiling.
MIN_FRAGMENT_AREA_FRAC = 0.00015

# Coverage score ramp.
COVERAGE_RAMP = (0.030, 0.130)
# Surrounding-darkening score ramp.
HALO_RAMP = (0.018, 0.045)
# Isolation score ramp; lower isolation indicates diffuse soiling.
DIFFUSENESS_RAMP = (7.0, 3.2)       # reversed: lower isolation is dirtier
# Cluster-fill score ramp.
CLUSTER_RAMP = (0.25, 0.45)
# Mean-darkening score ramp.
DEPTH_RAMP = (0.08, 0.20)

# Minimum combined score for detection.
DECISION_THRESHOLD = 0.50


def _cluster_fill(defect_mask, equivalent_radius):
    """Fraction of the dilated fragment envelope actually occupied by fragments."""
    import cv2

    if defect_mask is None or not defect_mask.any():
        return 0.0
    radius = max(3, int(0.04 * equivalent_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    envelope = cv2.dilate(defect_mask, kernel)
    envelope_area = max(1, int(np.count_nonzero(envelope)))
    return float(np.count_nonzero(defect_mask) / envelope_area)


def detect_dirty(processed: dict, segmentation: dict) -> dict:
    """Detect diffuse soiling on a segmented glove."""
    result = {
        "defect_name": "dirty",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": "Illumination-normalised LAB anomaly + connected-component "
                     "analysis + soiling-halo (surround darkening) measurement",
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }

    fields, candidate, blobs, interior = _anomaly.prepare(processed, segmentation)
    if fields is None:
        result["measurements"] = {"note": "glove interior too small to analyse"}
        return result

    surrounded = _anomaly.accept_blobs(blobs)
    fragments = [b for b in surrounded if b["area_frac"] >= MIN_FRAGMENT_AREA_FRAC]

    dirt_mask = _anomaly.blobs_to_mask(fragments, candidate, interior.shape)
    stats = _anomaly.cloud_statistics(fragments, dirt_mask, fields, interior)

    if stats["count"] == 0:
        result["measurements"] = {"fragment_count": 0, "area_pct": 0.0}
        return result

    fill = _cluster_fill(dirt_mask, stats["equivalent_radius"])

    # Coverage and the soiling halo carry the most weight - soiling is defined by area covered and by dulling the surrounding surface.
    evidence = _anomaly.combine([
        (0.30, _anomaly.ramp(stats["area_frac"], *COVERAGE_RAMP)),
        (0.25, _anomaly.ramp(stats["ring_dark"], *HALO_RAMP)),
        (0.20, _anomaly.ramp(stats["isolation"], *DIFFUSENESS_RAMP)),
        (0.15, _anomaly.ramp(fill, *CLUSTER_RAMP)),
        (0.10, _anomaly.ramp(stats["mean_dark"], *DEPTH_RAMP)),
    ])

    # Shadow rejection: a bunched/folded glove produces the same broad, low-contrast, diffuse-halo signature, so shape evidence alone can't tell soiling from a crease.
    confidence = _anomaly.chromatic_confidence(stats)
    score = evidence * confidence

    result["detected"] = bool(score >= DECISION_THRESHOLD)
    result["detection_score"] = round(float(score), 4)
    result["bounding_box"] = _anomaly.dominant_bbox(
        _anomaly.localisation_blobs(fragments), stats["equivalent_radius"])
    result["mask"] = dirt_mask
    result["measurements"] = {
        "fragment_count": int(stats["count"]),
        "area_pct": round(100.0 * stats["area_frac"], 3),
        "surrounding_darkening": round(stats["ring_dark"], 4),
        "isolation_ratio": round(stats["isolation"], 2),
        "cluster_fill": round(fill, 3),
        "mean_darkening": round(stats["mean_dark"], 4),
        "median_fragment_area_px": round(stats["median_area"], 1),
        "chromatic_fraction": round(stats["chromatic_fraction"], 3),
        "shape_evidence": round(float(evidence), 4),
        "chromatic_confidence": round(float(confidence), 3),
    }
    return result
