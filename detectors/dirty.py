"""
detectors/dirty.py
------------------
Dirty: diffuse soiling spread across part of the glove surface.

Author: Ka Sin (M2)

What dirty looks like in this dataset
-------------------------------------
On knit cotton the soiling is granular - dust and grime caught in the
weave, which thresholds into many small dark fragments. On latex it is
a smoother grey-brown smear. The two look very different at the level
of individual regions, which is why this detector does not classify
regions one at a time.

What they share, and what no other defect in the set shares, is that
the glove *between* the dark fragments is itself darkened. Soiling has
a halo: it fades outward instead of stopping at a hard edge. Spotting,
by contrast, is discrete marks on clean glove.

So the decisive feature here is `ring_dark` from
_anomaly.cloud_statistics - the average darkening of the glove
immediately around the detected regions, with the regions and their
neighbours excluded - together with its inverse, the `isolation` ratio.
Dirty is the low-isolation case; spotting and stain are the high ones.

Technique
---------
Illumination-normalised LAB anomaly extraction (shared front-end)
-> connected-component analysis
-> soiling-halo measurement on the glove around each region
-> weighted evidence over coverage, halo darkening and low isolation.

Thresholds
----------
Chosen by inspecting feature distributions on the group's own dataset.
They are not learned, and they were tuned on the same images used for
testing, which makes the reported detection rate optimistic. This is
stated as a limitation in the report.
"""

import numpy as np

from . import _anomaly


# Dirt covers area. A handful of tiny specks is spotting, not soiling.
MIN_FRAGMENT_AREA_FRAC = 0.00015

# Evidence ramps: (low, high) - low scores 0, high scores 1.
COVERAGE_RAMP = (0.030, 0.130)
HALO_RAMP = (0.018, 0.045)
DIFFUSENESS_RAMP = (7.0, 3.2)       # reversed: lower isolation is dirtier
CLUSTER_RAMP = (0.25, 0.45)         # how densely the fragments pack
DEPTH_RAMP = (0.08, 0.20)

DECISION_THRESHOLD = 0.50


def _cluster_fill(defect_mask, equivalent_radius):
    """
    How densely the detected fragments pack into the area they occupy.

    The mask is dilated so that neighbouring fragments merge into one
    envelope, and the result is the fraction of that envelope which is
    actually flagged. Granular dirt packs tightly and fills a large part
    of its envelope; scattered spots leave most of theirs empty.
    """
    import cv2

    if defect_mask is None or not defect_mask.any():
        return 0.0
    radius = max(3, int(0.04 * equivalent_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    envelope = cv2.dilate(defect_mask, kernel)
    envelope_area = max(1, int(np.count_nonzero(envelope)))
    return float(np.count_nonzero(defect_mask) / envelope_area)


def detect_dirty(processed: dict, segmentation: dict) -> dict:
    """
    Detect diffuse soiling on a segmented glove.

    Parameters
    ----------
    processed : dict
        Output of preprocessing.preprocess_image().
    segmentation : dict
        Output of segmentation.segment_glove().

    Returns
    -------
    dict
        Result dictionary in the shared detector contract format.
    """
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

    # Coverage and the soiling halo carry the most weight: soiling is
    # defined by covering area and by dulling the surface around it.
    evidence = _anomaly.combine([
        (0.30, _anomaly.ramp(stats["area_frac"], *COVERAGE_RAMP)),
        (0.25, _anomaly.ramp(stats["ring_dark"], *HALO_RAMP)),
        (0.20, _anomaly.ramp(stats["isolation"], *DIFFUSENESS_RAMP)),
        (0.15, _anomaly.ramp(fill, *CLUSTER_RAMP)),
        (0.10, _anomaly.ramp(stats["mean_dark"], *DEPTH_RAMP)),
    ])

    # Shadow rejection. A bunched or folded glove produces exactly the
    # shape signature this detector looks for - broad, low-contrast,
    # diffuse darkening with a soft halo - so the geometric evidence
    # alone cannot tell soiling from a crease. Requiring colour
    # evidence is what makes the difference.
    confidence = _anomaly.chromatic_confidence(stats)
    score = evidence * confidence

    result["detected"] = bool(score >= DECISION_THRESHOLD)
    result["detection_score"] = round(float(score), 4)
    result["bounding_box"] = _anomaly.union_bbox(fragments)
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
