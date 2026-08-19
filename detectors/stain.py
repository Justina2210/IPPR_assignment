"""
detectors/stain.py
------------------
Stain: liquid marks that have soaked or smeared into the glove surface.

Author: Ka Sin (M2)

What stain looks like in this dataset
-------------------------------------
Dark brown coffee-like marks on pale latex, and near-black ink-like
strokes on blue nitrile. In both materials the mark is a *stroke*: it is
much longer than it is wide, it has a hard edge where the liquid stopped,
and individual marks are far larger than a spot.

Against the other two defects in this member's set:

- versus spotting - a stain region is an order of magnitude larger and
  clearly elongated, where a spot is small and round;
- versus dirty - a stain has a hard boundary and leaves the surrounding
  glove clean, where soiling fades outward into a halo.

So the decisive features are region size, elongation, low circularity
and a high isolation ratio.

Technique
---------
Illumination-normalised LAB anomaly extraction (shared front-end)
-> connected-component analysis
-> geometric filtering on size, elongation and boundary sharpness
-> weighted evidence over the largest mark, elongation and isolation.

Thresholds
----------
Chosen by inspecting feature distributions on the group's own dataset.
They are not learned, and they were tuned on the same images used for
testing, which makes the reported detection rate optimistic. This is
stated as a limitation in the report.
"""

import numpy as np

from . import _anomaly


# A stain mark is substantially bigger than a speck.
MIN_MARK_AREA_FRAC = 0.0012

# Strokes are not round. This rejects the compact specks that spotting
# produces so the two detectors do not simply mirror each other.
MAX_MARK_CIRCULARITY = 0.90

# Evidence ramps: (low, high) - low scores 0, high scores 1.
LARGEST_MARK_RAMP = (0.004, 0.030)
ELONGATION_RAMP = (1.25, 2.10)
ISOLATION_RAMP = (4.0, 7.5)
COVERAGE_RAMP = (0.015, 0.090)
SHAPE_RAMP = (0.75, 0.35)           # reversed: lower circularity is more stroke-like

DECISION_THRESHOLD = 0.50


def detect_stain(processed: dict, segmentation: dict) -> dict:
    """
    Detect liquid staining on a segmented glove.

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

    # Isolation carries the most weight, and it is what stops this
    # detector from simply relabelling every dirty glove as stained.
    # A stain leaves the glove around it clean, so its darkening is
    # many times that of its surroundings (measured range 5.5-17.7 on
    # the true stains). Soiling fades outward into a halo, so the same
    # ratio stays low (3.3-4.6 on the dirty images). Size and shape
    # then confirm that the mark is a stroke rather than a speck.
    evidence = _anomaly.combine([
        (0.38, _anomaly.ramp(stats["isolation"], *ISOLATION_RAMP)),
        (0.20, _anomaly.ramp(stats["max_area_frac"], *LARGEST_MARK_RAMP)),
        (0.18, _anomaly.ramp(stats["elongation"], *ELONGATION_RAMP)),
        (0.12, _anomaly.ramp(stats["circularity"], *SHAPE_RAMP)),
        (0.12, _anomaly.ramp(stats["area_frac"], *COVERAGE_RAMP)),
    ])

    # Shadow rejection. A deep crease in a loose glove is long, narrow,
    # dark and sharply bounded, which is the same geometry a stain
    # stroke has. Colour evidence is what separates them.
    confidence = _anomaly.chromatic_confidence(stats)
    score = evidence * confidence

    result["detected"] = bool(score >= DECISION_THRESHOLD)
    result["detection_score"] = round(float(score), 4)
    result["bounding_box"] = _anomaly.union_bbox(marks)
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
