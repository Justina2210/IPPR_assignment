"""
detectors/spotting.py
---------------------
Spotting: many small, discrete specks scattered over the glove.

Author: Ka Sin (M2)

What spotting looks like in this dataset
----------------------------------------
On pale latex the specks are rust/orange coloured; on blue nitrile they
are near-black. In both cases the individual marks are small (a few
dozen pixels at the working resolution of 1000 px on the long side),
close to circular, and separated from each other by clean glove.

That last property is what makes spotting different from dirty, which
otherwise produces a similar count of small dark regions on knit cotton.
Dirt is a smear: the glove *between* the specks is also soiled. Spots
are discrete: the glove between them is clean. The `isolation` statistic
in _anomaly.cloud_statistics measures exactly this, and it is the single
feature that keeps the two apart.

Technique
---------
Illumination-normalised LAB anomaly extraction (shared front-end)
-> connected-component analysis
-> shape filtering on circularity, elongation and size
-> weighted evidence score over count, shape, size and isolation.

Thresholds
----------
Every constant below was chosen by inspecting the feature distributions
of the group's own 68-image dataset. They are not learned, and they were
tuned on the same images the system is tested on, so the reported
detection rate is optimistic. This is recorded as a limitation in the
report rather than hidden.
"""

import numpy as np

from . import _anomaly


# Individual specks are small. A region larger than this fraction of the
# glove is a smear or a stain, not a spot.
MAX_SPECK_AREA_FRAC = 0.010

# A speck is roughly round. These reject wrinkle fragments and streaks.
MIN_SPECK_CIRCULARITY = 0.55
MAX_SPECK_ELONGATION = 2.2

# Evidence ramps: (low, high) - low scores 0, high scores 1.
COUNT_RAMP = (12.0, 55.0)
CIRCULARITY_RAMP = (0.70, 0.95)
ISOLATION_RAMP = (4.0, 9.0)
COVERAGE_RAMP = (0.010, 0.060)
SMALLNESS_RAMP = (200.0, 60.0)      # reversed: smaller median area is better

DECISION_THRESHOLD = 0.50


def detect_spotting(processed: dict, segmentation: dict) -> dict:
    """
    Detect scattered spotting on a segmented glove.

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

    # Keep only regions surrounded by glove, then only those shaped
    # like a speck.
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

    # Weighted evidence. Count and isolation carry the most weight
    # because they are what distinguish spotting from a dirt smear;
    # shape and size confirm that the regions really are specks.
    evidence = _anomaly.combine([
        (0.30, _anomaly.ramp(stats["count"], *COUNT_RAMP)),
        (0.25, _anomaly.ramp(stats["isolation"], *ISOLATION_RAMP)),
        (0.20, _anomaly.ramp(stats["circularity"], *CIRCULARITY_RAMP)),
        (0.15, _anomaly.ramp(stats["median_area"], *SMALLNESS_RAMP)),
        (0.10, _anomaly.ramp(stats["area_frac"], *COVERAGE_RAMP)),
    ])

    # The same shadow rejection the other two detectors use. The shape
    # filters above already make this detector hard to fool, so the
    # gate changes little here - but applying it keeps all three
    # detectors on one consistent rule.
    confidence = _anomaly.chromatic_confidence(stats)
    score = evidence * confidence

    result["detected"] = bool(score >= DECISION_THRESHOLD)
    result["detection_score"] = round(float(score), 4)
    result["bounding_box"] = _anomaly.union_bbox(specks)
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
