"""
_template.py
------------
Skeleton for a new detector. Copy this file to detectors/<defect_name>.py,
rename detect_xxx() to match your defect, fill in the TODOs, and add one
line to DETECTOR_REGISTRY in evaluate.py:

    "<defect_name>": "detectors.<defect_name>.detect_xxx"

That's the only wiring needed - evaluate.py, app.py's single-defect mode,
and the accuracy evaluation all pick it up automatically from the
registry.

THE 3 HARD RULES (per the assignment brief - every detector must obey
these):
  1. Classical image processing ONLY. No Haar cascades, no
     TensorFlow/Keras/PyTorch/sklearn, no cv2.matchTemplate or other
     trained/pattern-matching models. OpenCV + numpy primitives only.
  2. All analysis happens INSIDE the glove. Always gate on
     segmentation["glove_mask"] - never look at background pixels.
  3. Any area percentage in "measurements" MUST use
     segmentation["glove_area"] as the denominator - never
     image.shape / the whole-frame pixel count.
"""

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================
# Every threshold/weight below must be a named, commented constant (no
# bare magic numbers inline in the detection logic). If a constant was
# arrived at by looking at how it performed on the dataset rather than
# derived from a physical/geometric argument, tag it:
#     # TUNED-BY-EYE on the 68-image dataset
# so it's clear in the report which numbers are principled vs empirical.

# TODO: replace with your own thresholds, e.g.
# MIN_COLOUR_DISTANCE = 18   # LAB a/b distance considered anomalous
# DETECTION_SCORE_THRESHOLD = 0.5   # detected/not-detected cutoff on the 0-1 score

ALGORITHM = "TODO: short human-readable description of the technique used"


def _empty_result():
    """Return value for 'nothing found / could not run' - keeps every
    early-return in detect_xxx() consistent with the full-result shape
    below, so callers never have to special-case a missing key."""
    return {
        "defect_name": "TODO_defect_name",   # str, must match the datasets/ folder name
        "detected": False,                    # bool
        "detection_score": 0.0,               # float, must stay within 0.0-1.0
        "algorithm": ALGORITHM,               # str
        "bounding_box": None,                 # (x, y, w, h) tuple, or None
        "mask": None,                         # np.ndarray (binary defect mask), or None
        "measurements": {},                   # dict, can be {}
    }


def detect_xxx(processed, segmentation):
    """
    TODO: one-line description of what this detects.

    Parameters
    ----------
    processed : dict
        Output of preprocess_image() - has "original", "gray",
        "gray_enhanced", "hsv", "lab", "denoised", etc.
    segmentation : dict
        Output of segment_glove() - has "glove_mask" (binary mask,
        glove=255/background=0) and "glove_area" (pixel count).

    Returns
    -------
    dict
        Result dict following the evaluate.py detector contract:
        defect_name, detected, detection_score, algorithm,
        bounding_box, mask, measurements. Must pass evaluate.py's
        validate_result().
    """
    glove_mask = segmentation.get("glove_mask")
    glove_area = segmentation.get("glove_area", 0)

    # Rule 2: bail out cleanly rather than analysing background pixels
    # if segmentation didn't produce anything usable.
    if glove_mask is None or not glove_area or glove_area <= 0:
        return _empty_result()

    # TODO: pull whichever preprocessed image(s) you need, e.g.
    # gray_enhanced = processed.get("gray_enhanced")
    # lab = processed.get("lab")

    # TODO: do the actual (classical CV) analysis here, restricted to
    # glove_mask. Common pattern: build a binary anomaly/defect mask,
    # then find contours / connected components on it.
    #
    # anomaly_mask = ...  # np.uint8, 0/255, same shape as glove_mask
    #
    # if cv2.countNonZero(anomaly_mask) == 0:
    #     return _empty_result()

    # TODO: score the candidate, clamped to [0, 1] - never return a raw
    # unclamped ratio, np.clip (or an equivalent bound) is mandatory
    # since evaluate.py's validate_result() rejects anything outside
    # 0.0-1.0.
    #
    # detection_score = float(np.clip(some_ratio, 0.0, 1.0))
    # detected = detection_score >= DETECTION_SCORE_THRESHOLD

    # TODO: derive a tight bounding box + mask for the winning
    # candidate, e.g.
    #
    # ys, xs = np.where(candidate_mask > 0)
    # x, y = int(xs.min()), int(ys.min())
    # w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)

    # TODO: replace this stub return with the real result. Rule 3: any
    # area_pct here must divide by glove_area, not image.shape.
    #
    # return {
    #     "defect_name": "TODO_defect_name",
    #     "detected": bool(detected),
    #     "detection_score": detection_score,
    #     "algorithm": ALGORITHM,
    #     "bounding_box": (x, y, w, h),
    #     "mask": candidate_mask,
    #     "measurements": {
    #         "area_pct": round(100.0 * cv2.countNonZero(candidate_mask) / glove_area, 3),
    #     },
    # }

    return _empty_result()
