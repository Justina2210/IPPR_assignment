import cv2
import numpy as np

# Every threshold/weight below must be a named, commented constant (no
# bare magic numbers inline). If a constant was arrived at by looking
# at how it performed on the dataset rather than derived from a
# physical/geometric argument, tag it:
#     # TUNED-BY-EYE on the 68-image dataset

# TODO: replace with your own thresholds, e.g.
# MIN_COLOUR_DISTANCE = 18   # LAB a/b distance considered anomalous
# DETECTION_SCORE_THRESHOLD = 0.5   # detected/not-detected cutoff on the 0-1 score

ALGORITHM = "TODO: short human-readable description of the technique used"


def _empty_result():
    """Return value for 'nothing found / could not run' - keeps every early return in detect_xxx() consistent."""
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
    """TODO: one-line description of what this detects; must return a dict following the detector contract (see CLAUDE.md)."""
    glove_mask = segmentation.get("glove_mask")
    glove_area = segmentation.get("glove_area", 0)

    # Rule 2: bail out cleanly if segmentation gave nothing usable.
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
