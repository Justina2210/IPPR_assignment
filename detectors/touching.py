"""Touching-finger detector.

Detects adjacent glove fingers that are stuck/touching by analysing whether the
upper glove silhouette contains the deep valleys that normally separate fingers.
Designed to consume the shared ``processed`` and ``segmentation`` dictionaries.
"""

import cv2
import numpy as np


def _empty_result():
    return {
        "defect_name": "touching",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": "finger-valley depth + convexity-defect silhouette analysis",
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def detect_touching(processed: dict, segmentation: dict) -> dict:
    """Detect fingers that are abnormally joined/touching.

    The detector examines the upper 72% of the glove (finger/hand region), then
    measures deep convexity defects. A normal open-hand glove generally has
    several deep valleys between fingers. A touching pair removes or greatly
    shallows at least one valley.
    """
    result = _empty_result()
    mask = segmentation.get("glove_mask")
    if mask is None or np.count_nonzero(mask) < 500:
        return result

    mask = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return result

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    glove_h = max(1, y1 - y0 + 1)
    glove_w = max(1, x1 - x0 + 1)

    # Exclude the cuff/lower palm. Finger valleys should live in this region.
    hand_bottom = min(y1 + 1, y0 + int(0.72 * glove_h))
    roi_mask = np.zeros_like(mask)
    roi_mask[y0:hand_bottom, x0:x1 + 1] = mask[y0:hand_bottom, x0:x1 + 1]

    contour = _largest_contour(roi_mask)
    if contour is None or len(contour) < 4:
        return result

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        return result

    defects = cv2.convexityDefects(contour, hull_idx)
    deep_defects = []
    moderate_defects = []

    # Depth from cv2.convexityDefects is fixed point with 8 fractional bits.
    deep_threshold = 0.065 * glove_w
    moderate_threshold = 0.035 * glove_w

    if defects is not None:
        for d in defects[:, 0]:
            s, e, f, depth_raw = map(int, d)
            depth = depth_raw / 256.0
            far = tuple(contour[f][0])
            # Restrict to realistic finger-valley zone, avoiding side wrist/palm concavities.
            fy = far[1]
            fx = far[0]
            if not (y0 + 0.10 * glove_h <= fy <= y0 + 0.60 * glove_h):
                continue
            if not (x0 + 0.08 * glove_w <= fx <= x1 - 0.08 * glove_w):
                continue
            if depth >= moderate_threshold:
                moderate_defects.append((depth, far))
            if depth >= deep_threshold:
                deep_defects.append((depth, far))

    # Four separated fingers normally provide ~3 strong interdigital valleys.
    # Fewer strong valleys increases touching likelihood. Keep scoring smooth
    # because glove pose/material can hide one valley even in a good sample.
    deep_count = len(deep_defects)
    moderate_count = len(moderate_defects)
    missing_deep = np.clip((3.0 - deep_count) / 3.0, 0.0, 1.0)
    missing_moderate = np.clip((3.0 - moderate_count) / 3.0, 0.0, 1.0)

    # Convex-hull fill ratio becomes higher when gaps between fingers disappear.
    contour_area = float(cv2.contourArea(contour))
    hull_pts = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull_pts)), 1.0)
    solidity = contour_area / hull_area
    solidity_signal = np.clip((solidity - 0.70) / 0.22, 0.0, 1.0)

    score = float(np.clip(0.55 * missing_deep + 0.25 * missing_moderate + 0.20 * solidity_signal, 0.0, 1.0))
    detected = score >= 0.50

    defect_mask = np.zeros_like(mask)
    bbox = None
    if detected:
        # Localise the likely touching zone. If valleys are missing, highlight the
        # central finger band rather than incorrectly marking the entire glove.
        bx = x0 + int(0.12 * glove_w)
        by = y0 + int(0.04 * glove_h)
        bw = max(1, int(0.76 * glove_w))
        bh = max(1, int(0.54 * glove_h))
        cv2.rectangle(defect_mask, (bx, by), (min(x1, bx + bw), min(y1, by + bh)), 255, -1)
        defect_mask = cv2.bitwise_and(defect_mask, roi_mask)
        bbox = (bx, by, min(bw, x1 - bx + 1), min(bh, y1 - by + 1))

    area_pct = 100.0 * np.count_nonzero(defect_mask) / max(np.count_nonzero(mask), 1)
    result.update({
        "detected": bool(detected),
        "detection_score": score,
        "bounding_box": bbox,
        "mask": defect_mask,
        "measurements": {
            "area_pct": round(float(area_pct), 3),
            "deep_finger_valleys": int(deep_count),
            "moderate_finger_valleys": int(moderate_count),
            "upper_hand_solidity": round(float(solidity), 4),
        },
    })
    return result
