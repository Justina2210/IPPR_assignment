"""
detectors/finger_not_enough.py
--------------------------------
Classical OpenCV detector for "finger_not_enough" defects (a
knitting/moulding defect where two adjacent fingers are fused together
or a finger fails to form as its own lobe, so the glove has fewer than
5 separate finger silhouettes). No ML.

Approach
--------
Count fingers directly from glove_mask's silhouette using the classic
convexity-defects finger-counting technique:

1. Take the glove's outer contour and its convex hull.
2. cv2.convexityDefects() finds every point where the contour dips
   inward from the hull - each deep, narrow, acute-angled dip between
   two hull vertices is the valley between two adjacent fingers. (This
   is the same signal detectors/tearing.py tried and dropped for tear
   detection - it fires on every finger gap on every hand, which made
   it useless for spotting a tear, but is exactly the wanted signal
   here: those gaps ARE what's being counted.)
3. Filter defects to keep only genuine inter-finger valleys:
   - deep enough, relative to the glove's own size (a shallow dip is
     contour digitisation noise or a soft flex, not the crease between
     two distinct fingers)
   - acute-angled at the far point (the standard "angle < ~90 deg" cut
     from hand-gesture-recognition tutorials - the wrist/cuff boundary
     and other body concavities are wide and obtuse, not sharp finger
     creases)
   - in the upper portion of the glove's bounding box (fingers are
     always the topmost structures in this dataset's consistent
     fingers-up photography; a defensive check against any other
     concavity elsewhere being mistaken for a finger gap)
4. finger_count = (number of confirmed valleys) + 1, since N valleys
   sit between N+1 separate finger lobes, clipped to [1, 5].

detected = True when finger_count < 5.

Two fingers fused together at manufacture (visible in this dataset as
one wide lobe with only a shallow nick, or a partial stitch mark, where
the valley should be) drop the valley count below 4, so finger_count
comes out at 4 or lower - exactly what this defect looks like.

Known limitation
-----------------
This is a pure *count* signal, as asked for - it cannot catch a defect
where all 5 fingers are present as separate lobes but one is
significantly under-length (not enough material knitted into it). One
image in this dataset (cotton_finger_not_enough_2.jpg) is exactly that
case: five clearly separated finger lobes, but the ring finger is
visibly shorter than its neighbours. A count-based detector will not
flag it - a length/proportion signal (e.g. comparing each finger's
protrusion length, as detectors/tearing_fingertip.py already measures,
against its neighbours) would be needed for that case, but that's a
different defect definition than "finger count < 5".
"""

import cv2
import numpy as np


# ============================================================
# THRESHOLDS (documented here so they can be copied into the report)
# ============================================================

# Minimum convexity-defect depth to count as a real inter-finger
# valley rather than digitisation noise, as a fraction of sqrt(glove_area)
# (a linear size measure, since depth is itself linear).
MIN_VALLEY_DEPTH_RATIO = 0.15

# A genuine finger valley is a sharp notch; the wrist/cuff boundary
# and other body concavities are wide and obtuse. The classic
# tutorial cutoff of 90 degrees rejected the (wider) thumb-index gap
# on real photos, so this is relaxed to still exclude flat/obtuse
# concavities while keeping every inter-finger valley.
MAX_VALLEY_ANGLE_DEG = 115.0

# Candidate valleys below this fraction of the glove's own bounding
# box (from the top) are excluded - fingers are always the topmost
# structures in this dataset's fingers-up photography, so a deep
# concavity lower down is not a finger gap.
MAX_VALLEY_DEPTH_Y_FRACTION = 0.75

# Two defects within this fraction of the contour's point count of
# each other are treated as the same physical valley (a slightly
# uneven valley floor can otherwise register as two adjacent defects).
NMS_SEPARATION_FRAC = 0.03

DETECTION_SCORE_ONE_MISSING = 0.75   # finger_count == 4
DETECTION_SCORE_SEVERE = 1.0         # finger_count <= 3
DETECTION_SCORE_THRESHOLD = 0.5

ALGORITHM = (
    "Finger count via convexity-defect finger-gap analysis on the "
    "glove silhouette's outer contour (depth + acute-angle + "
    "upper-region filters), detected when count < 5"
)


def _empty_result():
    return {
        "defect_name": "finger_not_enough",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": ALGORITHM,
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def _valley_angle_degrees(far, start, end):
    """Interior angle at `far` in the triangle far-start-end, degrees."""
    far, start, end = np.array(far, dtype=np.float64), np.array(start, dtype=np.float64), np.array(end, dtype=np.float64)
    a = np.linalg.norm(far - start)
    b = np.linalg.norm(far - end)
    c = np.linalg.norm(start - end)
    if a < 1e-6 or b < 1e-6:
        return 180.0
    cos_angle = np.clip((a ** 2 + b ** 2 - c ** 2) / (2 * a * b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def count_fingers(glove_mask):
    """
    Count separate finger lobes in glove_mask via convexity-defect
    finger-gap analysis.

    Returns
    -------
    tuple(int, list, numpy.ndarray or None)
        (finger_count, confirmed_valley_far_points, outer_contour)
        finger_count is clipped to [1, 5]. Returns (0, [], None) if the
        mask has no usable contour.
    """
    contours, _ = cv2.findContours(glove_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, [], None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        return 0, [], None

    glove_area = cv2.contourArea(contour)
    if glove_area <= 0:
        return 0, [], None
    size_scale = float(np.sqrt(glove_area))
    min_depth_px = MIN_VALLEY_DEPTH_RATIO * size_scale

    _, bbox_y, _, bbox_h = cv2.boundingRect(contour)
    max_valid_y = bbox_y + MAX_VALLEY_DEPTH_Y_FRACTION * bbox_h

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    hull_idx = np.unique(hull_idx.flatten())
    hull_idx = np.sort(hull_idx).reshape(-1, 1)
    if len(hull_idx) < 4:
        return 1, [], contour

    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        return 1, [], contour

    candidates = []
    for start_idx, end_idx, far_idx, depth in defects[:, 0]:
        depth_px = depth / 256.0
        if depth_px < min_depth_px:
            continue

        far_pt = tuple(int(v) for v in contour[far_idx][0])
        if far_pt[1] > max_valid_y:
            continue

        start_pt = tuple(int(v) for v in contour[start_idx][0])
        end_pt = tuple(int(v) for v in contour[end_idx][0])
        angle = _valley_angle_degrees(far_pt, start_pt, end_pt)
        if angle > MAX_VALLEY_ANGLE_DEG:
            continue

        candidates.append((far_idx, far_pt, depth_px))

    # Non-max suppression: keep the deepest defect among any cluster of
    # candidates that are close together on the contour (an uneven
    # valley floor can otherwise register as two adjacent defects).
    n = len(contour)
    min_sep = int(n * NMS_SEPARATION_FRAC)
    candidates.sort(key=lambda c: c[2], reverse=True)
    confirmed = []
    for far_idx, far_pt, depth_px in candidates:
        if all(min(abs(far_idx - j), n - abs(far_idx - j)) > min_sep for j, _ in confirmed):
            confirmed.append((far_idx, far_pt))

    finger_count = int(np.clip(len(confirmed) + 1, 1, 5))
    return finger_count, [pt for _, pt in confirmed], contour


def detect_finger_not_enough(processed, segmentation):
    """
    Detect a missing/fused finger in a segmented glove.

    Parameters
    ----------
    processed : dict
        Output of preprocess_image(). Unused directly - the signal is
        purely geometric - but accepted for a consistent detector
        signature.
    segmentation : dict
        Output of segment_glove().

    Returns
    -------
    dict
        Result dict following the evaluate.py detector contract:
        defect_name, detected, detection_score, algorithm,
        bounding_box, mask, measurements.
    """
    glove_mask = segmentation.get("glove_mask")
    glove_area = segmentation.get("glove_area", 0)

    if glove_mask is None or not glove_area or glove_area <= 0:
        return _empty_result()

    finger_count, valley_points, contour = count_fingers(glove_mask)
    if contour is None:
        return _empty_result()

    detected = finger_count < 5
    if not detected:
        score = 0.0
    elif finger_count == 4:
        score = DETECTION_SCORE_ONE_MISSING
    else:
        score = DETECTION_SCORE_SEVERE

    bounding_box = None
    mask = None
    if detected:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_box = (int(x), int(y), int(w), int(h))
        mask = np.zeros_like(glove_mask)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return {
        "defect_name": "finger_not_enough",
        "detected": bool(detected),
        "detection_score": float(score),
        "algorithm": ALGORITHM,
        "bounding_box": bounding_box,
        "mask": mask,
        "measurements": {
            "area_pct": round(100.0 * cv2.contourArea(contour) / glove_area, 3) if detected else None,
            "finger_count": finger_count,
            "valleys_found": len(valley_points),
        },
    }
