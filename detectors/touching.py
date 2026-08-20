"""Touching-finger detector.

Detects adjacent glove fingers that are stuck/touching by analysing whether the
upper glove silhouette contains the deep valleys that normally separate fingers.
Designed to consume the shared ``processed`` and ``segmentation`` dictionaries.
"""

import cv2
import numpy as np

from detectors.finger_not_enough import locate_peaks


def _empty_result():
    return {
        "defect_name": "touching",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": "adjacent fingertip shallow-gap + local contact-notch analysis",
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def _column_top(mask, left, right):
    """Return topmost foreground y for each usable column in [left, right]."""
    h, w = mask.shape[:2]
    values = []
    for x in range(max(0, int(left)), min(w, int(right) + 1)):
        ys = np.flatnonzero(mask[:, x] > 0)
        if ys.size:
            values.append(float(ys[0]))
    return values


def _find_touching_pair(mask, peaks, glove_h, glove_w):
    """Find adjacent fingertip peaks whose separating notch is too shallow."""
    ordered = sorted(peaks, key=lambda p: p[0])
    gaps = []
    for left, right in zip(ordered, ordered[1:]):
        spacing = float(right[0] - left[0])
        if not (0.035 * glove_w <= spacing <= 0.32 * glove_w):
            continue

        # Trim the fingertip centres slightly. This prevents either rounded tip
        # itself from dominating the measurement while retaining the seam/notch.
        trim = max(1, int(0.16 * spacing))
        tops = _column_top(mask, left[0] + trim, right[0] - trim)
        if len(tops) < 3:
            continue

        tip_y = 0.5 * (left[1] + right[1])
        valley_y = float(np.percentile(tops, 85))
        depth = max(0.0, valley_y - tip_y)
        gaps.append({
            "left": left,
            "right": right,
            "valley_y": valley_y,
            "depth": depth,
            "depth_ratio": depth / max(glove_h, 1),
        })

    if not gaps:
        return None, gaps

    depths = [g["depth"] for g in gaps]
    typical_depth = float(np.median(depths))
    candidate = min(gaps, key=lambda g: g["depth_ratio"])
    candidate["relative_depth"] = candidate["depth"] / max(typical_depth, 1.0)

    # A genuine touching seam is shallow in absolute terms and also clearly
    # shallower than the glove's other interdigital valleys.
    if candidate["depth_ratio"] <= 0.115 and candidate["relative_depth"] <= 0.62:
        return candidate, gaps
    return None, gaps


def _find_shallow_contact(contour, defects, x0, y0, glove_h, glove_w):
    """Locate a missing/shallow interdigital notch on the glove contour.

    A touching or overlapping pair often produces only one fingertip peak, so
    there is no adjacent peak pair for ``_find_touching_pair`` to examine.  The
    small notch at the end of the contact seam is still represented as a
    convexity defect, however.  Keep only compact upper-hand notches; this
    rejects the large palm/thumb and wrist concavities.
    """
    if defects is None:
        return None

    candidates = []
    for d in defects[:, 0]:
        s, e, f, depth_raw = map(int, d)
        start = contour[s][0].astype(np.float32)
        end = contour[e][0].astype(np.float32)
        far = contour[f][0].astype(np.float32)
        depth = depth_raw / 256.0
        chord = float(np.linalg.norm(end - start))

        fx, fy = float(far[0]), float(far[1])
        if not (x0 + 0.08 * glove_w <= fx <= x0 + 0.92 * glove_w):
            continue
        if not (y0 + 0.02 * glove_h <= fy <= y0 + 0.60 * glove_h):
            continue
        if not (0.004 * glove_w <= depth <= 0.065 * glove_w):
            continue
        if not (0.045 * glove_w <= chord <= 0.42 * glove_w):
            continue

        # Prefer a definite notch over tiny outline noise, then prefer notches
        # higher on the fingers over creases close to the palm.
        depth_strength = min(depth / max(0.035 * glove_w, 1.0), 1.0)
        height_strength = 1.0 - (fy - y0) / max(0.60 * glove_h, 1.0)
        candidates.append((0.65 * depth_strength + 0.35 * height_strength,
                           (int(round(fx)), int(round(fy))), depth))

    return max(candidates, key=lambda item: item[0]) if candidates else None


def _contact_region(mask, centre, glove_h, glove_w):
    """Return a small circular mask and box centred on the contact point."""
    cx, cy = centre
    radius = max(7, int(round(0.055 * glove_w)))
    region = np.zeros_like(mask)
    cv2.circle(region, (int(cx), int(cy)), radius, 255, -1)

    # Include a narrow amount of material on both sides of the seam.  Do not
    # expand to the whole finger or upper hand.
    region = cv2.bitwise_and(region, mask)
    x = max(0, int(cx) - radius)
    y = max(0, int(cy) - radius)
    x2 = min(mask.shape[1] - 1, int(cx) + radius)
    y2 = min(mask.shape[0] - 1, int(cy) + radius)
    return region, (x, y, x2 - x + 1, y2 - y + 1), radius


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

    shallow_contact = _find_shallow_contact(
        contour, defects, x0, y0, glove_h, glove_w
    )

    # Five separated digits provide four interdigital valleys, including the
    # thumb/index separation. The old expected count of three was the main false
    # negative: one touching pair could erase a valley and still appear normal.
    deep_count = len(deep_defects)
    moderate_count = len(moderate_defects)
    missing_deep = np.clip((4.0 - deep_count) / 2.0, 0.0, 1.0)
    missing_moderate = np.clip((4.0 - moderate_count) / 2.0, 0.0, 1.0)

    # Convex-hull fill ratio becomes higher when gaps between fingers disappear.
    contour_area = float(cv2.contourArea(contour))
    hull_pts = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull_pts)), 1.0)
    solidity = contour_area / hull_area
    solidity_signal = np.clip((solidity - 0.70) / 0.22, 0.0, 1.0)

    peaks = locate_peaks(mask)
    touching_pair, pair_gaps = _find_touching_pair(mask, peaks, glove_h, glove_w)
    pair_signal = 0.0
    if touching_pair is not None:
        absolute_signal = np.clip((0.115 - touching_pair["depth_ratio"]) / 0.10, 0.0, 1.0)
        relative_signal = np.clip((0.62 - touching_pair["relative_depth"]) / 0.52, 0.0, 1.0)
        pair_signal = float(0.55 * absolute_signal + 0.45 * relative_signal)

    score = float(np.clip(
        0.48 * missing_deep + 0.18 * missing_moderate
        + 0.14 * solidity_signal + 0.45 * pair_signal,
        0.0, 1.0,
    ))
    if touching_pair is not None:
        score = max(score, 0.72)
    detected = score >= 0.50

    defect_mask = np.zeros_like(mask)
    bbox = None
    contact_point = None
    contact_radius = None
    if detected and touching_pair is not None:
        left, right = touching_pair["left"], touching_pair["right"]
        contact_point = (
            int(round(0.5 * (left[0] + right[0]))),
            int(round(touching_pair["valley_y"])),
        )
    elif detected and shallow_contact is not None:
        # Peak detection can merge two overlapping fingers into one peak.  In
        # that case localise the small contour notch at the contact-seam end.
        contact_point = shallow_contact[1]

    if contact_point is not None:
        defect_mask, bbox, contact_radius = _contact_region(
            mask, contact_point, glove_h, glove_w
        )

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
            "fingertip_peaks": int(len(peaks)),
            "candidate_pair_gaps": int(len(pair_gaps)),
            "touching_pair_found": touching_pair is not None,
            "contact_point": contact_point,
            "contact_radius_px": contact_radius,
            "shallow_contact_found": shallow_contact is not None,
            "touching_gap_depth_ratio": (
                round(float(touching_pair["depth_ratio"]), 4)
                if touching_pair is not None else None
            ),
            "touching_gap_relative_depth": (
                round(float(touching_pair["relative_depth"]), 4)
                if touching_pair is not None else None
            ),
        },
    })
    return result
