import cv2
import numpy as np


PEAK_SMOOTH_WINDOW = 15

PEAK_LOCAL_MAX_WINDOW_FRAC = 0.01

# Lower than tearing_fingertip.py's 0.06: at this detector's much lower prominence cutoff, two genuinely separate adjacent fingertips can sit only ~0.055-0.06 apart in index terms and would otherwise get merged.
PEAK_NMS_SEPARATION_FRAC = 0.03

# Deliberately much lower than tearing_fingertip.py's 0.35, since a short/stubby finger still needs to register as its own peak rather than being smoothed into the palm.
PEAK_MIN_PROMINENCE_FRAC = 0.12

MAX_FINGERTIPS = 5

# Candidate points in the bottom fraction of the glove's own bounding box are excluded - the cuff/wrist trim, not a finger.
PEAK_BOTTOM_MARGIN_FRACTION = 0.05

EXPECTED_MAIN_FINGERS = 4

# A confirmed thumb's length ratio was 0.16-0.28 on this dataset, while a genuinely-present main finger that's merely shorter than its neighbours never dropped below ~0.4.
# TUNED-BY-EYE on the 68-image dataset
THUMB_LENGTH_RATIO = 0.4

# Calibrated so a confirmed thumb bulge (15-109px prominence across this dataset's cases) still passes, while background/mask noise (<5px) does not.
# TUNED-BY-EYE on the 68-image dataset
BULGE_MIN_PROMINENCE_PX = 20

# An edge gap only counts as a plausible missing-finger location if it's at least this fraction of the median interior gap wide - otherwise it's just the normal margin between the outermost finger and the glove's silhouette edge.
EDGE_MIN_RATIO_FOR_CANDIDACY = 0.7

# A genuinely fused/missing interior finger showed a 2.3-3.3x margin over the narrowest other gap on this dataset, well clear of the ~1.2-1.3x margin that's just normal finger-spacing variation.
# TUNED-BY-EYE on the 68-image dataset
STRONG_INTERIOR_MARGIN = 2.0

# Same idea as STRONG_INTERIOR_MARGIN but as an absolute ratio, for when there's only one interior gap to judge.
# TUNED-BY-EYE on the 68-image dataset
STRONG_INTERIOR_SINGLE_RATIO = 1.3

# The reported box is the middle fraction of the gap's x-range; GAP_TRIM_FRAC is trimmed off each side so it doesn't overlap the genuine fingers flanking it.
GAP_TRIM_FRAC = 0.2

# Floors the gap box's height so a gap whose local surface happens to sit unusually high still renders a legible box instead of a sliver.
MIN_GAP_BOX_HEIGHT_FRACTION = 0.6

# Gap-path detection_score is a category verdict from finger count alone - a blended gap-width signal was tried and dropped since it didn't separate this dataset's 5 real from 14 false gap-branch positives at all.
FINGER_COUNT_SCORE_ONE_MISSING = 0.75   # exactly 4 of 5 fingers found
FINGER_COUNT_SCORE_SEVERE = 1.0         # 3 or fewer fingers found

MIN_VALLEY_DEPTH_RATIO = 0.10
MAX_VALLEY_ANGLE_DEG = 115.0
MAX_VALLEY_DEPTH_Y_FRACTION = 0.75
NMS_SEPARATION_FRAC = 0.03
SHORT_FINGER_RATIO_THRESHOLD = 0.45
MIN_VALLEYS_FOR_LOCALIZATION = 4

DETECTION_SCORE_THRESHOLD = 0.5

ALGORITHM = (
    "Fingertip localisation via contour distance-from-centroid peaks "
    "against a hard prior of 4 main fingers + 1 thumb; the thumb is "
    "identified separately (by length ratio, or a column-top-height "
    "bulge search) so it never masks a genuinely missing main finger; "
    "when fewer than 4 main fingers are found, the missing one is "
    "localised to an abnormally wide interior gap when one clearly "
    "stands out, else to whichever edge is both wide enough and free "
    "of real material (no bulge) - catching an edge-missing finger "
    "that widens no interior gap at all; when all 4 main fingers are "
    "found, falls back to convexity-defect valley/slot analysis to "
    "catch a finger that's present but short of its neighbours' length"
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


def _largest_contour(glove_mask):
    """Returns the largest contour in glove_mask, or None if unusable."""
    contours, _ = cv2.findContours(glove_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4 or cv2.contourArea(contour) <= 0:
        return None

    return contour


def _circular_smooth(values, window):
    n = len(values)
    if window <= 1 or n == 0:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.concatenate([values[-window:], values, values[:window]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[window:window + n]


def locate_peaks(glove_mask):
    """Find up to MAX_FINGERTIPS fingertip peaks via distance-from-centroid contour protrusion (same technique as tearing_fingertip.py, lower prominence cutoff); returns (x, y, length) left-to-right by x."""
    contours, _ = cv2.findContours(glove_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    n = len(contour)
    if n < 20:
        return []

    # Re-roots the contour array at its bottom-most point so cv2.findContours' arbitrary start seam always falls in the wrist/cuff region (excluded by min_valid_y below) instead of possibly between two real fingertips, which would make the circular NMS below wrongly treat them as index-adjacent.
    contour = np.roll(contour, -int(np.argmax(contour[:, 1])), axis=0)

    moments = cv2.moments(glove_mask)
    if moments["m00"] == 0:
        return []
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    dists = np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)
    smoothed = _circular_smooth(dists, PEAK_SMOOTH_WINDOW)

    median_d = float(np.median(smoothed))
    max_d = float(smoothed.max())
    prominence_threshold = median_d + PEAK_MIN_PROMINENCE_FRAC * (max_d - median_d)

    _, bbox_y, _, bbox_h = cv2.boundingRect(contour)
    min_valid_y = bbox_y + (1.0 - PEAK_BOTTOM_MARGIN_FRACTION) * bbox_h

    half_win = max(3, int(n * PEAK_LOCAL_MAX_WINDOW_FRAC))
    candidates = []
    for i in range(n):
        if contour[i][1] > min_valid_y:
            continue
        idxs = np.arange(i - half_win, i + half_win + 1) % n
        if smoothed[i] >= smoothed[idxs].max() and smoothed[i] > prominence_threshold:
            candidates.append(i)
    candidates.sort(key=lambda i: smoothed[i], reverse=True)

    min_sep = int(n * PEAK_NMS_SEPARATION_FRAC)
    selected = []
    for i in candidates:
        if all(min(abs(i - j), n - abs(i - j)) > min_sep for j in selected):
            selected.append(i)
        if len(selected) >= MAX_FINGERTIPS:
            break

    def _to_point(i):
        return (int(contour[i][0]), int(contour[i][1]), float(smoothed[i] - median_d))

    selected.sort(key=lambda i: contour[i][0])
    return [_to_point(i) for i in selected]


def _find_bulge(glove_mask, x_left, x_right, step=4):
    """Find the most prominent local minimum in glove_mask's column-top-height profile over [x_left, x_right] - real material bulging above an otherwise-tapering silhouette, distinguishing a weak digit from genuinely empty space; returns (x, y, prominence) or None."""
    h, w = glove_mask.shape[:2]
    xs, ys = [], []
    for x in range(max(0, int(x_left)), min(w, int(x_right) + 1), step):
        col = np.nonzero(glove_mask[:, x])[0]
        if col.size:
            xs.append(x)
            ys.append(int(col[0]))
    if len(ys) < 5:
        return None

    best = None
    for i in range(1, len(ys) - 1):
        prominence = min(max(ys[:i]), max(ys[i:])) - ys[i]
        if prominence > 0 and (best is None or prominence > best[2]):
            best = (xs[i], ys[i], float(prominence))
    return best


def _identify_thumb(peaks, glove_mask, bx):
    """Identify the thumb as the leftmost peak, confirmed either by being dramatically shorter than the others (THUMB_LENGTH_RATIO) or by a genuine bulge found between the glove's left edge and the leftmost peak; returns (thumb_x or None, main_peaks with a length-confirmed thumb removed)."""
    if not peaks:
        return None, []

    ordered = sorted(peaks, key=lambda p: p[0])
    leftmost = ordered[0]
    others = ordered[1:]

    if others:
        median_other_len = float(np.median([p[2] for p in others]))
        if median_other_len > 1e-6 and leftmost[2] / median_other_len < THUMB_LENGTH_RATIO:
            return float(leftmost[0]), others

    bulge = _find_bulge(glove_mask, bx, leftmost[0])
    if bulge is not None and bulge[2] >= BULGE_MIN_PROMINENCE_PX:
        return float(bulge[0]), list(ordered)

    return None, list(ordered)


def _localize_missing_main_finger(main_peaks, thumb_x, glove_mask, bx, bw):
    """Locate the one missing/fused main finger: a standout interior gap first, else whichever edge is wide enough and free of real material, else the best non-standout fallback."""
    xs = [p[0] for p in main_peaks]
    ys = [p[1] for p in main_peaks]

    interior = [
        {"type": "interior", "x_left": xs[i], "x_right": xs[i + 1], "width": xs[i + 1] - xs[i],
         "bound_ys": [ys[i], ys[i + 1]]}
        for i in range(len(xs) - 1)
    ]
    widths = [g["width"] for g in interior if g["width"] > 0]
    median_gap = float(np.median(widths)) if widths else max(1.0, bw / float(EXPECTED_MAIN_FINGERS))
    for g in interior:
        g["ratio"] = g["width"] / median_gap if median_gap > 1e-6 else 0.0

    best_interior = max(interior, key=lambda g: g["ratio"]) if interior else None
    strong_interior = False
    if best_interior is not None:
        if len(interior) >= 2:
            worst_ratio = min(g["ratio"] for g in interior)
            strong_interior = (best_interior["ratio"] / max(worst_ratio, 0.05)) >= STRONG_INTERIOR_MARGIN
        else:
            strong_interior = best_interior["ratio"] >= STRONG_INTERIOR_SINGLE_RATIO

    if strong_interior:
        return best_interior

    left_edge_x0 = thumb_x if thumb_x is not None else bx
    right_edge_x1 = bx + bw
    edge_candidates = []

    left_bulge = _find_bulge(glove_mask, left_edge_x0, xs[0])
    if left_bulge is None or left_bulge[2] < BULGE_MIN_PROMINENCE_PX:
        width = xs[0] - left_edge_x0
        ratio = width / median_gap if median_gap > 1e-6 else 0.0
        if ratio >= EDGE_MIN_RATIO_FOR_CANDIDACY:
            edge_candidates.append({"type": "edge_left", "x_left": left_edge_x0, "x_right": xs[0],
                                     "width": width, "ratio": ratio, "bound_ys": [ys[0]]})

    right_bulge = _find_bulge(glove_mask, xs[-1], right_edge_x1)
    if right_bulge is None or right_bulge[2] < BULGE_MIN_PROMINENCE_PX:
        width = right_edge_x1 - xs[-1]
        ratio = width / median_gap if median_gap > 1e-6 else 0.0
        if ratio >= EDGE_MIN_RATIO_FOR_CANDIDACY:
            edge_candidates.append({"type": "edge_right", "x_left": xs[-1], "x_right": right_edge_x1,
                                     "width": width, "ratio": ratio, "bound_ys": [ys[-1]]})

    if edge_candidates:
        return max(edge_candidates, key=lambda g: g["ratio"])

    if best_interior is not None:
        return best_interior

    # Last resort: only 1 main peak, so no interior gap exists at all - falls back to whichever edge is wider.
    left_width = xs[0] - left_edge_x0
    right_width = right_edge_x1 - xs[-1]
    if right_width >= left_width:
        return {"type": "edge_right", "x_left": xs[-1], "x_right": right_edge_x1, "width": right_width,
                "ratio": right_width / median_gap if median_gap > 1e-6 else 0.0, "bound_ys": [ys[-1]]}
    return {"type": "edge_left", "x_left": left_edge_x0, "x_right": xs[0], "width": left_width,
            "ratio": left_width / median_gap if median_gap > 1e-6 else 0.0, "bound_ys": [ys[0]]}


def _column_top_ys(glove_mask, x_left, x_right):
    """Topmost glove_mask y at each integer column in [x_left, x_right]."""
    h, w = glove_mask.shape[:2]
    tops = []
    for x in range(max(0, int(x_left)), min(w, int(x_right) + 1)):
        col = np.nonzero(glove_mask[:, x])[0]
        if col.size:
            tops.append(int(col[0]))
    return tops


def _gap_bounding_box(gap, glove_mask, min_height, glove_bottom):
    """The box for a flagged gap: middle GAP_TRIM_FRAC-trimmed x-range, down to the MEDIAN (not minimum) column-top y so a lone bump doesn't collapse it to a sliver."""
    width = gap["x_right"] - gap["x_left"]
    trim = GAP_TRIM_FRAC * width
    shrunk_left = gap["x_left"] + trim
    shrunk_right = gap["x_right"] - trim
    if shrunk_right <= shrunk_left:
        shrunk_left, shrunk_right = gap["x_left"], gap["x_right"]

    target_peak_y = int(np.median(gap["bound_ys"]))

    col_tops = _column_top_ys(glove_mask, shrunk_left, shrunk_right)
    baseline_y = int(np.median(col_tops)) if col_tops else target_peak_y + min_height

    box_y = min(target_peak_y, baseline_y)
    box_h = max(baseline_y - box_y, min_height)
    box_h = max(5, min(box_h, glove_bottom - box_y))
    box_x = int(shrunk_left)
    box_w = max(5, int(shrunk_right - shrunk_left))
    return box_x, box_y, box_w, box_h


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


def find_valleys(contour, min_depth_ratio=MIN_VALLEY_DEPTH_RATIO):
    """Find confirmed inter-finger valley points via convexity defects, left-to-right; only used for the secondary short-finger check, since a fused/missing finger's own valleys are exactly what this can't rely on."""
    glove_area = cv2.contourArea(contour)
    if glove_area <= 0:
        return []
    size_scale = float(np.sqrt(glove_area))
    min_depth_px = min_depth_ratio * size_scale

    _, bbox_y, _, bbox_h = cv2.boundingRect(contour)
    max_valid_y = bbox_y + MAX_VALLEY_DEPTH_Y_FRACTION * bbox_h

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    hull_idx = np.unique(hull_idx.flatten())
    hull_idx = np.sort(hull_idx).reshape(-1, 1)
    if len(hull_idx) < 4:
        return []

    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        return []

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

    # Non-max suppression: keep the deepest defect among any cluster of candidates that are close together on the contour.
    n = len(contour)
    min_sep = int(n * NMS_SEPARATION_FRAC)
    candidates.sort(key=lambda c: c[2], reverse=True)
    confirmed = []
    for far_idx, far_pt, depth_px in candidates:
        if all(min(abs(far_idx - j), n - abs(far_idx - j)) > min_sep for j, _, _ in confirmed):
            confirmed.append((far_idx, far_pt, depth_px))

    confirmed.sort(key=lambda c: c[1][0])  # left -> right by x
    return confirmed


def locate_missing_finger(contour, valleys):
    """Measure every finger slot's length (valley baseline to fingertip peak) and return the worst (most below-median) one if it clears SHORT_FINGER_RATIO_THRESHOLD; only called when peak-counting already found all 5 fingers."""
    if len(valleys) < MIN_VALLEYS_FOR_LOCALIZATION:
        return None

    valleys = sorted(valleys, key=lambda v: v[2], reverse=True)[:4]
    valleys = sorted(valleys, key=lambda v: v[1][0])
    valley_xs = [v[1][0] for v in valleys]
    valley_ys = [v[1][1] for v in valleys]

    bx, by, bw, bh = cv2.boundingRect(contour)
    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]

    boundaries = [bx] + valley_xs + [bx + bw]

    slots = []
    for i in range(len(boundaries) - 1):
        x_left, x_right = boundaries[i], boundaries[i + 1]
        if x_right <= x_left:
            continue
        in_slot = (xs >= x_left) & (xs <= x_right)
        if not np.any(in_slot):
            continue
        peak_y = int(ys[in_slot].min())

        neighbour_ys = []
        if i > 0:
            neighbour_ys.append(valley_ys[i - 1])
        if i < len(valley_ys):
            neighbour_ys.append(valley_ys[i])
        baseline_y = float(np.mean(neighbour_ys)) if neighbour_ys else float(by + bh)

        slots.append({
            "index": i,
            "x_left": int(x_left),
            "x_right": int(x_right),
            "peak_y": peak_y,
            "baseline_y": baseline_y,
            "length": max(0.0, baseline_y - peak_y),
        })

    if len(slots) < 3:
        return None

    lengths = [s["length"] for s in slots]
    worst = None
    for i, s in enumerate(slots):
        others = lengths[:i] + lengths[i + 1:]
        median_other = float(np.median(others)) if others else 0.0
        ratio = (s["length"] / median_other) if median_other > 1e-6 else 1.0
        s["ratio"] = ratio

        # Outer slots (thumb-side/pinky-side) only have one neighbouring valley to anchor a baseline on, and the thumb points diagonally rather than straight up, so this metric systematically misreads it as "short" - restricted to the inner, two-neighbour slots.
        if i == 0 or i == len(slots) - 1:
            continue
        if worst is None or ratio < worst["ratio"]:
            worst = s

    if worst is None or worst["ratio"] >= SHORT_FINGER_RATIO_THRESHOLD:
        return None

    normal_peaks = [s["peak_y"] for s in slots if s is not worst]
    target_peak_y = int(np.median(normal_peaks)) if normal_peaks else worst["peak_y"]

    box_y = min(target_peak_y, int(worst["baseline_y"]))
    box_h = max(1, int(worst["baseline_y"]) - box_y)
    box_x = worst["x_left"]
    box_w = max(1, worst["x_right"] - worst["x_left"])

    return {
        "slot_index": worst["index"],
        "length_ratio": float(worst["ratio"]),
        "bounding_box": (int(box_x), int(box_y), int(box_w), int(box_h)),
        "finger_count": len(slots),
    }


def detect_finger_not_enough(processed, segmentation):
    """Detect a missing/fused/short finger in a segmented glove; returns a result dict per the detector contract (processed is unused - the signal is purely geometric - but accepted for a consistent detector signature)."""
    glove_mask = segmentation.get("glove_mask")
    glove_area = segmentation.get("glove_area", 0)

    if glove_mask is None or not glove_area or glove_area <= 0:
        return _empty_result()

    contour = _largest_contour(glove_mask)
    if contour is None:
        return _empty_result()

    peaks = locate_peaks(glove_mask)
    bx, by, bw, bh = cv2.boundingRect(contour)
    thumb_x, main_peaks = _identify_thumb(peaks, glove_mask, bx)

    def _whole_glove_fallback():
        # Localisation has completely failed (fewer than 2 peaks, or a thumb-only hand with zero main peaks) - scored 0.0/not detected, since a whole-glove box with no real localisation is the least trustworthy result this detector can produce.
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros_like(glove_mask)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        return {
            "defect_name": "finger_not_enough",
            "detected": False,
            "detection_score": 0.0,
            "algorithm": ALGORITHM,
            "bounding_box": (int(x), int(y), int(w), int(h)),
            "mask": mask,
            "measurements": {
                "area_pct": None,
                "fingers_found": len(peaks),
                "localization": "unavailable",
            },
        }

    # Fewer than 2 peaks total: no pair to compute any spacing from.
    if len(peaks) < 2:
        return _whole_glove_fallback()

    # All 4 main fingers found: check the present-but-short signal as a secondary check (only meaningful once count-wise nothing looks missing); whether the thumb was confirmed doesn't matter here.
    if len(main_peaks) >= EXPECTED_MAIN_FINGERS:
        valleys = find_valleys(contour)
        located = locate_missing_finger(contour, valleys)

        if located is None:
            return {
                "defect_name": "finger_not_enough",
                "detected": False,
                "detection_score": 0.0,
                "algorithm": ALGORITHM,
                "bounding_box": None,
                "mask": None,
                "measurements": {"area_pct": None, "fingers_found": len(peaks)},
            }

        x, y, w, h = located["bounding_box"]
        mask = np.zeros_like(glove_mask)
        mask[y:y + h, x:x + w] = 255
        detection_score = float(np.clip(1.0 - located["length_ratio"], 0.0, 1.0))
        detected = detection_score >= DETECTION_SCORE_THRESHOLD

        return {
            "defect_name": "finger_not_enough",
            "detected": detected,
            "detection_score": detection_score,
            "algorithm": ALGORITHM,
            "bounding_box": (int(x), int(y), int(w), int(h)),
            "mask": mask,
            "measurements": {
                "area_pct": None,
                "fingers_found": len(peaks),
                "missing_slot_index": located["slot_index"],
                "length_ratio": round(located["length_ratio"], 3),
                "localization": "slot",
            },
        }

    # Fewer than 4 main fingers: a genuine finger is missing or fused - localise it via _localize_missing_main_finger.
    if not main_peaks:
        return _whole_glove_fallback()

    worst_gap = _localize_missing_main_finger(main_peaks, thumb_x, glove_mask, bx, bw)
    median_finger_length = float(np.median([p[2] for p in main_peaks]))
    min_gap_height = max(5, int(MIN_GAP_BOX_HEIGHT_FRACTION * median_finger_length))
    x, y, w, h = _gap_bounding_box(worst_gap, glove_mask, min_gap_height, by + bh)

    # Category verdict from finger count alone; worst_gap["ratio"] is reported below in measurements but not used to compute this score (see FINGER_COUNT_SCORE_ONE_MISSING).
    detection_score = (
        FINGER_COUNT_SCORE_ONE_MISSING if len(main_peaks) == EXPECTED_MAIN_FINGERS - 1
        else FINGER_COUNT_SCORE_SEVERE
    )
    detected = detection_score >= DETECTION_SCORE_THRESHOLD

    mask = np.zeros_like(glove_mask)
    mask[y:y + h, x:x + w] = 255

    return {
        "defect_name": "finger_not_enough",
        "detected": detected,
        "detection_score": float(detection_score),
        "algorithm": ALGORITHM,
        "bounding_box": (int(x), int(y), int(w), int(h)),
        "mask": mask,
        "measurements": {
            "area_pct": None,
            "fingers_found": len(peaks),
            "main_fingers_found": len(main_peaks),
            "thumb_confirmed": thumb_x is not None,
            "gap_type": worst_gap["type"],
            "gap_ratio": round(worst_gap["ratio"], 3),
            "localization": "gap",
        },
    }
