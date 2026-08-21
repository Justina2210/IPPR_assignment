"""Classical detector for a glove damaged by a large fold.

The target is a long raised/folded ridge or band across the palm/wrist, not
ordinary short wrinkles. Detection is restricted to the supplied glove mask.
"""
import cv2
import numpy as np


# TUNED-BY-EYE starting values from the six supplied positive samples.
PALM_TOP_RATIO = 0.40
PALM_BOTTOM_RATIO = 0.91
BOUNDARY_ERODE_RATIO = 0.018

MIN_LINE_LENGTH_RATIO = 0.16
STRONG_LINE_LENGTH_RATIO = 0.42
MAX_HORIZONTAL_ANGLE_DEG = 58.0
SIDE_FOLD_MIN_ANGLE_DEG = 58.0
SIDE_FOLD_X_RATIO = 0.64
VERTICAL_FOLD_MIN_TOP_RATIO = 0.47
VERTICAL_FOLD_MIN_BOTTOM_RATIO = 0.68
VERTICAL_FOLD_SIDE_MARGIN_RATIO = 0.12
MAX_HOUGH_LINES = 80

MIN_EDGE_CONTINUITY = 0.46
MIN_PAIRED_SUPPORT = 0.18
STRONG_PAIRED_SUPPORT = 0.62
MIN_LOCAL_CONTRAST = 0.10
MIN_CANDIDATE_SCORE = 0.46
DETECTION_SCORE_THRESHOLD = 0.52

FOLD_OVERLAY_THICKNESS_RATIO = 0.040
COTTON_HORIZONTAL_Y_SHIFT_RATIO = -0.045
COTTON_VERTICAL_X_SHIFT_RATIO = 0.050

ALGORITHM = (
    "Adaptive masked-palm edge analysis for a long fold ridge, confirmed by "
    "line continuity, parallel fold-band support, local contrast and geometry"
)


def _empty_result(shape=None):
    return {
        "defect_name": "damaged_by_fold",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": ALGORITHM,
        "bounding_box": None,
        "mask": None if shape is None else np.zeros(shape, dtype=np.uint8),
        "measurements": {},
    }


def _samples(line, count=31):
    x1, y1, x2, y2 = line
    return [
        (int(round(x1 + t * (x2 - x1))), int(round(y1 + t * (y2 - y1))))
        for t in np.linspace(0.0, 1.0, count)
    ]


def _line_geometry(line):
    x1, y1, x2, y2 = line
    dx, dy = float(x2 - x1), float(y2 - y1)
    length = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(abs(dy), max(abs(dx), 1e-6))))
    return length, angle


def _continuity(edges, line, radius, count=31):
    h, w = edges.shape
    hits = 0
    points = _samples(line, count)
    for x, y in points:
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        hits += int(np.any(edges[y0:y1, x0:x1] > 0))
    return float(hits / max(len(points), 1))


def _shift_line(line, normal_x, normal_y, distance):
    x1, y1, x2, y2 = line
    return (
        int(round(x1 + normal_x * distance)),
        int(round(y1 + normal_y * distance)),
        int(round(x2 + normal_x * distance)),
        int(round(y2 + normal_y * distance)),
    )


def _paired_edge_support(edges, line, glove_width):
    """A physical fold usually has another roughly parallel ridge/shadow."""
    x1, y1, x2, y2 = line
    dx, dy = float(x2 - x1), float(y2 - y1)
    length = float(np.hypot(dx, dy))
    if length <= 1.0:
        return 0.0
    nx, ny = -dy / length, dx / length
    radius = max(1, int(round(0.004 * glove_width)))
    offsets = [
        max(4, int(round(ratio * glove_width)))
        for ratio in (0.014, 0.024, 0.038, 0.055)
    ]
    scores = []
    for distance in offsets:
        scores.append(_continuity(
            edges, _shift_line(line, nx, ny, distance), radius, 25))
        scores.append(_continuity(
            edges, _shift_line(line, nx, ny, -distance), radius, 25))
    # Use the strongest nearby parallel boundary; requiring several would reject
    # folds where only one side produces a shadow.
    return float(max(scores, default=0.0))


def _side_material_support(mask, line, glove_width):
    """Reject external glove outlines by requiring material on both sides."""
    x1, y1, x2, y2 = line
    dx, dy = float(x2 - x1), float(y2 - y1)
    length = float(np.hypot(dx, dy))
    if length <= 1.0:
        return 0.0
    nx, ny = -dy / length, dx / length
    distance = max(3, int(round(0.018 * glove_width)))
    h, w = mask.shape
    supported = 0
    points = _samples(line, 23)[2:-2]
    for x, y in points:
        coordinates = [
            (int(round(x + nx * distance)), int(round(y + ny * distance))),
            (int(round(x - nx * distance)), int(round(y - ny * distance))),
        ]
        supported += int(all(
            0 <= px < w and 0 <= py < h and mask[py, px] > 0
            for px, py in coordinates))
    return float(supported / max(len(points), 1))


def _local_contrast(gray, mask, line, glove_width):
    """Measure line-to-neighbour contrast relative to the glove's own range."""
    x1, y1, x2, y2 = line
    dx, dy = float(x2 - x1), float(y2 - y1)
    length = float(np.hypot(dx, dy))
    if length <= 1.0:
        return 0.0, 0.0
    nx, ny = -dy / length, dx / length
    side_offset = max(4, int(round(0.025 * glove_width)))
    h, w = gray.shape
    centre_values, side_values = [], []
    for x, y in _samples(line, 27)[2:-2]:
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            centre_values.append(float(gray[y, x]))
        for sign in (-1.0, 1.0):
            px = int(round(x + sign * nx * side_offset))
            py = int(round(y + sign * ny * side_offset))
            if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
                side_values.append(float(gray[py, px]))
    if len(centre_values) < 8 or len(side_values) < 8:
        return 0.0, 0.0
    difference = abs(float(np.median(centre_values)) - float(np.median(side_values)))
    glove_values = gray[mask > 0]
    scale = max(float(np.percentile(glove_values, 90) - np.percentile(glove_values, 10)), 12.0)
    return float(np.clip(difference / scale, 0.0, 1.0)), float(difference)


def _adaptive_internal_edges(gray, mask, bounds):
    gx, gy, gw, gh = bounds
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    palm = np.zeros_like(mask)
    y0 = gy + int(round(PALM_TOP_RATIO * gh))
    y1 = gy + int(round(PALM_BOTTOM_RATIO * gh))
    palm[y0:y1, gx:gx + gw] = mask[y0:y1, gx:gx + gw]
    values = magnitude[palm > 0]
    if values.size < 50:
        return np.zeros_like(mask), 0, 0, 0.0

    low = int(np.clip(np.percentile(values, 64), 10, 130))
    high = int(np.clip(np.percentile(values, 90), low + 10, 245))
    edges = cv2.Canny(blurred, low, high)

    erosion = max(3, int(round(BOUNDARY_ERODE_RATIO * gw)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
    interior = cv2.erode(mask, kernel)
    search_mask = cv2.bitwise_and(interior, palm)
    edges = cv2.bitwise_and(edges, search_mask)
    density = float(np.count_nonzero(edges) / max(np.count_nonzero(search_mask), 1))
    return edges, low, high, density


def _candidates(edges, gray, mask, bounds):
    gx, gy, gw, gh = bounds
    minimum_length = max(20, int(round(MIN_LINE_LENGTH_RATIO * gw)))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(14, int(round(0.025 * gw))),
        minLineLength=minimum_length,
        maxLineGap=max(8, int(round(0.030 * gw))),
    )
    if lines is None:
        return []

    quick = []
    for raw in lines[:, 0]:
        line = tuple(int(value) for value in raw)
        length, angle = _line_geometry(line)
        centre_x = 0.5 * (line[0] + line[2])
        line_top = min(line[1], line[3])
        line_bottom = max(line[1], line[3])
        top_ratio = (line_top - gy) / float(gh)
        bottom_ratio = (line_bottom - gy) / float(gh)
        horizontal_or_diagonal = angle <= MAX_HORIZONTAL_ANGLE_DEG
        side_fold = bool(
            angle >= SIDE_FOLD_MIN_ANGLE_DEG
            and centre_x >= gx + SIDE_FOLD_X_RATIO * gw)
        vertical_fold = bool(
            angle >= SIDE_FOLD_MIN_ANGLE_DEG
            and top_ratio >= VERTICAL_FOLD_MIN_TOP_RATIO
            and bottom_ratio >= VERTICAL_FOLD_MIN_BOTTOM_RATIO
            and gx + VERTICAL_FOLD_SIDE_MARGIN_RATIO * gw <= centre_x
            <= gx + (1.0 - VERTICAL_FOLD_SIDE_MARGIN_RATIO) * gw)
        if length >= minimum_length and (
                horizontal_or_diagonal or side_fold or vertical_fold):
            quick.append((
                line, length, angle, side_fold, vertical_fold,
                top_ratio, bottom_ratio))
    quick.sort(key=lambda item: item[1], reverse=True)
    quick = quick[:MAX_HOUGH_LINES]

    candidates = []
    for (line, length, angle, side_fold, vertical_fold,
         line_top_ratio, line_bottom_ratio) in quick:
        radius = max(2, int(round(0.006 * gw)))
        continuity = _continuity(edges, line, radius)
        if continuity < MIN_EDGE_CONTINUITY:
            continue
        material = _side_material_support(mask, line, gw)
        if material < 0.62:
            continue
        paired = _paired_edge_support(edges, line, gw)
        if paired < MIN_PAIRED_SUPPORT:
            continue
        contrast, contrast_gray = _local_contrast(gray, mask, line, gw)
        if contrast < MIN_LOCAL_CONTRAST:
            continue

        # Vertical folds are compared with glove height; horizontal/diagonal
        # folds are compared with glove width.
        length_scale = gh if vertical_fold else gw
        length_ratio = length / float(length_scale)
        length_score = float(np.clip(
            (length_ratio - MIN_LINE_LENGTH_RATIO)
            / (STRONG_LINE_LENGTH_RATIO - MIN_LINE_LENGTH_RATIO), 0.0, 1.0))
        paired_score = float(np.clip(
            (paired - MIN_PAIRED_SUPPORT)
            / (STRONG_PAIRED_SUPPORT - MIN_PAIRED_SUPPORT), 0.0, 1.0))
        position_y = 0.5 * (line[1] + line[3])
        relative_y = (position_y - gy) / float(gh)
        palm_centre_score = float(np.clip(1.0 - abs(relative_y - 0.67) / 0.30, 0.0, 1.0))
        if vertical_fold:
            geometry_score = 0.74
        elif side_fold:
            geometry_score = 0.66
        else:
            geometry_score = 0.82
        score = float(np.clip(
            0.25 * length_score + 0.19 * continuity + 0.17 * paired_score
            + 0.16 * contrast + 0.13 * material
            + 0.06 * palm_centre_score + 0.04 * geometry_score,
            0.0, 1.0))
        candidates.append({
            "line": line, "score": score, "length": length,
            "length_ratio": length_ratio, "angle": angle,
            "continuity": continuity, "paired_support": paired,
            "side_material": material, "contrast": contrast,
            "contrast_gray": contrast_gray, "relative_y": relative_y,
            "side_fold": bool(side_fold),
            "vertical_fold": bool(vertical_fold),
            "line_top_ratio": float(line_top_ratio),
            "line_bottom_ratio": float(line_bottom_ratio),
            "centre_x_ratio": float((0.5 * (line[0] + line[2]) - gx) / gw),
        })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _select_textured_candidate(candidates):
    """Choose the physical fold rather than a regular cotton knit transition."""
    eligible = []
    for item in candidates:
        # Horizontal folds sit in the lower palm. A true cotton vertical fold
        # may begin slightly higher, but it must be long, right-of-centre and
        # continue toward the wrist. This rejects ordinary vertical knit ribs.
        if item["vertical_fold"]:
            valid_vertical = bool(
                item["line_top_ratio"] >= 0.47
                and item["line_bottom_ratio"] >= 0.75
                and item["length_ratio"] >= 0.16
                and 0.52 <= item["centre_x_ratio"] <= 0.88
                and item["contrast"] >= 0.12
                and item["paired_support"] >= 0.25)
            if not valid_vertical:
                continue
        elif item["relative_y"] < 0.58:
            continue
        lower_position = float(np.clip(
            (item["relative_y"] - 0.58) / 0.18, 0.0, 1.0))
        contrast_strength = float(np.clip(
            (item["contrast"] - MIN_LOCAL_CONTRAST) / 0.35, 0.0, 1.0))
        paired_strength = float(np.clip(
            (item["paired_support"] - MIN_PAIRED_SUPPORT)
            / (STRONG_PAIRED_SUPPORT - MIN_PAIRED_SUPPORT), 0.0, 1.0))
        # This value is only for candidate selection. Confidence remains the
        # candidate's measurement-derived detector score.
        vertical_preference = 0.32 if item["vertical_fold"] else 0.0
        rank = float(
            item["score"] + 0.22 * lower_position
            + 0.18 * contrast_strength + 0.10 * paired_strength
            + vertical_preference)
        copy = dict(item)
        copy["textured_rank"] = rank
        copy["lower_position_score"] = lower_position
        copy["vertical_preference"] = float(vertical_preference)
        eligible.append(copy)
    return max(eligible, key=lambda item: item["textured_rank"], default=None)


def detect_damaged_by_fold(processed, segmentation):
    glove_mask = None if segmentation is None else segmentation.get("glove_mask")
    glove_area = 0 if segmentation is None else int(segmentation.get("glove_area", 0) or 0)
    if glove_mask is None or glove_area <= 0 or glove_mask.ndim != 2:
        return _empty_result(None if glove_mask is None else glove_mask.shape)
    mask = np.where(glove_mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty_result(mask.shape)
    contour = max(contours, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(contour)
    if gw < 30 or gh < 50:
        return _empty_result(mask.shape)

    gray = None if processed is None else processed.get("gray_enhanced")
    if gray is None and processed is not None:
        gray = processed.get("gray")
    if gray is None or gray.shape[:2] != mask.shape:
        return _empty_result(mask.shape)

    bounds = (gx, gy, gw, gh)
    edges, canny_low, canny_high, edge_density = _adaptive_internal_edges(
        gray, mask, bounds)
    candidates = _candidates(edges, gray, mask, bounds)
    textured_glove = edge_density >= 0.10
    if textured_glove:
        best = _select_textured_candidate(candidates)
    else:
        best = candidates[0] if candidates else None

    score = 0.0 if best is None else float(best["score"])
    detected = bool(best is not None and score >= DETECTION_SCORE_THRESHOLD)
    if not detected:
        score = min(score, DETECTION_SCORE_THRESHOLD - 0.01)

    defect_mask = np.zeros_like(mask)
    box = None
    if detected:
        thickness = max(8, int(round(FOLD_OVERLAY_THICKNESS_RATIO * gw)))
        x1, y1, x2, y2 = best["line"]
        localisation_shift_x = 0
        localisation_shift_y = 0
        if textured_glove and best["vertical_fold"]:
            localisation_shift_x = int(round(COTTON_VERTICAL_X_SHIFT_RATIO * gw))
        elif textured_glove:
            localisation_shift_y = int(round(COTTON_HORIZONTAL_Y_SHIFT_RATIO * gh))
        x1, x2 = x1 + localisation_shift_x, x2 + localisation_shift_x
        y1, y2 = y1 + localisation_shift_y, y2 + localisation_shift_y
        cv2.line(
            defect_mask, (x1, y1), (x2, y2), 255,
            2 * thickness + 1, cv2.LINE_AA)
        defect_mask = cv2.bitwise_and(defect_mask, mask)
        points = cv2.findNonZero(defect_mask)
        if points is not None:
            x, y, w, h = cv2.boundingRect(points)
            box = (int(x), int(y), int(w), int(h))

    defect_pixels = int(np.count_nonzero(defect_mask))
    measurements = {
        "area_pct": round(100.0 * defect_pixels / float(glove_area), 3),
        "canny_low": int(canny_low),
        "canny_high": int(canny_high),
        "palm_edge_density": round(float(edge_density), 4),
        "candidate_count": int(len(candidates)),
        "textured_glove": bool(textured_glove),
    }
    if best is not None:
        measurements.update({
            "length_ratio": round(float(best["length_ratio"]), 4),
            "fold_angle_deg": round(float(best["angle"]), 2),
            "edge_continuity": round(float(best["continuity"]), 4),
            "paired_edge_support": round(float(best["paired_support"]), 4),
            "side_material_support": round(float(best["side_material"]), 4),
            "local_contrast": round(float(best["contrast"]), 4),
            "local_contrast_gray": round(float(best["contrast_gray"]), 2),
            "fold_relative_y": round(float(best["relative_y"]), 4),
            "side_fold": bool(best["side_fold"]),
            "vertical_fold": bool(best["vertical_fold"]),
            "line_top_ratio": round(float(best["line_top_ratio"]), 4),
            "line_bottom_ratio": round(float(best["line_bottom_ratio"]), 4),
            "centre_x_ratio": round(float(best["centre_x_ratio"]), 4),
            "textured_candidate_rank": round(
                float(best.get("textured_rank", 0.0)), 4),
            "lower_position_score": round(
                float(best.get("lower_position_score", 0.0)), 4),
            "vertical_preference": round(
                float(best.get("vertical_preference", 0.0)), 4),
            "localisation_shift_x_px": int(
                round(COTTON_VERTICAL_X_SHIFT_RATIO * gw)
                if textured_glove and best["vertical_fold"] else 0),
            "localisation_shift_y_px": int(
                round(COTTON_HORIZONTAL_Y_SHIFT_RATIO * gh)
                if textured_glove and not best["vertical_fold"] else 0),
        })

    return {
        "defect_name": "damaged_by_fold",
        "detected": bool(detected),
        "detection_score": float(np.clip(score, 0.0, 1.0)),
        "algorithm": ALGORITHM,
        "bounding_box": box,
        "mask": defect_mask,
        "measurements": measurements,
    }
