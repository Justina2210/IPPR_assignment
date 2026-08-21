"""Hybrid classical detector for joined/touching glove fingers.

Evidence is deliberately complementary:
1. silhouette topology: one of four normal finger valleys is missing;
2. a long internal seam lies inside the merged finger;
3. glove exists on both sides and the local lobe is wider than one finger.

This separation helps reject missing fingers (topology only), wrinkles/folds
(edge only), and the external glove boundary. OpenCV/Numpy only.
"""
import cv2
import numpy as np


# All values are scale-relative. TUNED-BY-EYE starting values from the six
# supplied touching images; validate them with fp_sweep.py after dataset merge.
FINGER_REGION_BOTTOM = 0.70
SIDE_MARGIN_RATIO = 0.035
EXPECTED_GAPS = 4
MIN_GAP_DEPTH_RATIO = 0.052
MAX_GAP_ANGLE_DEG = 115.0
MIN_GAP_SPACING_RATIO = 0.055

BOUNDARY_ERODE_RATIO = 0.012
MIN_SEAM_LENGTH_RATIO = 0.080
STRONG_SEAM_LENGTH_RATIO = 0.25
MIN_VERTICALITY = 0.50
MIN_SIDE_SUPPORT = 0.54
MIN_BOUNDARY_DISTANCE_RATIO = 0.012
MIN_EDGE_CONTINUITY = 0.36

MIN_MERGED_WIDTH_RATIO = 1.15
STRONG_MERGED_WIDTH_RATIO = 1.40
MAX_MERGED_WIDTH_RATIO = 2.80
MAX_UPPER_EDGE_DENSITY = 0.24
MAX_HOUGH_LINES_TO_ANALYSE = 60

DETECTION_SCORE_THRESHOLD = 0.55

ALGORITHM = (
    "Hybrid masked-glove geometry: missing interdigital valley plus an "
    "adaptive-edge internal seam with two-sided material and merged-finger width"
)


def _empty_result(shape=None):
    return {
        "defect_name": "touching",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": ALGORITHM,
        "bounding_box": None,
        "mask": None if shape is None else np.zeros(shape, dtype=np.uint8),
        "measurements": {},
    }


def _angle_deg(a, vertex, b):
    va = a.astype(np.float64) - vertex.astype(np.float64)
    vb = b.astype(np.float64) - vertex.astype(np.float64)
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denominator <= 1e-9:
        return 180.0
    cosine = float(np.clip(np.dot(va, vb) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _deduplicate_x(items, spacing):
    kept = []
    for item in sorted(items, key=lambda value: value["depth"], reverse=True):
        if all(abs(item["x"] - old["x"]) >= spacing for old in kept):
            kept.append(item)
    return sorted(kept, key=lambda value: value["x"])


def _topology(contour, bounds):
    gx, gy, gw, gh = bounds
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return {"gap_count": 0, "missing": EXPECTED_GAPS, "valleys": []}
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return {"gap_count": 0, "missing": EXPECTED_GAPS, "valleys": []}

    bottom = gy + FINGER_REGION_BOTTOM * gh
    margin = SIDE_MARGIN_RATIO * gw
    minimum_depth = MIN_GAP_DEPTH_RATIO * gh
    candidates = []
    for start_i, end_i, far_i, raw_depth in defects[:, 0, :]:
        start = contour[int(start_i), 0]
        end = contour[int(end_i), 0]
        far = contour[int(far_i), 0]
        x, y = int(far[0]), int(far[1])
        depth = float(raw_depth) / 256.0
        if not (gy < y < bottom and gx + margin < x < gx + gw - margin):
            continue
        if depth < minimum_depth or _angle_deg(start, far, end) > MAX_GAP_ANGLE_DEG:
            continue
        candidates.append({"x": x, "y": y, "depth": depth})

    spacing = max(4, int(round(MIN_GAP_SPACING_RATIO * gw)))
    valleys = _deduplicate_x(candidates, spacing)
    count = len(valleys)
    return {
        "gap_count": int(count),
        "missing": int(max(0, EXPECTED_GAPS - count)),
        "valleys": valleys,
    }


def _adaptive_edges(gray, mask, bounds):
    """Use masked gradient percentiles so cotton texture raises its own threshold."""
    gx, gy, gw, gh = bounds
    enhanced = cv2.GaussianBlur(gray, (5, 5), 0)
    sx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sx, sy)

    upper = np.zeros_like(mask)
    y2 = min(mask.shape[0], gy + int(round(FINGER_REGION_BOTTOM * gh)))
    upper[gy:y2, gx:gx + gw] = mask[gy:y2, gx:gx + gw]
    values = magnitude[upper > 0]
    if values.size < 50:
        return np.zeros_like(mask), upper, 0.0, 0, 0

    low = int(np.clip(np.percentile(values, 62), 8, 120))
    high = int(np.clip(np.percentile(values, 88), low + 10, 240))
    edges = cv2.Canny(enhanced, low, high)

    erosion = max(3, int(round(BOUNDARY_ERODE_RATIO * gw)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1,) * 2)
    interior = cv2.erode(mask, kernel)
    search = cv2.bitwise_and(interior, upper)
    edges = cv2.bitwise_and(edges, search)
    density = float(np.count_nonzero(edges) / max(np.count_nonzero(search), 1))
    return edges, search, density, low, high


def _samples(line, count=27):
    x1, y1, x2, y2 = line
    return [
        (int(round(x1 + t * (x2 - x1))), int(round(y1 + t * (y2 - y1))))
        for t in np.linspace(0.0, 1.0, count)
    ]


def _line_continuity(edges, line, radius):
    h, w = edges.shape
    hits = 0
    points = _samples(line, 31)
    for x, y in points:
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        hits += int(np.any(edges[y0:y1, x0:x1] > 0))
    return float(hits / max(len(points), 1))


def _seam_prominence(edges, line, glove_w):
    """Compare a seam with nearby parallel edges to suppress knitted ribs.

    Cotton texture produces several similarly strong parallel lines. A physical
    overlap boundary should be more continuous than lines shifted to either
    side of it.
    """
    x1, y1, x2, y2 = line
    dx, dy = float(x2 - x1), float(y2 - y1)
    length = float(np.hypot(dx, dy))
    if length <= 1.0:
        return 0.0
    nx, ny = -dy / length, dx / length
    radius = max(1, int(round(0.004 * glove_w)))
    centre = _line_continuity(edges, line, radius)
    nearby = []
    for ratio in (-0.040, -0.025, -0.014, 0.014, 0.025, 0.040):
        offset = ratio * glove_w
        shifted = (
            int(round(x1 + nx * offset)), int(round(y1 + ny * offset)),
            int(round(x2 + nx * offset)), int(round(y2 + ny * offset)),
        )
        nearby.append(_line_continuity(edges, shifted, radius))
    reference = float(np.percentile(nearby, 65)) if nearby else 0.0
    return float(np.clip((centre - reference) / max(1.0 - reference, 0.15), 0.0, 1.0))


def _side_support(mask, line, glove_w):
    x1, y1, x2, y2 = line
    dx, dy = float(x2 - x1), float(y2 - y1)
    length = float(np.hypot(dx, dy))
    if length <= 1.0:
        return 0.0
    nx, ny = -dy / length, dx / length
    # Inspect every pixel close to the line. Sparse far-away probes can jump
    # across a narrow background gap and land in the neighbouring finger,
    # causing an external finger edge to look like an internal overlap seam.
    near_limit = max(3, int(round(0.018 * glove_w)))
    far_limit = max(near_limit + 2, int(round(0.045 * glove_w)))
    near_probes = range(2, near_limit + 1)
    far_probes = np.linspace(near_limit + 1, far_limit, 3).astype(int)
    h, w = mask.shape
    supported, total = 0, 0
    for t in np.linspace(0.16, 0.84, 13):
        cx, cy = x1 + t * dx, y1 + t * dy
        near_ratios = []
        far_votes = []
        for sign in (1.0, -1.0):
            near_values = []
            for probe in near_probes:
                x = int(round(cx + sign * nx * probe))
                y = int(round(cy + sign * ny * probe))
                near_values.append(int(
                    0 <= x < w and 0 <= y < h and mask[y, x] > 0
                ))
            near_ratios.append(float(np.mean(near_values)) if near_values else 0.0)

            votes = 0
            for probe in far_probes:
                x = int(round(cx + sign * nx * int(probe)))
                y = int(round(cy + sign * ny * int(probe)))
                votes += int(0 <= x < w and 0 <= y < h and mask[y, x] > 0)
            far_votes.append(votes)
        total += 1
        # Both immediate sides must be almost uninterrupted glove. The looser
        # far test allows a true overlap seam that approaches a fingertip edge.
        supported += int(
            near_ratios[0] >= 0.88 and near_ratios[1] >= 0.88
            and far_votes[0] >= 1 and far_votes[1] >= 1
        )
    return float(supported / max(total, 1))


def _horizontal_runs(row):
    padded = np.pad((row > 0).astype(np.uint8), (1, 1))
    change = np.diff(padded.astype(np.int16))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1) - 1
    return [(int(a), int(b), int(b - a + 1)) for a, b in zip(starts, ends)]


def _typical_finger_width(mask, bounds):
    gx, gy, gw, gh = bounds
    widths = []
    for y in np.linspace(gy + 0.12 * gh, gy + 0.48 * gh, 14).astype(int):
        for _, _, width in _horizontal_runs(mask[y, gx:gx + gw]):
            if 0.045 * gw <= width <= 0.34 * gw:
                widths.append(width)
    return float(np.percentile(widths, 40)) if widths else float(0.16 * gw)


def _merged_width(mask, line, typical_width, glove_w):
    widths = []
    h, w = mask.shape
    for x, y in _samples(line, 11)[1:-1]:
        if not (0 <= x < w and 0 <= y < h and mask[y, x] > 0):
            continue
        left = x
        while left > 0 and mask[y, left - 1] > 0:
            left -= 1
        right = x
        while right + 1 < w and mask[y, right + 1] > 0:
            right += 1
        widths.append(right - left + 1)
    median = float(np.median(widths)) if widths else 0.0
    ratio = median / max(typical_width, 1.0)
    # Palm-crossing lines create implausibly huge widths and are rejected later.
    return median, float(ratio), bool(median <= 0.48 * glove_w)


def _seam_candidates(edges, mask, bounds):
    gx, gy, gw, gh = bounds
    minimum_length = max(18, int(round(MIN_SEAM_LENGTH_RATIO * gh)))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(12, int(round(0.022 * gh))),
        minLineLength=minimum_length,
        maxLineGap=max(7, int(round(0.025 * gh))),
    )
    if lines is None:
        return []

    # Cotton weave can create hundreds of Hough lines. Apply cheap orientation
    # and length checks first, then run costly side/width/prominence analysis on
    # only the longest plausible lines.
    prefiltered = []
    for raw in lines[:, 0]:
        line = tuple(int(value) for value in raw)
        x1, y1, x2, y2 = line
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        verticality = abs(dy) / max(length, 1.0)
        if length < minimum_length or verticality < MIN_VERTICALITY:
            continue
        prefiltered.append((line, length, verticality))

    prefiltered.sort(key=lambda item: item[1], reverse=True)
    prefiltered = prefiltered[:MAX_HOUGH_LINES_TO_ANALYSE]

    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    typical = _typical_finger_width(mask, bounds)
    candidates = []
    for line, length, verticality in prefiltered:
        x1, y1, x2, y2 = line

        distances = [distance[y, x] for x, y in _samples(line, 21)
                     if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]]
        boundary_distance = float(np.median(distances)) if distances else 0.0
        if boundary_distance < max(2.0, MIN_BOUNDARY_DISTANCE_RATIO * gw):
            continue

        side = _side_support(mask, line, gw)
        continuity = _line_continuity(edges, line, max(2, int(round(0.006 * gw))))
        width, width_ratio, plausible_width = _merged_width(mask, line, typical, gw)
        if side < MIN_SIDE_SUPPORT or continuity < MIN_EDGE_CONTINUITY or not plausible_width:
            continue
        # This is the most expensive candidate measurement, so compute it only
        # after the cheaper geometry checks have accepted the line.
        prominence = _seam_prominence(edges, line, gw)

        length_score = float(np.clip(
            (length / gh - MIN_SEAM_LENGTH_RATIO)
            / (STRONG_SEAM_LENGTH_RATIO - MIN_SEAM_LENGTH_RATIO), 0.0, 1.0))
        boundary_score = float(np.clip(boundary_distance / (0.05 * gw), 0.0, 1.0))
        width_score = float(np.clip(
            (width_ratio - MIN_MERGED_WIDTH_RATIO)
            / (STRONG_MERGED_WIDTH_RATIO - MIN_MERGED_WIDTH_RATIO), 0.0, 1.0))
        top_ratio = (min(y1, y2) - gy) / max(float(gh), 1.0)
        top_score = float(np.clip((0.50 - top_ratio) / 0.40, 0.0, 1.0))
        score = float(np.clip(
            0.16 * length_score + 0.12 * verticality + 0.15 * continuity
            + 0.14 * side + 0.08 * boundary_score + 0.22 * width_score
            + 0.04 * top_score + 0.09 * prominence, 0.0, 1.0))
        # An internal seam inside a genuinely broad lobe is more relevant than
        # a high-contrast wrinkle inside a normal-width finger.
        if width_ratio >= STRONG_MERGED_WIDTH_RATIO and side >= 0.68:
            score = min(1.0, score + 0.07)
        candidates.append({
            "line": line, "score": score, "length": length,
            "length_ratio": length / gh, "verticality": verticality,
            "continuity": continuity, "side_support": side,
            "prominence": prominence,
            "boundary_distance": boundary_distance, "typical_width": typical,
            "merged_width": width, "width_ratio": width_ratio,
        })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def detect_touching(processed, segmentation):
    glove_mask = None if segmentation is None else segmentation.get("glove_mask")
    glove_area = 0 if segmentation is None else int(segmentation.get("glove_area", 0) or 0)
    if glove_mask is None or glove_area <= 0 or glove_mask.ndim != 2:
        return _empty_result(None if glove_mask is None else glove_mask.shape)
    mask = np.where(glove_mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return _empty_result(mask.shape)
    contour = max(contours, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(contour)
    if gw < 30 or gh < 50:
        return _empty_result(mask.shape)
    bounds = (gx, gy, gw, gh)

    gray = None if processed is None else processed.get("gray_enhanced")
    if gray is None and processed is not None:
        gray = processed.get("gray")
    if gray is None or gray.shape[:2] != mask.shape:
        return _empty_result(mask.shape)

    topology = _topology(contour, bounds)
    edges, _, edge_density, canny_low, canny_high = _adaptive_edges(gray, mask, bounds)
    candidates = _seam_candidates(edges, mask, bounds)
    textured_glove = edge_density >= 0.10
    if textured_glove:
        # Rank only candidates that stand out from the weave and occur inside a
        # widened lobe. Do not let the strongest ordinary knitted rib prevent a
        # slightly weaker but physically plausible overlap seam from winning.
        eligible = [
            item for item in candidates
            if item["prominence"] >= 0.04 and item["width_ratio"] >= 1.35
        ]
        best = eligible[0] if eligible else None
    else:
        best = candidates[0] if candidates else None

    gap_count = topology["gap_count"]
    exactly_one_missing = gap_count == EXPECTED_GAPS - 1
    multiple_missing = gap_count <= EXPECTED_GAPS - 2
    seam_score = 0.0 if best is None else float(best["score"])
    width_ratio = 0.0 if best is None else float(best["width_ratio"])

    # Dense cotton weave or many wrinkles yield numerous competing Hough lines.
    texture_penalty = float(np.clip(
        (edge_density - 0.10) / (MAX_UPPER_EDGE_DENSITY - 0.10), 0.0, 1.0))
    multiplicity_penalty = float(np.clip((len(candidates) - 4) / 8.0, 0.0, 1.0))
    adjusted_seam = float(np.clip(
        seam_score - 0.16 * texture_penalty - 0.10 * multiplicity_penalty, 0.0, 1.0))

    # On a highly textured glove, accept only a locally unique seam situated
    # inside a clearly broad lobe. This removes straight knitted ribs while
    # retaining the darker boundary created by two overlapping fingers.
    texture_localisation_ok = bool(
        best is not None and (
            not textured_glove
            or (best["prominence"] >= 0.04 and width_ratio >= 1.35)
        )
    )

    normal_case = bool(
        best is not None and texture_localisation_ok
        and exactly_one_missing and adjusted_seam >= 0.36
        and 1.05 <= width_ratio <= MAX_MERGED_WIDTH_RATIO)
    strong_seam_case = bool(
        best is not None and texture_localisation_ok and adjusted_seam >= 0.68
        and 1.18 <= width_ratio <= MAX_MERGED_WIDTH_RATIO
        and best["side_support"] >= 0.68 and best["continuity"] >= 0.48)
    severe_overlap_case = bool(
        best is not None and texture_localisation_ok
        and multiple_missing and gap_count >= 2
        and adjusted_seam >= 0.62 and width_ratio >= STRONG_MERGED_WIDTH_RATIO)

    topology_score = 1.0 if exactly_one_missing else (0.58 if multiple_missing else 0.0)
    width_score = float(np.clip(
        (width_ratio - 1.0) / (STRONG_MERGED_WIDTH_RATIO - 1.0), 0.0, 1.0))
    raw_score = float(np.clip(
        0.43 * topology_score + 0.43 * adjusted_seam + 0.14 * width_score,
        0.0, 1.0))
    detected = bool(normal_case or strong_seam_case or severe_overlap_case)
    score = raw_score if detected else min(raw_score, DETECTION_SCORE_THRESHOLD - 0.01)

    defect_mask = np.zeros_like(mask)
    box = None
    if detected and best is not None:
        radius = max(7, int(round(0.032 * gw)))
        x1, y1, x2, y2 = best["line"]
        cv2.line(defect_mask, (x1, y1), (x2, y2), 255, 2 * radius + 1, cv2.LINE_AA)
        defect_mask = cv2.bitwise_and(defect_mask, mask)
        points = cv2.findNonZero(defect_mask)
        if points is not None:
            x, y, w, h = cv2.boundingRect(points)
            box = (int(x), int(y), int(w), int(h))

    area_pct = 100.0 * np.count_nonzero(defect_mask) / float(glove_area)
    measurements = {
        "area_pct": round(float(area_pct), 3),
        "deep_finger_gap_count": int(gap_count),
        "missing_gap_count": int(topology["missing"]),
        "upper_edge_density": round(float(edge_density), 4),
        "canny_low": int(canny_low),
        "canny_high": int(canny_high),
        "seam_candidates": int(len(candidates)),
        "adjusted_seam_score": round(float(adjusted_seam), 4),
        "texture_penalty": round(float(texture_penalty), 4),
        "textured_glove": bool(textured_glove),
        "texture_localisation_ok": bool(texture_localisation_ok),
        "merged_width_ratio": round(float(width_ratio), 4),
        "decision_path": (
            "one_missing_gap_plus_seam" if normal_case else
            "very_strong_seam" if strong_seam_case else
            "severe_overlap" if severe_overlap_case else "rejected"
        ),
    }
    if best is not None:
        measurements.update({
            "seam_length_ratio": round(float(best["length_ratio"]), 4),
            "seam_verticality": round(float(best["verticality"]), 4),
            "seam_continuity": round(float(best["continuity"]), 4),
            "seam_prominence": round(float(best["prominence"]), 4),
            "seam_side_support": round(float(best["side_support"]), 4),
            "boundary_distance_px": round(float(best["boundary_distance"]), 2),
        })

    return {
        "defect_name": "touching",
        "detected": bool(detected),
        "detection_score": float(np.clip(score, 0.0, 1.0)),
        "algorithm": ALGORITHM,
        "bounding_box": box,
        "mask": defect_mask,
        "measurements": measurements,
    }
