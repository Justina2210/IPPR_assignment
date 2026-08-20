"""
touching.py
-----------
Self-contained detector for the Touching / Overlapping Fingers defect.

Target defect
-------------
This detector is designed for cases where one glove finger overlaps another,
so the outer silhouette may look like ONE broad finger while an internal seam
is visible inside the merged finger region.

It does NOT depend on finger_not_enough.py.

Main evidence
-------------
1. Detect long vertical/diagonal INTERNAL seam lines inside the upper glove.
2. Require the seam to be safely away from the outer glove boundary.
3. Require glove material to exist on BOTH sides of the seam.
4. Measure the local merged-finger width around the seam.
5. Prefer seams that begin high in the finger region and extend downward.
6. Use silhouette/top-profile evidence only as supporting evidence.
7. Require a precise local overlap candidate before reporting Touching.

The function follows the shared detector contract used by evaluate.py:

    detect_touching(processed: dict, segmentation: dict) -> dict
"""

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DETECTION_THRESHOLD = 0.50

# Region of the glove used for overlap search.
SEARCH_TOP_FRAC = 0.02
SEARCH_BOTTOM_FRAC = 0.68
SEARCH_LEFT_FRAC = 0.06
SEARCH_RIGHT_FRAC = 0.94

# Erode the mask so the external glove outline cannot become an "internal seam".
BOUNDARY_ERODE_FRAC = 0.012

# Canny / Hough seam settings.
CANNY_LOW = 25
CANNY_HIGH = 80

MIN_SEAM_LENGTH_FRAC = 0.11
STRONG_SEAM_LENGTH_FRAC = 0.24

MIN_VERTICALITY = 0.60
MAX_HORIZONTAL_SLOPE = 1.15

# Internal seam must have enough distance from the external glove boundary.
MIN_BOUNDARY_DISTANCE_FRAC = 0.012

# Glove material should exist on both sides of a true overlap seam.
SIDE_PROBE_MIN_FRAC = 0.018
SIDE_PROBE_MAX_FRAC = 0.060
MIN_SIDE_SUPPORT = 0.62

# Candidate scoring.
MIN_CANDIDATE_SCORE = 0.43

# Top-profile smoothing / fingertip estimation.
PROFILE_SMOOTH_FRAC = 0.018
PEAK_MIN_SPACING_FRAC = 0.075
MAX_FINGERTIPS = 5


# ============================================================
# RESULT / BASIC HELPERS
# ============================================================

def _empty_result():
    return {
        "defect_name": "touching",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": (
            "overlapping-finger internal seam + merged-finger width "
            "+ upper-silhouette geometry"
        ),
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def _largest_contour(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return max(contours, key=cv2.contourArea) if contours else None


def _smooth_1d(values, window):
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return values.copy()

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1

    if window >= values.size:
        window = values.size - 1 if values.size % 2 == 0 else values.size
        window = max(3, window)

    if window > values.size:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)

    return np.convolve(padded, kernel, mode="valid")


def _binary_mask(mask):
    return (mask > 0).astype(np.uint8) * 255


def _glove_bounds(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None

    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())

    return (
        x0,
        y0,
        x1,
        y1,
        max(1, x1 - x0 + 1),
        max(1, y1 - y0 + 1),
    )


# ============================================================
# TOP PROFILE / FINGERTIP SUPPORT
# ============================================================

def _top_profile(mask, x0, x1, y0, y1):
    """
    Return the topmost glove y-coordinate for every x-column.

    Missing columns are interpolated to avoid tiny segmentation holes causing
    artificial silhouette peaks.
    """
    xs = np.arange(x0, x1 + 1, dtype=np.int32)
    profile = np.full(len(xs), np.nan, dtype=np.float32)

    for index, x in enumerate(xs):
        ys = np.flatnonzero(mask[y0:y1 + 1, x] > 0)
        if ys.size:
            profile[index] = float(y0 + ys[0])

    valid = np.flatnonzero(np.isfinite(profile))
    if valid.size < 2:
        return xs, profile

    missing = np.flatnonzero(~np.isfinite(profile))
    if missing.size:
        profile[missing] = np.interp(
            missing,
            valid,
            profile[valid],
        )

    return xs, profile


def _find_fingertips(mask, x0, x1, y0, glove_h, glove_w):
    """
    Self-contained fingertip detector.

    Fingertips are local minima of the upper silhouette because y increases
    downward in image coordinates.

    This is supporting evidence only. The overlap detector does NOT require
    five visible fingertip peaks because overlapping fingers can merge into one
    outer silhouette.
    """
    search_bottom = min(
        mask.shape[0] - 1,
        int(round(y0 + 0.60 * glove_h)),
    )

    xs, profile = _top_profile(
        mask,
        x0,
        x1,
        y0,
        search_bottom,
    )

    if len(profile) < 7 or np.count_nonzero(np.isfinite(profile)) < 7:
        return [], profile, profile

    smooth_window = max(
        5,
        int(round(PROFILE_SMOOTH_FRAC * glove_w)),
    )
    smooth = _smooth_1d(profile, smooth_window)

    upper_limit = y0 + 0.45 * glove_h
    prominence_radius = max(
        5,
        int(round(0.050 * glove_w)),
    )

    candidates = []

    for i in range(1, len(smooth) - 1):
        y = smooth[i]

        if y > upper_limit:
            continue

        is_minimum = (
            y <= smooth[i - 1]
            and y < smooth[i + 1]
        )
        if not is_minimum:
            continue

        lo = max(0, i - prominence_radius)
        hi = min(len(smooth), i + prominence_radius + 1)

        left_reference = float(
            np.percentile(smooth[lo:i + 1], 75)
        )
        right_reference = float(
            np.percentile(smooth[i:hi], 75)
        )

        prominence = min(
            left_reference,
            right_reference,
        ) - float(y)

        if prominence < max(2.0, 0.010 * glove_h):
            continue

        candidates.append({
            "x": int(xs[i]),
            "y": int(round(y)),
            "prominence": float(prominence),
        })

    if not candidates:
        return [], profile, smooth

    min_spacing = max(
        8,
        int(round(PEAK_MIN_SPACING_FRAC * glove_w)),
    )

    selected = []

    for candidate in sorted(
        candidates,
        key=lambda item: item["prominence"],
        reverse=True,
    ):
        if any(
            abs(candidate["x"] - chosen["x"]) < min_spacing
            for chosen in selected
        ):
            continue

        selected.append(candidate)

        if len(selected) >= MAX_FINGERTIPS:
            break

    selected.sort(key=lambda item: item["x"])

    peaks = [
        (item["x"], item["y"])
        for item in selected
    ]

    return peaks, profile, smooth


# ============================================================
# HORIZONTAL RUN / MERGED-FINGER WIDTH
# ============================================================

def _horizontal_run(mask, x, y):
    """
    Return the foreground run containing (x, y).

    Result:
        (left_x, right_x, width)

    Returns None if point is outside foreground.
    """
    h, w = mask.shape[:2]

    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    if mask[y, x] == 0:
        return None

    left = x
    while left > 0 and mask[y, left - 1] > 0:
        left -= 1

    right = x
    while right < w - 1 and mask[y, right + 1] > 0:
        right += 1

    return left, right, right - left + 1


def _estimate_typical_upper_run_width(
    mask,
    x0,
    x1,
    y0,
    glove_h,
    glove_w,
):
    """
    Estimate a typical single-finger horizontal width.

    The function samples connected foreground runs in several upper rows and
    keeps widths that look finger-sized rather than full-palm-sized.
    """
    widths = []

    y_start = int(round(y0 + 0.12 * glove_h))
    y_end = int(round(y0 + 0.48 * glove_h))

    if y_end <= y_start:
        return max(1.0, 0.16 * glove_w)

    for y in np.linspace(y_start, y_end, 14).astype(int):
        row = mask[y, x0:x1 + 1] > 0

        if not np.any(row):
            continue

        padded = np.pad(row.astype(np.uint8), (1, 1))
        diff = np.diff(padded.astype(np.int16))

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        for start, end in zip(starts, ends):
            width = int(end - start + 1)

            if (
                0.045 * glove_w
                <= width
                <= 0.34 * glove_w
            ):
                widths.append(width)

    if not widths:
        return max(1.0, 0.16 * glove_w)

    # Lower percentile is intentional: merged fingers are wider than individual
    # fingers, while palm-connected runs can be much wider.
    return float(np.percentile(widths, 40))


def _merged_width_support(
    mask,
    line,
    typical_width,
    glove_w,
):
    """
    Measure whether the candidate seam lies inside an unusually broad finger.

    A true overlap often looks like one broad outer finger with a seam inside.
    """
    x1, y1, x2, y2 = line

    sample_count = 9
    widths = []

    for t in np.linspace(0.10, 0.90, sample_count):
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))

        run = _horizontal_run(mask, x, y)
        if run is None:
            continue

        widths.append(float(run[2]))

    if not widths:
        return {
            "median_width": 0.0,
            "width_ratio": 0.0,
            "width_score": 0.0,
            "sample_count": 0,
        }

    median_width = float(np.median(widths))
    width_ratio = median_width / max(float(typical_width), 1.0)

    # Around 1.35x+ typical finger width starts becoming suspicious.
    width_score = float(np.clip(
        (width_ratio - 1.20) / 0.75,
        0.0,
        1.0,
    ))

    # Very huge runs are likely already in the palm. Penalise them.
    if median_width > 0.48 * glove_w:
        width_score *= 0.45

    return {
        "median_width": median_width,
        "width_ratio": float(width_ratio),
        "width_score": float(width_score),
        "sample_count": len(widths),
    }


# ============================================================
# INTERNAL SEAM EXTRACTION
# ============================================================

def _build_internal_edge_map(
    processed,
    mask,
    x0,
    y0,
    glove_w,
    glove_h,
):
    """
    Build an edge map containing only INTERNAL glove edges.

    The glove boundary is eroded away so the external silhouette cannot be
    detected as a touching seam.
    """
    gray = None

    if processed is not None:
        gray = processed.get("gray")

    if gray is None or gray.shape[:2] != mask.shape[:2]:
        return None, None, None

    boundary_margin = max(
        3,
        int(round(BOUNDARY_ERODE_FRAC * glove_w)),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * boundary_margin + 1,
            2 * boundary_margin + 1,
        ),
    )

    interior_mask = cv2.erode(mask, kernel)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        CANNY_LOW,
        CANNY_HIGH,
    )

    edges = cv2.bitwise_and(
        edges,
        interior_mask,
    )

    # Keep only upper glove / fingers.
    search_mask = np.zeros_like(mask)

    sx1 = max(
        0,
        int(round(x0 + SEARCH_LEFT_FRAC * glove_w)),
    )
    sx2 = min(
        mask.shape[1],
        int(round(x0 + SEARCH_RIGHT_FRAC * glove_w)),
    )
    sy1 = max(
        0,
        int(round(y0 + SEARCH_TOP_FRAC * glove_h)),
    )
    sy2 = min(
        mask.shape[0],
        int(round(y0 + SEARCH_BOTTOM_FRAC * glove_h)),
    )

    search_mask[sy1:sy2, sx1:sx2] = 255

    edges = cv2.bitwise_and(
        edges,
        search_mask,
    )

    # Distance transform gives distance from every glove pixel to background.
    distance = cv2.distanceTransform(
        (mask > 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    return edges, interior_mask, distance


def _line_verticality(line):
    x1, y1, x2, y2 = line

    dx = float(x2 - x1)
    dy = float(y2 - y1)

    length = float(np.hypot(dx, dy))

    if length <= 1e-6:
        return 0.0, 0.0

    verticality = abs(dy) / length
    horizontal_slope = abs(dx) / max(abs(dy), 1.0)

    return float(verticality), float(horizontal_slope)


def _line_samples(line, count=25):
    x1, y1, x2, y2 = line

    samples = []

    for t in np.linspace(0.0, 1.0, count):
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))
        samples.append((x, y))

    return samples


def _boundary_distance_score(
    distance_map,
    line,
    glove_w,
):
    """
    Ensure the seam is internal rather than part of the external outline.
    """
    if distance_map is None:
        return 0.0, 0.0

    values = []

    h, w = distance_map.shape[:2]

    for x, y in _line_samples(line, 21):
        if 0 <= x < w and 0 <= y < h:
            values.append(float(distance_map[y, x]))

    if not values:
        return 0.0, 0.0

    median_distance = float(np.median(values))

    minimum_required = max(
        2.0,
        MIN_BOUNDARY_DISTANCE_FRAC * glove_w,
    )

    score = float(np.clip(
        median_distance / max(2.5 * minimum_required, 1.0),
        0.0,
        1.0,
    ))

    return median_distance, score


def _side_material_support(
    mask,
    line,
    glove_w,
):
    """
    Check whether glove material exists on both sides of the seam.

    This is a very important discriminator:
    - outer silhouette: material exists on only one side
    - internal overlap seam: material exists on both sides
    """
    x1, y1, x2, y2 = line

    dx = float(x2 - x1)
    dy = float(y2 - y1)

    length = float(np.hypot(dx, dy))
    if length <= 1.0:
        return 0.0

    # Unit normal perpendicular to line.
    nx = -dy / length
    ny = dx / length

    min_probe = max(
        3,
        int(round(SIDE_PROBE_MIN_FRAC * glove_w)),
    )
    max_probe = max(
        min_probe,
        int(round(SIDE_PROBE_MAX_FRAC * glove_w)),
    )

    probe_distances = np.linspace(
        min_probe,
        max_probe,
        3,
    )

    h, w = mask.shape[:2]

    supported = 0
    total = 0

    # Skip line ends because they can approach a fingertip boundary.
    for t in np.linspace(0.16, 0.84, 13):
        cx = x1 + t * dx
        cy = y1 + t * dy

        left_votes = 0
        right_votes = 0

        for probe in probe_distances:
            lx = int(round(cx + nx * probe))
            ly = int(round(cy + ny * probe))

            rx = int(round(cx - nx * probe))
            ry = int(round(cy - ny * probe))

            if (
                0 <= lx < w
                and 0 <= ly < h
                and mask[ly, lx] > 0
            ):
                left_votes += 1

            if (
                0 <= rx < w
                and 0 <= ry < h
                and mask[ry, rx] > 0
            ):
                right_votes += 1

        total += 1

        if (
            left_votes >= 2
            and right_votes >= 2
        ):
            supported += 1

    if total == 0:
        return 0.0

    return float(supported / total)


def _edge_continuity_score(
    edges,
    line,
    glove_w,
):
    """
    Measure how continuously edge pixels follow the Hough seam.
    """
    if edges is None:
        return 0.0

    radius = max(
        2,
        int(round(0.006 * glove_w)),
    )

    h, w = edges.shape[:2]

    hits = 0
    total = 0

    for x, y in _line_samples(line, 31):
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)

        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)

        total += 1

        if np.any(edges[y1:y2, x1:x2] > 0):
            hits += 1

    if total == 0:
        return 0.0

    return float(hits / total)


def _top_position_score(
    line,
    y0,
    glove_h,
):
    """
    True overlap seams should begin high in the finger region.
    """
    _, y1, _, y2 = line
    line_top = min(y1, y2)

    relative_top = (
        line_top - y0
    ) / max(float(glove_h), 1.0)

    return float(np.clip(
        (0.48 - relative_top) / 0.38,
        0.0,
        1.0,
    ))


def _line_length_score(
    line,
    glove_h,
):
    x1, y1, x2, y2 = line

    length = float(
        np.hypot(
            x2 - x1,
            y2 - y1,
        )
    )

    score = float(np.clip(
        (
            length
            - MIN_SEAM_LENGTH_FRAC * glove_h
        )
        / max(
            (
                STRONG_SEAM_LENGTH_FRAC
                - MIN_SEAM_LENGTH_FRAC
            ) * glove_h,
            1.0,
        ),
        0.0,
        1.0,
    ))

    return length, score


def _detect_overlap_seams(
    processed,
    mask,
    x0,
    y0,
    glove_w,
    glove_h,
):
    """
    Find and rank long internal seam lines caused by overlapping fingers.
    """
    edges, interior_mask, distance = _build_internal_edge_map(
        processed,
        mask,
        x0,
        y0,
        glove_w,
        glove_h,
    )

    if edges is None:
        return [], None

    min_line_length = max(
        18,
        int(round(MIN_SEAM_LENGTH_FRAC * glove_h)),
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(
            12,
            int(round(0.022 * glove_h)),
        ),
        minLineLength=min_line_length,
        maxLineGap=max(
            7,
            int(round(0.025 * glove_h)),
        ),
    )

    if lines is None:
        return [], edges

    typical_width = _estimate_typical_upper_run_width(
        mask,
        x0,
        x0 + glove_w - 1,
        y0,
        glove_h,
        glove_w,
    )

    candidates = []

    for raw in lines[:, 0]:
        line = tuple(int(v) for v in raw)

        verticality, horizontal_slope = _line_verticality(
            line
        )

        if verticality < MIN_VERTICALITY:
            continue

        if horizontal_slope > MAX_HORIZONTAL_SLOPE:
            continue

        length, length_score = _line_length_score(
            line,
            glove_h,
        )

        if length < MIN_SEAM_LENGTH_FRAC * glove_h:
            continue

        boundary_distance, boundary_score = _boundary_distance_score(
            distance,
            line,
            glove_w,
        )

        if (
            boundary_distance
            < max(
                2.0,
                MIN_BOUNDARY_DISTANCE_FRAC * glove_w,
            )
        ):
            continue

        side_support = _side_material_support(
            mask,
            line,
            glove_w,
        )

        if side_support < MIN_SIDE_SUPPORT:
            continue

        continuity = _edge_continuity_score(
            edges,
            line,
            glove_w,
        )

        top_score = _top_position_score(
            line,
            y0,
            glove_h,
        )

        width_info = _merged_width_support(
            mask,
            line,
            typical_width,
            glove_w,
        )

        # A real overlap seam should be:
        # - long
        # - vertical/diagonal
        # - continuous
        # - internal
        # - surrounded by material on both sides
        # - preferably inside a broad/merged finger region
        geometry_score = float(np.clip(
            0.22 * length_score
            + 0.16 * verticality
            + 0.18 * continuity
            + 0.14 * boundary_score
            + 0.16 * side_support
            + 0.08 * top_score
            + 0.06 * width_info["width_score"],
            0.0,
            1.0,
        ))

        # Strong bonus for a genuinely broad merged finger.
        if (
            width_info["width_ratio"] >= 1.45
            and side_support >= 0.75
            and continuity >= 0.55
        ):
            geometry_score = min(
                1.0,
                geometry_score + 0.10,
            )

        point = (
            int(round(0.5 * (line[0] + line[2]))),
            int(round(0.5 * (line[1] + line[3]))),
        )

        candidates.append({
            "source": "overlap_internal_seam",
            "line": line,
            "point": point,
            "length": float(length),
            "length_ratio": float(
                length / max(float(glove_h), 1.0)
            ),
            "verticality": float(verticality),
            "edge_continuity": float(continuity),
            "boundary_distance": float(boundary_distance),
            "boundary_score": float(boundary_score),
            "side_support": float(side_support),
            "top_position_score": float(top_score),
            "typical_finger_width": float(typical_width),
            "local_merged_width": float(
                width_info["median_width"]
            ),
            "merged_width_ratio": float(
                width_info["width_ratio"]
            ),
            "merged_width_score": float(
                width_info["width_score"]
            ),
            "geometry_score": geometry_score,
        })

    candidates.sort(
        key=lambda item: item["geometry_score"],
        reverse=True,
    )

    return candidates[:12], edges


# ============================================================
# LOCAL DARK-SEAM / GRADIENT CONFIRMATION
# ============================================================

def _local_seam_strength(
    processed,
    mask,
    line,
    glove_w,
):
    """
    Compare gradient strength around the seam with the glove interior.

    This is supporting evidence only because wrinkles can also have gradients.
    """
    gray = None

    if processed is not None:
        gray = processed.get("gray")

    if gray is None or gray.shape[:2] != mask.shape[:2]:
        return 0.0

    margin = max(
        2,
        int(round(0.010 * glove_w)),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * margin + 1,
            2 * margin + 1,
        ),
    )

    interior = cv2.erode(
        mask,
        kernel,
    ) > 0

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    gx = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = cv2.magnitude(
        gx,
        gy,
    )

    glove_values = gradient[interior]

    if glove_values.size == 0:
        return 0.0

    reference = max(
        float(
            np.percentile(
                glove_values,
                90,
            )
        ),
        1.0,
    )

    radius = max(
        2,
        int(round(0.008 * glove_w)),
    )

    seam_values = []

    h, w = gray.shape[:2]

    for x, y in _line_samples(line, 35):
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)

        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)

        patch_mask = interior[y1:y2, x1:x2]

        if not np.any(patch_mask):
            continue

        patch_gradient = gradient[y1:y2, x1:x2]
        seam_values.extend(
            patch_gradient[patch_mask].tolist()
        )

    if not seam_values:
        return 0.0

    local_strength = float(
        np.percentile(
            seam_values,
            78,
        )
    )

    return float(np.clip(
        local_strength / reference,
        0.0,
        1.0,
    ))


def _dark_seam_score(
    processed,
    mask,
    line,
    glove_w,
):
    """
    Reward a darker line relative to surrounding glove material.

    This especially helps white/cream cotton and latex examples like the
    provided overlapping-finger image. It is kept low-weight so coloured
    nitrile gloves do not depend on it.
    """
    gray = None

    if processed is not None:
        gray = processed.get("gray")

    if gray is None or gray.shape[:2] != mask.shape[:2]:
        return 0.0

    x1, y1, x2, y2 = line

    dx = float(x2 - x1)
    dy = float(y2 - y1)

    length = float(np.hypot(dx, dy))

    if length <= 1.0:
        return 0.0

    nx = -dy / length
    ny = dx / length

    seam_radius = max(
        1,
        int(round(0.004 * glove_w)),
    )

    side_offset = max(
        4,
        int(round(0.025 * glove_w)),
    )

    h, w = gray.shape[:2]

    seam_vals = []
    side_vals = []

    for t in np.linspace(0.15, 0.85, 21):
        cx = x1 + t * dx
        cy = y1 + t * dy

        for r in range(-seam_radius, seam_radius + 1):
            sx = int(round(cx + nx * r))
            sy = int(round(cy + ny * r))

            if (
                0 <= sx < w
                and 0 <= sy < h
                and mask[sy, sx] > 0
            ):
                seam_vals.append(
                    float(gray[sy, sx])
                )

        for sign in (-1.0, 1.0):
            sx = int(
                round(
                    cx
                    + sign
                    * nx
                    * side_offset
                )
            )
            sy = int(
                round(
                    cy
                    + sign
                    * ny
                    * side_offset
                )
            )

            if (
                0 <= sx < w
                and 0 <= sy < h
                and mask[sy, sx] > 0
            ):
                side_vals.append(
                    float(gray[sy, sx])
                )

    if (
        len(seam_vals) < 8
        or len(side_vals) < 8
    ):
        return 0.0

    seam_mean = float(
        np.mean(seam_vals)
    )
    side_mean = float(
        np.mean(side_vals)
    )

    darkness = side_mean - seam_mean

    return float(np.clip(
        darkness / 32.0,
        0.0,
        1.0,
    ))


# ============================================================
# CONVEXITY / SILHOUETTE SUPPORT
# ============================================================

def _upper_convexity_features(
    mask,
    x0,
    y0,
    glove_w,
    glove_h,
):
    """
    Compute weak silhouette support.

    Overlapping fingers often remove one normal interdigital concavity, causing
    slightly fewer deep upper convexity defects and slightly higher solidity.
    These features are deliberately low-weight because segmentation errors can
    also change them.
    """
    hand_bottom = min(
        mask.shape[0],
        int(round(y0 + 0.72 * glove_h)),
    )

    roi = np.zeros_like(mask)
    roi[
        y0:hand_bottom,
        x0:x0 + glove_w
    ] = mask[
        y0:hand_bottom,
        x0:x0 + glove_w
    ]

    contour = _largest_contour(roi)

    if contour is None or len(contour) < 4:
        return {
            "deep_valleys": 0,
            "moderate_valleys": 0,
            "solidity": 0.0,
            "silhouette_signal": 0.0,
        }

    hull_points = cv2.convexHull(
        contour
    )

    contour_area = float(
        cv2.contourArea(
            contour
        )
    )

    hull_area = max(
        float(
            cv2.contourArea(
                hull_points
            )
        ),
        1.0,
    )

    solidity = (
        contour_area
        / hull_area
    )

    hull_index = cv2.convexHull(
        contour,
        returnPoints=False,
    )

    moderate = 0
    deep = 0

    if (
        hull_index is not None
        and len(hull_index) >= 4
    ):
        defects = cv2.convexityDefects(
            contour,
            hull_index,
        )

        if defects is not None:
            for d in defects[:, 0]:
                _, _, far_index, depth_raw = map(
                    int,
                    d,
                )

                far = contour[
                    far_index
                ][0]

                fx = float(far[0])
                fy = float(far[1])

                if not (
                    x0 + 0.06 * glove_w
                    <= fx
                    <= x0 + 0.94 * glove_w
                ):
                    continue

                if not (
                    y0 + 0.08 * glove_h
                    <= fy
                    <= y0 + 0.62 * glove_h
                ):
                    continue

                depth = (
                    float(depth_raw)
                    / 256.0
                )

                if depth >= 0.035 * glove_w:
                    moderate += 1

                if depth >= 0.065 * glove_w:
                    deep += 1

    # Three main upper-finger valleys are more stable than requiring the
    # anatomically different thumb/index valley.
    missing_valley_signal = float(np.clip(
        (3.0 - deep) / 3.0,
        0.0,
        1.0,
    ))

    solidity_signal = float(np.clip(
        (solidity - 0.74) / 0.18,
        0.0,
        1.0,
    ))

    silhouette_signal = float(
        0.68 * missing_valley_signal
        + 0.32 * solidity_signal
    )

    return {
        "deep_valleys": int(deep),
        "moderate_valleys": int(moderate),
        "solidity": float(solidity),
        "silhouette_signal": float(
            silhouette_signal
        ),
    }


# ============================================================
# CANDIDATE MERGING / DE-DUPLICATION
# ============================================================

def _line_angle_deg(line):
    x1, y1, x2, y2 = line
    return float(
        np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1,
            )
        )
    )


def _candidate_similarity(a, b, glove_w, glove_h):
    """
    Decide whether two Hough lines likely represent the same physical seam.
    """
    ax, ay = a["point"]
    bx, by = b["point"]

    point_distance = float(
        np.hypot(
            ax - bx,
            ay - by,
        )
    )

    if point_distance > 0.10 * max(
        glove_w,
        glove_h,
    ):
        return False

    angle_a = _line_angle_deg(
        a["line"]
    )
    angle_b = _line_angle_deg(
        b["line"]
    )

    angle_difference = abs(
        angle_a - angle_b
    )

    angle_difference = min(
        angle_difference,
        180.0 - angle_difference,
    )

    return angle_difference <= 18.0


def _deduplicate_candidates(
    candidates,
    glove_w,
    glove_h,
):
    """
    Keep one strongest representation of each physical internal seam.
    """
    kept = []

    for candidate in candidates:
        duplicate = False

        for existing in kept:
            if _candidate_similarity(
                candidate,
                existing,
                glove_w,
                glove_h,
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


# ============================================================
# DEFECT LOCALISATION
# ============================================================

def _line_region(
    mask,
    line,
    glove_w,
):
    """
    Produce a narrow defect mask around the overlap seam.
    """
    x1, y1, x2, y2 = line

    radius = max(
        7,
        int(round(0.032 * glove_w)),
    )

    region = np.zeros_like(mask)

    cv2.line(
        region,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        255,
        thickness=2 * radius + 1,
        lineType=cv2.LINE_AA,
    )

    region = cv2.bitwise_and(
        region,
        mask,
    )

    ys, xs = np.where(
        region > 0
    )

    if xs.size == 0:
        return (
            region,
            None,
            radius,
        )

    bx1 = int(xs.min())
    bx2 = int(xs.max())
    by1 = int(ys.min())
    by2 = int(ys.max())

    bbox = (
        bx1,
        by1,
        bx2 - bx1 + 1,
        by2 - by1 + 1,
    )

    return (
        region,
        bbox,
        radius,
    )


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_touching(
    processed: dict,
    segmentation: dict,
) -> dict:
    """
    Detect Touching where one glove finger overlaps another.

    The detector intentionally does not require two separate fingertip peaks.
    The strongest condition is a long INTERNAL seam surrounded by glove
    material on both sides in the upper finger region.
    """
    result = _empty_result()

    if segmentation is None:
        return result

    mask = segmentation.get(
        "glove_mask"
    )

    if (
        mask is None
        or np.count_nonzero(mask) < 500
    ):
        return result

    mask = _binary_mask(
        mask
    )

    bounds = _glove_bounds(
        mask
    )

    if bounds is None:
        return result

    (
        x0,
        y0,
        x1,
        y1,
        glove_w,
        glove_h,
    ) = bounds

    # --------------------------------------------------------
    # 1. Self-contained fingertip support
    # --------------------------------------------------------
    fingertip_peaks, _, _ = _find_fingertips(
        mask,
        x0,
        x1,
        y0,
        glove_h,
        glove_w,
    )

    # Missing one visible outer fingertip can support overlap, but never
    # decides the result by itself.
    peak_count = len(
        fingertip_peaks
    )

    merged_peak_signal = float(np.clip(
        (5.0 - peak_count) / 2.0,
        0.0,
        1.0,
    ))

    # --------------------------------------------------------
    # 2. Silhouette support
    # --------------------------------------------------------
    silhouette = _upper_convexity_features(
        mask,
        x0,
        y0,
        glove_w,
        glove_h,
    )

    # --------------------------------------------------------
    # 3. Find internal overlap seams
    # --------------------------------------------------------
    seam_candidates, edge_map = _detect_overlap_seams(
        processed,
        mask,
        x0,
        y0,
        glove_w,
        glove_h,
    )

    seam_candidates = _deduplicate_candidates(
        seam_candidates,
        glove_w,
        glove_h,
    )

    # --------------------------------------------------------
    # 4. Add intensity/gradient confirmation
    # --------------------------------------------------------
    ranked = []

    for candidate in seam_candidates:
        item = dict(
            candidate
        )

        gradient_score = _local_seam_strength(
            processed,
            mask,
            item["line"],
            glove_w,
        )

        dark_score = _dark_seam_score(
            processed,
            mask,
            item["line"],
            glove_w,
        )

        item["gradient_score"] = float(
            gradient_score
        )
        item["dark_seam_score"] = float(
            dark_score
        )

        # Geometry dominates. Gradient and darkness only confirm the seam.
        final_candidate_score = float(np.clip(
            0.78 * item["geometry_score"]
            + 0.14 * gradient_score
            + 0.08 * dark_score,
            0.0,
            1.0,
        ))

        # Very strong internal continuity + both-side material is especially
        # characteristic of the overlap shown in the user's example.
        if (
            item["side_support"] >= 0.82
            and item["edge_continuity"] >= 0.66
            and item["length_ratio"] >= 0.16
        ):
            final_candidate_score = min(
                1.0,
                final_candidate_score + 0.08,
            )

        item["score"] = float(
            final_candidate_score
        )

        if (
            item["score"]
            >= MIN_CANDIDATE_SCORE
        ):
            ranked.append(
                item
            )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = (
        ranked[0]
        if ranked
        else None
    )

    # --------------------------------------------------------
    # 5. Final score
    # --------------------------------------------------------
    if best is None:
        final_score = min(
            0.49,
            0.20 * silhouette["silhouette_signal"]
            + 0.10 * merged_peak_signal,
        )
        detected = False

    else:
        seam_signal = float(
            best["score"]
        )

        # Shape information is only supporting evidence because an overlap can
        # still preserve an apparently normal outer silhouette.
        support_signal = float(np.clip(
            0.58 * silhouette["silhouette_signal"]
            + 0.42 * merged_peak_signal,
            0.0,
            1.0,
        ))

        final_score = float(np.clip(
            0.86 * seam_signal
            + 0.14 * support_signal,
            0.0,
            1.0,
        ))

        # Strong overlap geometry should cross the decision boundary even when
        # the external silhouette still looks almost normal.
        if (
            best["side_support"] >= 0.78
            and best["edge_continuity"] >= 0.58
            and best["length_ratio"] >= 0.15
            and best["boundary_score"] >= 0.35
        ):
            final_score = max(
                final_score,
                0.55,
            )

        # Require an actual internal-overlap seam candidate.
        detected = bool(
            final_score >= DETECTION_THRESHOLD
            and best["source"] == "overlap_internal_seam"
            and best["side_support"] >= MIN_SIDE_SUPPORT
        )

    # --------------------------------------------------------
    # 6. Localise the actual overlap seam
    # --------------------------------------------------------
    defect_mask = np.zeros_like(
        mask
    )
    bbox = None
    seam_radius = None

    if detected and best is not None:
        (
            defect_mask,
            bbox,
            seam_radius,
        ) = _line_region(
            mask,
            best["line"],
            glove_w,
        )

    area_pct = (
        100.0
        * np.count_nonzero(
            defect_mask
        )
        / max(
            np.count_nonzero(mask),
            1,
        )
    )

    # --------------------------------------------------------
    # 7. Diagnostics
    # --------------------------------------------------------
    measurements = {
        "area_pct": round(
            float(area_pct),
            3,
        ),

        "fingertip_peaks": int(
            peak_count
        ),

        "fingertip_peak_points": [
            (
                int(x),
                int(y),
            )
            for x, y in fingertip_peaks
        ],

        "merged_peak_signal": round(
            float(merged_peak_signal),
            4,
        ),

        "deep_finger_valleys": int(
            silhouette[
                "deep_valleys"
            ]
        ),

        "moderate_finger_valleys": int(
            silhouette[
                "moderate_valleys"
            ]
        ),

        "upper_hand_solidity": round(
            float(
                silhouette[
                    "solidity"
                ]
            ),
            4,
        ),

        "silhouette_signal": round(
            float(
                silhouette[
                    "silhouette_signal"
                ]
            ),
            4,
        ),

        "raw_seam_candidates": int(
            len(seam_candidates)
        ),

        "accepted_seam_candidates": int(
            len(ranked)
        ),

        "overlap_seam_found": bool(
            best is not None
        ),

        "contact_candidate_source": (
            best["source"]
            if best is not None
            else None
        ),

        "contact_point": (
            best["point"]
            if best is not None
            else None
        ),

        "contact_line": (
            best["line"]
            if best is not None
            else None
        ),

        "contact_radius_px": (
            int(seam_radius)
            if seam_radius is not None
            else None
        ),

        "contact_candidate_score": (
            round(
                float(
                    best["score"]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "seam_geometry_score": (
            round(
                float(
                    best[
                        "geometry_score"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "seam_length_px": (
            round(
                float(
                    best["length"]
                ),
                2,
            )
            if best is not None
            else None
        ),

        "seam_length_ratio": (
            round(
                float(
                    best["length_ratio"]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "seam_verticality": (
            round(
                float(
                    best["verticality"]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "seam_edge_continuity": (
            round(
                float(
                    best[
                        "edge_continuity"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "seam_side_support": (
            round(
                float(
                    best[
                        "side_support"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "seam_boundary_distance_px": (
            round(
                float(
                    best[
                        "boundary_distance"
                    ]
                ),
                2,
            )
            if best is not None
            else None
        ),

        "seam_boundary_score": (
            round(
                float(
                    best[
                        "boundary_score"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "seam_top_position_score": (
            round(
                float(
                    best[
                        "top_position_score"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "typical_finger_width_px": (
            round(
                float(
                    best[
                        "typical_finger_width"
                    ]
                ),
                2,
            )
            if best is not None
            else None
        ),

        "local_merged_width_px": (
            round(
                float(
                    best[
                        "local_merged_width"
                    ]
                ),
                2,
            )
            if best is not None
            else None
        ),

        "merged_width_ratio": (
            round(
                float(
                    best[
                        "merged_width_ratio"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "merged_width_score": (
            round(
                float(
                    best[
                        "merged_width_score"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "contact_gradient_score": (
            round(
                float(
                    best[
                        "gradient_score"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "contact_dark_seam_score": (
            round(
                float(
                    best[
                        "dark_seam_score"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),
    }

    result.update({
        "detected": bool(
            detected
        ),
        "detection_score": round(
            float(final_score),
            6,
        ),
        "bounding_box": bbox,
        "mask": defect_mask,
        "measurements": measurements,
    })

    return result


# ============================================================
# OPTIONAL DEVELOPMENT VISUALISATION
# ============================================================

def debug_touching(
    processed: dict,
    segmentation: dict,
):
    """
    Optional helper for local development.

    Returns a BGR debug image showing:
    - glove mask outline
    - accepted overlap seam
    - final bounding box

    This function is not used by evaluate.py.
    """
    original = processed.get(
        "original"
    )

    if original is None:
        return None

    output = original.copy()

    result = detect_touching(
        processed,
        segmentation,
    )

    mask = segmentation.get(
        "glove_mask"
    )

    if mask is not None:
        contours, _ = cv2.findContours(
            _binary_mask(mask),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            output,
            contours,
            -1,
            (0, 255, 255),
            2,
        )

    line = result.get(
        "measurements",
        {},
    ).get(
        "contact_line"
    )

    if line is not None:
        x1, y1, x2, y2 = [
            int(v)
            for v in line
        ]

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    bbox = result.get(
        "bounding_box"
    )

    if bbox is not None:
        x, y, w, h = [
            int(v)
            for v in bbox
        ]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 255),
            2,
        )

    return output
