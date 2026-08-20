"""
incomplete_beading.py
=====================

Detector for INCOMPLETE BEADING at the glove cuff.

This version is OUTLINE-DRIVEN.

Why:
-----
Incomplete beading belongs to the BOTTOM CUFF RIM. Previous approaches could
incorrectly detect side edges because they searched for generic lines/creases.

Pipeline:
---------
1. Take the segmented glove mask.
2. Extract the largest glove contour.
3. Build a visible glove outline.
4. Locate the actual bottom-cuff arc from the contour.
5. Ignore left/right side-wall contour points.
6. Build a narrow band just INSIDE the bottom cuff outline.
7. Detect the rolled-bead edge/texture within that band.
8. Measure continuity along the cuff.
9. Flag only local missing / weak bead segments.
10. Return a tight box around the missing segment.

No dependency on any other detector.

Public API:
-----------
detect_incomplete_beading(processed: dict, segmentation: dict) -> dict
"""

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DETECTION_THRESHOLD = 0.50

# Portion of glove height where cuff is expected.
CUFF_TOP_FRAC = 0.72

# The bottom cuff contour is obtained from columns whose bottom-most glove
# point belongs to the lower glove region.
MIN_CUFF_COLUMN_DEPTH = 0.74

# Remove sides of detected cuff span.
CUFF_SIDE_TRIM_FRAC = 0.08

# Bottom profile smoothing.
PROFILE_SMOOTH_FRAC = 0.020

# Bead band: inspect pixels just INSIDE the bottom silhouette.
BEAD_BAND_INNER_FRAC = 0.010
BEAD_BAND_OUTER_FRAC = 0.055

# Edge detector.
CANNY_LOW = 22
CANNY_HIGH = 75

# Minimum local span that can be called incomplete.
MIN_MISSING_WIDTH_FRAC = 0.035
MAX_MISSING_WIDTH_FRAC = 0.40

# Continuity classification.
WEAK_SUPPORT_THRESHOLD = 0.30
STRONG_NORMAL_SUPPORT = 0.55

# Candidate acceptance.
MIN_CANDIDATE_SCORE = 0.42


# ============================================================
# BASIC HELPERS
# ============================================================

def _empty_result():
    return {
        "defect_name": "incomplete_beading",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": (
            "glove-outline cuff isolation + inner bead-band continuity analysis"
        ),
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def _binary(mask):
    return (mask > 0).astype(np.uint8) * 255


def _largest_contour(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _bounds(mask):
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


def _smooth_1d(values, window):
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return values.copy()

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1

    if window >= values.size:
        window = values.size - 1 if values.size % 2 == 0 else values.size
        if window < 3:
            return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


# ============================================================
# GLOVE OUTLINE
# ============================================================

def _make_outline(mask, thickness=2):
    """
    Draw the full external glove outline.

    This is useful both conceptually and for optional debug visualization.
    """
    contour = _largest_contour(mask)

    outline = np.zeros_like(mask)

    if contour is not None:
        cv2.drawContours(
            outline,
            [contour],
            -1,
            255,
            thickness,
            cv2.LINE_AA,
        )

    return outline, contour


# ============================================================
# BOTTOM CUFF PROFILE
# ============================================================

def _bottom_profile(mask, x0, x1):
    """
    For each x column, get the lowest glove pixel.

    This directly describes the lower external glove outline.
    """
    xs = np.arange(x0, x1 + 1, dtype=np.int32)
    ys = np.full(len(xs), np.nan, dtype=np.float32)

    for i, x in enumerate(xs):
        foreground = np.flatnonzero(mask[:, x] > 0)

        if foreground.size:
            ys[i] = float(foreground[-1])

    return xs, ys


def _continuous_runs(flags):
    """Return (start,end) runs of True values."""
    flags = np.asarray(flags, dtype=bool)

    if flags.size == 0:
        return []

    padded = np.pad(flags.astype(np.uint8), (1, 1))
    diff = np.diff(padded.astype(np.int16))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    return list(zip(starts, ends))


def _find_cuff_span(mask, x0, x1, y0, y1, glove_h, glove_w):
    """
    Determine which part of the bottom outline is the actual cuff.

    Important:
    The left/right vertical side of the glove is NOT considered cuff.
    We first find a continuous horizontal run of columns whose bottom-most
    point lies sufficiently low in the glove.
    """
    xs, profile = _bottom_profile(mask, x0, x1)

    valid = np.isfinite(profile)

    lower_limit = (
        y0 + MIN_CUFF_COLUMN_DEPTH * glove_h
    )

    low_columns = valid & (profile >= lower_limit)

    runs = _continuous_runs(low_columns)

    if not runs:
        return None

    # Prefer a broad run close to the bottom of the image.
    candidates = []

    for start, end in runs:
        width = end - start + 1

        if width < max(8, int(round(0.10 * glove_w))):
            continue

        segment_y = profile[start:end + 1]
        segment_y = segment_y[np.isfinite(segment_y)]

        if segment_y.size == 0:
            continue

        mean_y = float(np.mean(segment_y))

        width_score = width / max(float(glove_w), 1.0)
        bottom_score = (
            mean_y - y0
        ) / max(float(glove_h), 1.0)

        score = (
            0.55 * width_score
            + 0.45 * bottom_score
        )

        candidates.append(
            (score, start, end)
        )

    if not candidates:
        return None

    _, start, end = max(
        candidates,
        key=lambda item: item[0],
    )

    cuff_left = int(xs[start])
    cuff_right = int(xs[end])

    # Trim endpoints: this is the key side-edge rejection.
    trim = max(
        3,
        int(round(
            CUFF_SIDE_TRIM_FRAC
            * (cuff_right - cuff_left + 1)
        )),
    )

    if cuff_right - cuff_left > 2 * trim + 5:
        cuff_left += trim
        cuff_right -= trim

    return (
        cuff_left,
        cuff_right,
    )


def _cuff_profile(mask, cuff_left, cuff_right):
    """
    Extract bottom cuff outline and smooth it.
    """
    xs, raw = _bottom_profile(
        mask,
        cuff_left,
        cuff_right,
    )

    valid = np.flatnonzero(
        np.isfinite(raw)
    )

    if valid.size < 3:
        return xs, raw, raw

    missing = np.flatnonzero(
        ~np.isfinite(raw)
    )

    if missing.size:
        raw[missing] = np.interp(
            missing,
            valid,
            raw[valid],
        )

    width = cuff_right - cuff_left + 1

    smooth_window = max(
        5,
        int(round(
            PROFILE_SMOOTH_FRAC * width
        )),
    )

    smooth = _smooth_1d(
        raw,
        smooth_window,
    )

    return xs, raw, smooth


# ============================================================
# BEAD BAND CREATION
# ============================================================

def _build_bead_band(
    mask,
    xs,
    profile,
    glove_h,
    glove_w,
):
    """
    Build a curved band INSIDE the bottom outline.

    Instead of searching the entire lower glove, each x-column gets its own
    local cuff y-coordinate. This follows curved / tilted cuffs.

    Example:

        glove material
        █████████████████
        █  BEAD BAND   █
        █████████████████
        ----------------- <- outer cuff profile

    """
    band = np.zeros_like(mask)

    inner_offset = max(
        2,
        int(round(
            BEAD_BAND_INNER_FRAC * glove_h
        )),
    )

    outer_offset = max(
        inner_offset + 3,
        int(round(
            BEAD_BAND_OUTER_FRAC * glove_h
        )),
    )

    h, w = mask.shape[:2]

    for x, bottom_y in zip(xs, profile):
        if not np.isfinite(bottom_y):
            continue

        x = int(x)
        y = int(round(bottom_y))

        # Search upward from bottom contour.
        y_top = max(
            0,
            y - outer_offset,
        )

        y_bottom = max(
            0,
            y - inner_offset,
        )

        if (
            0 <= x < w
            and y_bottom >= y_top
        ):
            band[
                y_top:y_bottom + 1,
                x
            ] = 255

    # Keep only actual glove material.
    band = cv2.bitwise_and(
        band,
        mask,
    )

    return (
        band,
        inner_offset,
        outer_offset,
    )


# ============================================================
# IMAGE EDGE MAP
# ============================================================

def _coarse_edge_map(processed, mask, glove_w):
    gray = None

    if processed is not None:
        gray = processed.get("gray")

    if (
        gray is None
        or gray.shape[:2] != mask.shape[:2]
    ):
        return None

    # Suppress glove wrinkles and cotton texture as much as possible.
    sigma = max(
        1.4,
        0.006 * glove_w,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    edges = cv2.Canny(
        blurred,
        CANNY_LOW,
        CANNY_HIGH,
    )

    return cv2.bitwise_and(
        edges,
        mask,
    )


# ============================================================
# BEAD SUPPORT BY COLUMN
# ============================================================

def _column_bead_support(
    edges,
    band,
    xs,
):
    """
    For each cuff x-column, estimate whether a bead edge is visible inside
    the curved cuff band.

    Output is 0..1.
    """
    support = np.zeros(
        len(xs),
        dtype=np.float32,
    )

    if edges is None:
        return support

    h, w = edges.shape[:2]

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 3),
    )

    # Connect small gaps in the rolled-bead edge while preserving major missing
    # sections.
    bead_edges = cv2.morphologyEx(
        cv2.bitwise_and(edges, band),
        cv2.MORPH_CLOSE,
        horizontal_kernel,
    )

    # Use a horizontal-oriented gradient-like support:
    # count edge pixels inside a small x neighborhood.
    radius = 3

    for i, x in enumerate(xs):
        x = int(x)

        xa = max(
            0,
            x - radius,
        )
        xb = min(
            w,
            x + radius + 1,
        )

        local_band = band[:, xa:xb] > 0

        band_pixels = int(
            np.count_nonzero(local_band)
        )

        if band_pixels == 0:
            support[i] = 0.0
            continue

        local_edges = bead_edges[:, xa:xb] > 0

        edge_pixels = int(
            np.count_nonzero(
                local_edges & local_band
            )
        )

        # Normalize against neighborhood width rather than entire band area.
        # One bead line needs only a few edge pixels per column.
        normalized = (
            edge_pixels
            / max(
                2.0 * (xb - xa),
                1.0,
            )
        )

        support[i] = float(np.clip(
            normalized,
            0.0,
            1.0,
        ))

    # Smooth support along cuff.
    support = _smooth_1d(
        support,
        max(
            5,
            int(round(0.025 * len(xs))),
        ),
    )

    return support


# ============================================================
# NORMALIZE RELATIVE TO THE REST OF THE CUFF
# ============================================================

def _relative_support_score(support):
    """
    A key improvement:

    Do NOT require one universal intensity threshold.

    Instead, determine what "normal bead evidence" looks like in THIS glove
    and compare local segments with the stronger parts of the same cuff.

    This is much more robust across latex / nitrile / cotton.
    """
    if support.size == 0:
        return support, 0.0

    positive = support[support > 0]

    if positive.size == 0:
        return np.zeros_like(support), 0.0

    reference = max(
        float(
            np.percentile(
                support,
                75,
            )
        ),
        0.05,
    )

    relative = np.clip(
        support / reference,
        0.0,
        1.0,
    )

    return relative.astype(np.float32), reference


# ============================================================
# MISSING SEGMENT DETECTION
# ============================================================

def _find_missing_segments(
    xs,
    relative_support,
    cuff_width,
    glove_w,
):
    """
    Find sections where bead evidence becomes much weaker than the rest of
    the cuff.
    """
    if len(xs) == 0:
        return []

    weak = (
        relative_support
        < WEAK_SUPPORT_THRESHOLD
    )

    # Don't use the first/last few columns of the cuff profile.
    edge_guard = max(
        3,
        int(round(0.04 * len(xs))),
    )

    if len(weak) > 2 * edge_guard:
        weak[:edge_guard] = False
        weak[-edge_guard:] = False

    runs = _continuous_runs(
        weak
    )

    min_width = max(
        4,
        int(round(
            MIN_MISSING_WIDTH_FRAC * cuff_width
        )),
    )

    max_width = max(
        min_width,
        int(round(
            MAX_MISSING_WIDTH_FRAC * cuff_width
        )),
    )

    candidates = []

    for start, end in runs:
        width = end - start + 1

        if width < min_width:
            continue

        if width > max_width:
            continue

        local_support = relative_support[
            start:end + 1
        ]

        mean_support = float(
            np.mean(local_support)
        )

        min_support = float(
            np.min(local_support)
        )

        missing_strength = float(np.clip(
            (
                WEAK_SUPPORT_THRESHOLD
                - mean_support
            )
            / max(
                WEAK_SUPPORT_THRESHOLD,
                1e-6,
            ),
            0.0,
            1.0,
        ))

        width_score = float(np.clip(
            width
            / max(
                0.14 * cuff_width,
                1.0,
            ),
            0.0,
            1.0,
        ))

        score = float(
            0.76 * missing_strength
            + 0.24 * width_score
        )

        candidates.append({
            "source": "missing_bead_segment",
            "start_index": int(start),
            "end_index": int(end),
            "x_start": int(xs[start]),
            "x_end": int(xs[end]),
            "width_px": int(width),
            "mean_relative_support": mean_support,
            "minimum_relative_support": min_support,
            "missing_strength": missing_strength,
            "width_score": width_score,
            "geometry_score": score,
        })

    candidates.sort(
        key=lambda item: item["geometry_score"],
        reverse=True,
    )

    return candidates


# ============================================================
# OUTER PROFILE SHAPE SUPPORT
# ============================================================

def _profile_irregularity(
    profile,
    start,
    end,
    cuff_width,
):
    """
    Missing beading can also produce an uneven outside cuff boundary.

    This is secondary evidence only.
    """
    if profile is None or len(profile) < 5:
        return 0.0, 0.0

    baseline_window = max(
        7,
        int(round(
            0.12 * cuff_width
        )),
    )

    baseline = _smooth_1d(
        profile,
        baseline_window,
    )

    residual = np.abs(
        profile - baseline
    )

    local = residual[
        start:end + 1
    ]

    if local.size == 0:
        return 0.0, 0.0

    peak = float(
        np.max(local)
    )

    robust_reference = max(
        float(
            np.percentile(
                residual,
                70,
            )
        ),
        1.0,
    )

    score = float(np.clip(
        (
            float(np.mean(local))
            - robust_reference
        )
        / max(
            2.0 * robust_reference,
            1.0,
        ),
        0.0,
        1.0,
    ))

    return peak, score


# ============================================================
# LOCALIZE DEFECT
# ============================================================

def _defect_region(
    mask,
    band,
    candidate,
    profile,
    xs,
    glove_h,
    glove_w,
):
    """
    Create a tight region around the missing bead span following the cuff.
    """
    start = candidate["start_index"]
    end = candidate["end_index"]

    x_start = int(xs[start])
    x_end = int(xs[end])

    segment_y = profile[
        start:end + 1
    ]

    finite = segment_y[
        np.isfinite(segment_y)
    ]

    if finite.size == 0:
        return np.zeros_like(mask), None

    bottom_y = int(
        round(
            np.max(finite)
        )
    )

    upper_pad = max(
        5,
        int(round(
            0.075 * glove_h
        )),
    )

    lower_pad = max(
        2,
        int(round(
            0.015 * glove_h
        )),
    )

    horizontal_pad = max(
        3,
        int(round(
            0.012 * glove_w
        )),
    )

    xa = max(
        0,
        x_start - horizontal_pad,
    )

    xb = min(
        mask.shape[1] - 1,
        x_end + horizontal_pad,
    )

    ya = max(
        0,
        bottom_y - upper_pad,
    )

    yb = min(
        mask.shape[0] - 1,
        bottom_y + lower_pad,
    )

    region = np.zeros_like(mask)

    # Use cuff/bead band plus a little actual glove area.
    local = np.zeros_like(mask)
    local[
        ya:yb + 1,
        xa:xb + 1,
    ] = 255

    region = cv2.bitwise_and(
        local,
        mask,
    )

    bbox = (
        xa,
        ya,
        xb - xa + 1,
        yb - ya + 1,
    )

    return region, bbox


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_incomplete_beading(
    processed: dict,
    segmentation: dict,
) -> dict:
    """
    Detect incomplete beading using only the bottom cuff outline/band.
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

    mask = _binary(
        mask
    )

    bounds = _bounds(
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
    # 1. Full glove outline
    # --------------------------------------------------------
    outline_mask, contour = _make_outline(
        mask,
        thickness=2,
    )

    if contour is None:
        return result

    # --------------------------------------------------------
    # 2. Actual bottom cuff span
    # --------------------------------------------------------
    cuff_span = _find_cuff_span(
        mask,
        x0,
        x1,
        y0,
        y1,
        glove_h,
        glove_w,
    )

    if cuff_span is None:
        result["measurements"] = {
            "reason": "bottom_cuff_span_not_found",
        }
        return result

    cuff_left, cuff_right = cuff_span

    cuff_width = max(
        1,
        cuff_right - cuff_left + 1,
    )

    # --------------------------------------------------------
    # 3. Curved cuff profile
    # --------------------------------------------------------
    xs, raw_profile, smooth_profile = _cuff_profile(
        mask,
        cuff_left,
        cuff_right,
    )

    if (
        smooth_profile.size < 7
        or np.count_nonzero(
            np.isfinite(smooth_profile)
        ) < 7
    ):
        result["measurements"] = {
            "reason": "cuff_profile_too_small",
        }
        return result

    # --------------------------------------------------------
    # 4. Inner bead band following contour
    # --------------------------------------------------------
    bead_band, inner_offset, outer_offset = _build_bead_band(
        mask,
        xs,
        smooth_profile,
        glove_h,
        glove_w,
    )

    # --------------------------------------------------------
    # 5. Image edges only inside bead band
    # --------------------------------------------------------
    edges = _coarse_edge_map(
        processed,
        mask,
        glove_w,
    )

    support = _column_bead_support(
        edges,
        bead_band,
        xs,
    )

    relative_support, support_reference = _relative_support_score(
        support
    )

    # --------------------------------------------------------
    # 6. Find missing / interrupted bead sections
    # --------------------------------------------------------
    candidates = _find_missing_segments(
        xs,
        relative_support,
        cuff_width,
        glove_w,
    )

    # Add outer-profile irregularity as secondary evidence.
    ranked = []

    for candidate in candidates:
        item = dict(candidate)

        peak_irregularity, irregularity_score = _profile_irregularity(
            smooth_profile,
            item["start_index"],
            item["end_index"],
            cuff_width,
        )

        item["profile_peak_irregularity_px"] = float(
            peak_irregularity
        )

        item["profile_irregularity_score"] = float(
            irregularity_score
        )

        # Missing bead evidence dominates.
        score = float(np.clip(
            0.84 * item["geometry_score"]
            + 0.16 * irregularity_score,
            0.0,
            1.0,
        ))

        item["score"] = score

        if score >= MIN_CANDIDATE_SCORE:
            ranked.append(item)

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
    # 7. Final decision
    # --------------------------------------------------------
    if best is None:
        score = 0.0
        detected = False

    else:
        score = float(
            best["score"]
        )

        # A substantial, very weak section should cross decision threshold.
        strong_missing = (
            best["missing_strength"] >= 0.55
            and best["width_px"] >= max(
                4,
                int(round(
                    0.04 * cuff_width
                )),
            )
        )

        if strong_missing:
            score = max(
                score,
                0.56,
            )

        detected = bool(
            score >= DETECTION_THRESHOLD
        )

    # --------------------------------------------------------
    # 8. Localize only the bottom cuff defect
    # --------------------------------------------------------
    defect_mask = np.zeros_like(
        mask
    )

    bbox = None

    if detected and best is not None:
        defect_mask, bbox = _defect_region(
            mask,
            bead_band,
            best,
            smooth_profile,
            xs,
            glove_h,
            glove_w,
        )

    area_pct = (
        100.0
        * np.count_nonzero(defect_mask)
        / max(
            np.count_nonzero(mask),
            1,
        )
    )

    # --------------------------------------------------------
    # 9. Diagnostics
    # --------------------------------------------------------
    measurements = {
        "area_pct": round(
            float(area_pct),
            3,
        ),

        "cuff_left_x": int(
            cuff_left
        ),

        "cuff_right_x": int(
            cuff_right
        ),

        "cuff_width_px": int(
            cuff_width
        ),

        "bead_band_inner_offset_px": int(
            inner_offset
        ),

        "bead_band_outer_offset_px": int(
            outer_offset
        ),

        "bead_support_reference": round(
            float(support_reference),
            4,
        ),

        "mean_bead_support": round(
            float(
                np.mean(
                    relative_support
                )
            ),
            4,
        ) if relative_support.size else 0.0,

        "minimum_bead_support": round(
            float(
                np.min(
                    relative_support
                )
            ),
            4,
        ) if relative_support.size else 0.0,

        "raw_missing_segments": int(
            len(candidates)
        ),

        "accepted_candidates": int(
            len(ranked)
        ),

        "beading_candidate_found": bool(
            best is not None
        ),

        "candidate_source": (
            best["source"]
            if best is not None
            else None
        ),

        "candidate_x_start": (
            int(best["x_start"])
            if best is not None
            else None
        ),

        "candidate_x_end": (
            int(best["x_end"])
            if best is not None
            else None
        ),

        "candidate_width_px": (
            int(best["width_px"])
            if best is not None
            else None
        ),

        "candidate_mean_support": (
            round(
                float(
                    best[
                        "mean_relative_support"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "candidate_minimum_support": (
            round(
                float(
                    best[
                        "minimum_relative_support"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "missing_strength": (
            round(
                float(
                    best[
                        "missing_strength"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "profile_peak_irregularity_px": (
            round(
                float(
                    best[
                        "profile_peak_irregularity_px"
                    ]
                ),
                2,
            )
            if best is not None
            else None
        ),

        "profile_irregularity_score": (
            round(
                float(
                    best[
                        "profile_irregularity_score"
                    ]
                ),
                4,
            )
            if best is not None
            else None
        ),

        "candidate_score": (
            round(
                float(
                    best["score"]
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
            float(score),
            6,
        ),
        "bounding_box": bbox,
        "mask": defect_mask,
        "measurements": measurements,

        # Extra debug masks are inside measurements only as simple metadata is
        # preferred by evaluate.py. Use debug_incomplete_beading() below when
        # visual debugging is needed.
    })

    return result


# ============================================================
# OPTIONAL DEBUG VISUALIZATION
# ============================================================

def debug_incomplete_beading(
    processed: dict,
    segmentation: dict,
):
    """
    Return a debug image showing:

    YELLOW = complete glove outline
    CYAN   = detected bottom cuff outline
    BLUE   = bead search band
    RED    = detected incomplete bead area

    This helper is NOT required by evaluate.py.
    """
    original = None

    if processed is not None:
        original = processed.get("original")

    if original is None:
        return None

    output = original.copy()

    mask = segmentation.get("glove_mask")
    if mask is None:
        return output

    mask = _binary(mask)

    bounds = _bounds(mask)
    if bounds is None:
        return output

    x0, y0, x1, y1, glove_w, glove_h = bounds

    outline, contour = _make_outline(mask, thickness=2)

    # Full outline: yellow.
    output[outline > 0] = (0, 255, 255)

    cuff_span = _find_cuff_span(
        mask,
        x0,
        x1,
        y0,
        y1,
        glove_h,
        glove_w,
    )

    if cuff_span is not None:
        cuff_left, cuff_right = cuff_span

        xs, _, profile = _cuff_profile(
            mask,
            cuff_left,
            cuff_right,
        )

        if len(xs) and len(profile):
            # Cuff outline: cyan.
            points = []

            for x, y in zip(xs, profile):
                if np.isfinite(y):
                    points.append([
                        int(x),
                        int(round(y)),
                    ])

            if len(points) >= 2:
                cv2.polylines(
                    output,
                    [np.asarray(points, dtype=np.int32)],
                    False,
                    (255, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

            band, _, _ = _build_bead_band(
                mask,
                xs,
                profile,
                glove_h,
                glove_w,
            )

            # Light blue overlay for band.
            overlay = output.copy()
            overlay[band > 0] = (255, 120, 0)

            output = cv2.addWeighted(
                overlay,
                0.25,
                output,
                0.75,
                0,
            )

    result = detect_incomplete_beading(
        processed,
        segmentation,
    )

    defect_mask = result.get("mask")

    if defect_mask is not None and np.any(defect_mask):
        overlay = output.copy()
        overlay[defect_mask > 0] = (0, 0, 255)

        output = cv2.addWeighted(
            overlay,
            0.45,
            output,
            0.55,
            0,
        )

    bbox = result.get("bounding_box")

    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3,
        )

    return output
