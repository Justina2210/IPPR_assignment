import cv2
import numpy as np


# TUNED-BY-EYE starting values from the five supplied positive samples.
CUFF_SEARCH_START_RATIO = 0.70
MIN_CUFF_WIDTH_RATIO = 0.16
SIDE_TRIM_RATIO = 0.06
PROFILE_SMOOTH_RATIO = 0.018

MIN_NOTCH_DEPTH_RATIO = 0.018
STRONG_NOTCH_DEPTH_RATIO = 0.115
MIN_NOTCH_SPAN_RATIO = 0.025
STRONG_NOTCH_SPAN_RATIO = 0.20
MIN_ROUGHNESS_RATIO = 0.006
STRONG_ROUGHNESS_RATIO = 0.032
MIN_ABNORMAL_AREA_RATIO = 0.0008
STRONG_ABNORMAL_AREA_RATIO = 0.020
SHALLOW_WIDE_DEPTH_RATIO = 0.008
SHALLOW_WIDE_COVERAGE = 0.25
SHALLOW_WIDE_P90_DEPTH_RATIO = 0.012

BOUNDARY_OVERLAY_THICKNESS_RATIO = 0.018
DETECTION_SCORE_THRESHOLD = 0.45

ALGORITHM = (
    "Robust lower glove-mask cuff profile: detect deep/wide missing bead "
    "notches and abnormal edge roughness relative to a fitted cuff baseline"
)


def _empty_result(shape=None):
    return {
        "defect_name": "incomplete_beading",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": ALGORITHM,
        "bounding_box": None,
        "mask": None if shape is None else np.zeros(shape, dtype=np.uint8),
        "measurements": {},
    }


def _smooth_profile(values, window):
    """Median-smooth a 1-D profile without adding external dependencies."""
    values = np.asarray(values, dtype=np.float64)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window >= values.size:
        window = max(3, values.size - 1 if values.size % 2 == 0 else values.size)
    if values.size < 3 or window > values.size:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.array([
        np.median(padded[index:index + window]) for index in range(values.size)
    ], dtype=np.float64)


def _runs(binary):
    """Return inclusive start/end indices of True runs in a 1-D array."""
    padded = np.pad(np.asarray(binary, dtype=np.uint8), (1, 1))
    change = np.diff(padded.astype(np.int16))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _robust_cuff_baseline(xs, ys):
    """Fit a line mainly from the lowest intact portions of the cuff edge."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2:
        return ys.copy(), (0.0, float(ys[0]) if ys.size else 0.0)

    # Missing-bead regions move upward, so initialise from the lower 45% of boundary points and iteratively discard upward outliers.
    threshold = float(np.percentile(ys, 55))
    keep = ys >= threshold
    if np.count_nonzero(keep) < 2:
        keep = np.ones_like(ys, dtype=bool)

    coefficients = np.polyfit(xs[keep], ys[keep], 1)
    for _ in range(3):
        predicted = np.polyval(coefficients, xs)
        residual = predicted - ys
        scale = max(float(np.median(np.abs(residual - np.median(residual)))), 1.0)
        keep = residual <= max(3.0, 2.5 * scale)
        # Retain enough lower-edge anchors for a stable baseline.
        keep |= ys >= np.percentile(ys, 72)
        if np.count_nonzero(keep) < 2:
            break
        coefficients = np.polyfit(xs[keep], ys[keep], 1)

    return np.polyval(coefficients, xs), (float(coefficients[0]), float(coefficients[1]))


def _cuff_profile(mask, bounds):
    """Extract the bottommost glove pixel for columns belonging to the cuff."""
    gx, gy, gw, gh = bounds
    search_top = gy + int(round(CUFF_SEARCH_START_RATIO * gh))
    bottom_by_x = []
    for x in range(gx, gx + gw):
        ys = np.flatnonzero(mask[search_top:gy + gh, x] > 0)
        bottom_by_x.append(np.nan if ys.size == 0 else float(search_top + ys[-1]))
    bottom_by_x = np.asarray(bottom_by_x, dtype=np.float64)

    valid = np.isfinite(bottom_by_x)
    if np.count_nonzero(valid) < max(8, int(round(MIN_CUFF_WIDTH_RATIO * gw))):
        return None

    valid_indices = np.flatnonzero(valid)
    left, right = int(valid_indices[0]), int(valid_indices[-1])
    trim = int(round(SIDE_TRIM_RATIO * (right - left + 1)))
    left, right = left + trim, right - trim
    if right - left + 1 < max(8, int(round(MIN_CUFF_WIDTH_RATIO * gw))):
        return None

    xs = np.arange(gx + left, gx + right + 1, dtype=np.int32)
    profile = bottom_by_x[left:right + 1]
    known = np.flatnonzero(np.isfinite(profile))
    if known.size < 2:
        return None
    missing = np.flatnonzero(~np.isfinite(profile))
    if missing.size:
        profile[missing] = np.interp(missing, known, profile[known])

    smooth_window = max(3, int(round(PROFILE_SMOOTH_RATIO * gw)))
    profile = _smooth_profile(profile, smooth_window)
    return xs, profile, int(search_top)


def detect_incomplete_beading(processed, segmentation):
    """Detect incomplete beading and return the standard result dictionary."""
    del processed  # Only the supplied glove mask is analysed.
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

    extracted = _cuff_profile(mask, (gx, gy, gw, gh))
    if extracted is None:
        return _empty_result(mask.shape)
    xs, observed, search_top = extracted
    expected, (slope, intercept) = _robust_cuff_baseline(xs, observed)

    # Positive residual means the cuff retreats upward from the expected edge; negative values are harmless downward protrusions.
    residual = np.maximum(expected - observed, 0.0)
    minimum_depth = max(2.0, MIN_NOTCH_DEPTH_RATIO * gh)
    abnormal = residual >= minimum_depth
    abnormal_runs = _runs(abnormal)

    candidates = []
    for start, end in abnormal_runs:
        span = end - start + 1
        span_ratio = span / float(gw)
        if span_ratio < MIN_NOTCH_SPAN_RATIO:
            continue
        run_residual = residual[start:end + 1]
        candidates.append({
            "start": int(start),
            "end": int(end),
            "span": int(span),
            "span_ratio": float(span_ratio),
            "max_depth": float(np.max(run_residual)),
            "mean_depth": float(np.mean(run_residual)),
            "missing_area": float(np.sum(run_residual)),
        })

    best = max(
        candidates,
        key=lambda item: item["missing_area"],
        default=None,
    )

    # Roughness uses percentile 80 of the detrended differences so a few extreme points can't dominate.
    detrended = observed - expected
    differences = np.abs(np.diff(detrended))
    roughness = float(np.percentile(differences, 80)) if differences.size else 0.0
    roughness_ratio = roughness / float(gh)

    # Some incomplete beads are shallow but irregular across much of the cuff rather than one deep notch (e.g. nitrile sample 1).
    shallow_abnormal = residual >= SHALLOW_WIDE_DEPTH_RATIO * gh
    shallow_coverage = float(np.count_nonzero(shallow_abnormal) / max(residual.size, 1))
    p90_depth_ratio = float(np.percentile(residual, 90) / float(gh))
    shallow_wide_case = bool(
        shallow_coverage >= SHALLOW_WIDE_COVERAGE
        and p90_depth_ratio >= SHALLOW_WIDE_P90_DEPTH_RATIO)

    if best is None:
        depth_ratio = span_ratio = abnormal_area_ratio = 0.0
    else:
        depth_ratio = best["max_depth"] / float(gh)
        span_ratio = best["span_ratio"]
        abnormal_area_ratio = best["missing_area"] / float(glove_area)

    depth_score = float(np.clip(
        (depth_ratio - MIN_NOTCH_DEPTH_RATIO)
        / (STRONG_NOTCH_DEPTH_RATIO - MIN_NOTCH_DEPTH_RATIO), 0.0, 1.0))
    span_score = float(np.clip(
        (span_ratio - MIN_NOTCH_SPAN_RATIO)
        / (STRONG_NOTCH_SPAN_RATIO - MIN_NOTCH_SPAN_RATIO), 0.0, 1.0))
    roughness_score = float(np.clip(
        (roughness_ratio - MIN_ROUGHNESS_RATIO)
        / (STRONG_ROUGHNESS_RATIO - MIN_ROUGHNESS_RATIO), 0.0, 1.0))
    area_score = float(np.clip(
        (abnormal_area_ratio - MIN_ABNORMAL_AREA_RATIO)
        / (STRONG_ABNORMAL_AREA_RATIO - MIN_ABNORMAL_AREA_RATIO), 0.0, 1.0))

    # A local notch is the primary evidence; roughness only supports shallow jagged cases, never triggering detection alone.
    score = float(np.clip(
        0.38 * depth_score + 0.27 * span_score
        + 0.20 * area_score + 0.15 * roughness_score,
        0.0, 1.0))
    has_notch = best is not None and depth_ratio >= MIN_NOTCH_DEPTH_RATIO
    enough_extent = span_ratio >= MIN_NOTCH_SPAN_RATIO
    deep_split_case = bool(
        best is not None and depth_ratio >= 0.10 and span_ratio >= 0.035)
    detected = bool(
        (has_notch and enough_extent and score >= DETECTION_SCORE_THRESHOLD)
        or shallow_wide_case or deep_split_case)
    if deep_split_case:
        score = max(score, 0.68)
    if shallow_wide_case:
        shallow_strength = float(np.clip(
            0.55 * (shallow_coverage / SHALLOW_WIDE_COVERAGE)
            + 0.45 * (p90_depth_ratio / SHALLOW_WIDE_P90_DEPTH_RATIO),
            0.0, 1.0))
        score = max(score, 0.52 + 0.18 * shallow_strength)

    defect_mask = np.zeros_like(mask)
    box = None
    if detected:
        # Cuff columns are localised as those reaching the last 12% of glove height, since exact missing-pixel estimation is unstable.
        near_bottom = gy + int(round(0.88 * gh))
        reaches_bottom = np.array([
            np.any(mask[near_bottom:gy + gh, x] > 0)
            for x in range(gx, gx + gw)
        ], dtype=bool)
        cuff_columns = np.flatnonzero(reaches_bottom)
        if cuff_columns.size:
            x0 = gx + int(cuff_columns[0])
            x1 = gx + int(cuff_columns[-1]) + 1
        else:
            x0, x1 = int(xs[0]), int(xs[-1]) + 1
        y0 = max(gy, gy + int(round(0.72 * gh)))
        y1 = gy + gh
        boundary_thickness = max(3, int(round(0.010 * gh)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * boundary_thickness + 1, 2 * boundary_thickness + 1),
        )
        inner_boundary = cv2.subtract(mask, cv2.erode(mask, kernel))
        defect_mask[y0:y1, x0:x1] = inner_boundary[y0:y1, x0:x1]
        points = cv2.findNonZero(defect_mask)
        if points is not None:
            bx, by, bw, bh = cv2.boundingRect(points)
            box = (int(bx), int(by), int(bw), int(bh))

    defect_pixels = int(np.count_nonzero(defect_mask))
    return {
        "defect_name": "incomplete_beading",
        "detected": bool(detected),
        "detection_score": float(np.clip(score, 0.0, 1.0)),
        "algorithm": ALGORITHM,
        "bounding_box": box,
        "mask": defect_mask,
        "measurements": {
            "area_pct": round(100.0 * defect_pixels / float(glove_area), 3),
            "notch_depth_ratio": round(float(depth_ratio), 4),
            "notch_span_ratio": round(float(span_ratio), 4),
            "abnormal_area_ratio": round(float(abnormal_area_ratio), 5),
            "cuff_roughness_ratio": round(float(roughness_ratio), 5),
            "shallow_abnormal_coverage": round(float(shallow_coverage), 4),
            "cuff_p90_depth_ratio": round(float(p90_depth_ratio), 4),
            "shallow_wide_case": bool(shallow_wide_case),
            "deep_split_case": bool(deep_split_case),
            "cuff_profile_columns": int(xs.size),
            "candidate_notches": int(len(candidates)),
            "baseline_slope": round(float(slope), 5),
            "baseline_intercept": round(float(intercept), 2),
        },
    }
