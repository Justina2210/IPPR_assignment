"""Incomplete-beading / incomplete-cuff-hem detector.

Uses two complementary signals: continuity of the cuff ridge and regularity of
the actual cuff-opening silhouette.  The latter is important for latex and
nitrile samples where the defect is a missing/notched cuff edge rather than a
visible break in an internal horizontal ridge.
"""

import cv2
import numpy as np


def _empty_result():
    return {
        "defect_name": "incomplete_beading",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": "cuff ridge continuity + lower-silhouette notch analysis",
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def _longest_false_run(values):
    longest = current = 0
    start_best = end_best = 0
    start = 0
    for i, v in enumerate(values):
        if not v:
            if current == 0:
                start = i
            current += 1
            if current > longest:
                longest = current
                start_best, end_best = start, i
        else:
            current = 0
    return longest, start_best, end_best


def _smooth_1d(values, width):
    """Smooth a one-dimensional profile without changing its length."""
    width = max(3, int(width) | 1)
    return cv2.GaussianBlur(values.astype(np.float32)[None, :], (width, 1), 0)[0]


def detect_incomplete_beading(processed: dict, segmentation: dict) -> dict:
    result = _empty_result()
    mask = segmentation.get("glove_mask")
    gray = processed.get("gray_enhanced")
    if mask is None or gray is None or np.count_nonzero(mask) < 500:
        return result

    mask = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return result

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    gh = max(1, y1 - y0 + 1)
    gw = max(1, x1 - x0 + 1)

    # Cuff/bead is expected near the lower part of the glove, but not exactly at
    # the silhouette border where the background transition dominates.
    band_top = y0 + int(0.68 * gh)
    band_bottom = min(y1, y0 + int(0.96 * gh))
    if band_bottom <= band_top + 3:
        return result

    # Horizontal ridge -> strong vertical intensity gradient (Sobel dy).
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_y = np.abs(grad_y)

    band_mask = mask[band_top:band_bottom + 1, x0:x1 + 1] > 0
    band_grad = grad_y[band_top:band_bottom + 1, x0:x1 + 1].copy()
    band_grad[~band_mask] = 0

    # Ignore side edges of the cuff to avoid silhouette boundary gradients.
    side_trim = max(2, int(0.06 * gw))
    if band_grad.shape[1] > 2 * side_trim:
        band_grad[:, :side_trim] = 0
        band_grad[:, -side_trim:] = 0

    # Find the row with strongest sustained horizontal ridge evidence.
    row_strength = np.sum(band_grad, axis=1)
    if row_strength.size == 0 or float(row_strength.max()) <= 0:
        return result
    local_row = int(np.argmax(row_strength))
    bead_y = band_top + local_row

    # Aggregate a small vertical window because the bead/hem has thickness.
    radius = max(2, int(0.008 * max(mask.shape)))
    r0 = max(band_top, bead_y - radius)
    r1 = min(band_bottom, bead_y + radius)
    profile = grad_y[r0:r1 + 1, x0:x1 + 1].max(axis=0)
    cuff_present = (mask[r0:r1 + 1, x0:x1 + 1] > 0).any(axis=0)

    valid_idx = np.where(cuff_present)[0]
    if valid_idx.size < max(20, int(0.20 * gw)):
        return result
    left, right = int(valid_idx.min()), int(valid_idx.max())
    profile = profile[left:right + 1]
    cuff_present = cuff_present[left:right + 1]

    valid_values = profile[cuff_present]
    if valid_values.size == 0:
        return result
    edge_threshold = max(14.0, float(np.percentile(valid_values, 58)))
    bead_present = (profile >= edge_threshold) & cuff_present

    # Smooth very tiny gaps; manufacturing defects should span more than a few pixels.
    arr = bead_present.astype(np.uint8)[None, :] * 255
    close_width = max(3, int(0.018 * max(1, len(bead_present))))
    if close_width % 2 == 0:
        close_width += 1
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 1)))
    bead_present = arr[0] > 0

    # Restrict gap analysis to actual cuff span.
    span = len(bead_present)
    coverage = float(np.mean(bead_present)) if span else 0.0
    longest_gap, gap_start, gap_end = _longest_false_run(bead_present)
    gap_ratio = longest_gap / max(span, 1)

    # A true incomplete bead normally has some bead present + a meaningful local gap.
    # If almost no ridge exists at all, confidence is lower because lighting/texture may
    # simply make the bead invisible.
    gap_signal = np.clip((gap_ratio - 0.05) / 0.25, 0.0, 1.0)
    coverage_context = np.clip((coverage - 0.20) / 0.35, 0.0, 1.0)
    brokenness = np.clip((0.90 - coverage) / 0.55, 0.0, 1.0)
    ridge_score = float(np.clip(
        0.62 * gap_signal + 0.20 * coverage_context + 0.18 * brokenness,
        0.0, 1.0,
    ))

    # Analyse the true lower silhouette. A complete cuff opening is approximately
    # straight or gently curved; incomplete beading produces a deep local notch or
    # a substantial run that ends well above the two outer cuff edges.
    lower_y = np.full(gw, np.nan, dtype=np.float32)
    for col in range(gw):
        col_ys = np.flatnonzero(mask[:, x0 + col] > 0)
        if col_ys.size:
            lower_y[col] = float(col_ys[-1])

    trim = max(2, int(0.08 * gw))
    valid_lower = np.isfinite(lower_y)
    core_idx = np.flatnonzero(valid_lower)[trim: max(trim, np.count_nonzero(valid_lower) - trim)]
    notch_depth = notch_ratio = notch_width_ratio = 0.0
    notch_start = notch_end = 0
    silhouette_score = 0.0
    if core_idx.size >= 20:
        lo, hi = int(core_idx[0]), int(core_idx[-1])
        profile = lower_y[lo:hi + 1]
        finite = np.isfinite(profile)
        if np.count_nonzero(finite) >= 20:
            # Interpolate rare missing columns before smoothing.
            xx = np.arange(profile.size)
            profile[~finite] = np.interp(xx[~finite], xx[finite], profile[finite])
            profile = _smooth_1d(profile, max(5, int(0.025 * gw)))
            edge_n = max(4, int(0.16 * profile.size))
            edge_level = float(np.median(np.r_[profile[:edge_n], profile[-edge_n:]]))
            depth = edge_level - profile
            notch_depth = float(max(0.0, depth.max()))
            notch_ratio = notch_depth / gh
            deep = depth >= max(0.025 * gh, 0.35 * notch_depth)
            run, rs, re = _longest_false_run(~deep)
            notch_width_ratio = run / max(profile.size, 1)
            notch_start, notch_end = lo + rs, lo + re
            depth_signal = np.clip((notch_ratio - 0.025) / 0.10, 0.0, 1.0)
            width_signal = np.clip((notch_width_ratio - 0.06) / 0.28, 0.0, 1.0)
            silhouette_score = float(0.68 * depth_signal + 0.32 * width_signal)

    score = float(max(ridge_score, silhouette_score))
    ridge_detected = ridge_score >= 0.50 and gap_ratio >= 0.08 and coverage >= 0.18
    notch_detected = silhouette_score >= 0.50 and notch_ratio >= 0.035 and notch_width_ratio >= 0.07
    detected = ridge_detected or notch_detected

    defect_mask = np.zeros_like(mask)
    bbox = None
    if notch_detected:
        gx0 = x0 + notch_start
        gx1 = x0 + notch_end
        gy0 = max(y0, int(y1 - max(notch_depth, 0.08 * gh)))
        gy1 = y1
        cv2.rectangle(defect_mask, (gx0, gy0), (gx1, gy1), 255, -1)
        bbox = (gx0, gy0, max(1, gx1 - gx0 + 1), max(1, gy1 - gy0 + 1))
    elif ridge_detected and longest_gap > 0:
        gx0 = x0 + left + gap_start
        gx1 = x0 + left + gap_end
        gy0 = max(y0, bead_y - max(6, radius * 2))
        gy1 = min(y1, bead_y + max(6, radius * 2))
        cv2.rectangle(defect_mask, (gx0, gy0), (gx1, gy1), 255, -1)
        bbox = (gx0, gy0, max(1, gx1 - gx0 + 1), max(1, gy1 - gy0 + 1))
    defect_mask = cv2.bitwise_and(defect_mask, mask)

    area_pct = 100.0 * np.count_nonzero(defect_mask) / max(np.count_nonzero(mask), 1)
    result.update({
        "detected": bool(detected),
        "detection_score": score,
        "bounding_box": bbox,
        "mask": defect_mask,
        "measurements": {
            "area_pct": round(float(area_pct), 3),
            "bead_y": int(bead_y),
            "bead_coverage": round(coverage, 4),
            "longest_gap_px": int(longest_gap),
            "longest_gap_ratio": round(float(gap_ratio), 4),
            "edge_threshold": round(float(edge_threshold), 2),
            "ridge_score": round(ridge_score, 4),
            "cuff_notch_depth_px": round(notch_depth, 2),
            "cuff_notch_depth_ratio": round(notch_ratio, 4),
            "cuff_notch_width_ratio": round(notch_width_ratio, 4),
            "silhouette_score": round(silhouette_score, 4),
        },
    })
    return result
