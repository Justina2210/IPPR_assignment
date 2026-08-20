"""
discoloration.py
----------------
Revised detector for the "discoloration" defect.

Main idea
---------
The previous detector used full LAB delta-E at two local-reference scales.
That made folds, shadows, finger valleys and cuff edges look like colour
defects because delta-E also includes lightness (L).

This version uses a simpler material-adaptive chromatic strategy:

1. Work only inside the segmented glove mask.
2. Erode only a small boundary margin to avoid glove/background blending.
3. Smooth colour channels so fine wrinkles do not dominate.
4. Automatically choose the colour feature from glove saturation:
   - High-saturation glove (e.g. blue nitrile):
       use HSV hue deviation from the glove's robust median hue.
   - Low-saturation/pale glove (e.g. pale latex):
       use positive LAB b* shift, which captures the yellow discoloration
       present in the current dataset while largely ignoring brightness.
5. Use a robust median + MAD threshold with a minimum floor.
6. Clean candidates using morphology and connected components.
7. Score EACH component using:
       colour strength + affected area + compactness/shape
   instead of selecting the largest surviving component.
8. Return only the best-scoring component as the defect mask/bounding box.

This keeps the detector compatible with evaluate.py:
    detect_discoloration(processed, segmentation) -> dict
"""

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Small edge exclusion only for segmentation anti-aliasing/background bleed.
EDGE_MARGIN_PX = 12

# Smooth broad colour patches while suppressing tiny wrinkle/texture changes.
GAUSSIAN_SIGMA = 10.0

# Decide which colour model to use automatically.
# Blue nitrile samples have very high saturation; pale latex samples do not.
SATURATION_MODE_THRESHOLD = 120.0

# Robust threshold settings.
ROBUST_K = 2.5

# Minimum meaningful chromatic changes.
NITRILE_MIN_HUE_DEVIATION = 2.5   # OpenCV hue units, range 0-179
PALE_MIN_B_SHIFT = 4.0             # LAB b* units

# Connected-component filtering.
MIN_REGION_AREA_PX = 250
MIN_EXTENT = 0.30
MAX_ASPECT_RATIO = 4.0

# Values used only to normalise scores.
NITRILE_REFERENCE_HUE_DEVIATION = 4.5
PALE_REFERENCE_B_SHIFT = 9.0
REFERENCE_AREA_FRACTION = 0.03

# Component ranking:
# shape is deliberately important because the previous false positives
# were elongated crease/finger-valley regions.
COMPONENT_SCORE_WEIGHTS = {
    "strength": 0.35,
    "area": 0.25,
    "shape": 0.40,
}

LOCAL_DETECTION_THRESHOLD = 0.50


# ============================================================
# HELPERS
# ============================================================

def _robust_threshold(values, floor, k=ROBUST_K):
    """
    Robust threshold = max(floor, median + k * robust_std),
    where robust_std is estimated from MAD.
    """
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return float(floor)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_std = 1.4826 * mad

    return float(max(floor, median + k * robust_std))


def _erode_glove_mask(mask_bool, margin_px=EDGE_MARGIN_PX):
    """
    Remove only a shallow glove boundary band.

    This avoids anti-aliased glove/background pixels without removing
    a large portion of the glove surface.
    """
    mask_u8 = mask_bool.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * margin_px + 1, 2 * margin_px + 1),
    )

    eroded = cv2.erode(mask_u8, kernel)
    interior = eroded > 0

    # Fallback for unusual thin/cropped gloves.
    if not np.any(interior):
        return mask_bool.copy()

    return interior


def _circular_hue_distance(hue, reference_hue):
    """
    Circular absolute hue distance for OpenCV hue values [0, 179].
    """
    diff = np.abs(hue - reference_hue)
    return np.minimum(diff, 180.0 - diff)


def _clean_binary_mask(mask_bool):
    """
    Morphological cleanup before connected-component analysis.
    """
    mask_u8 = mask_bool.astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)
    )

    cleaned = cv2.morphologyEx(
        mask_u8, cv2.MORPH_OPEN, open_kernel
    )
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, close_kernel
    )

    return cleaned


def _component_shape_score(extent, aspect):
    """
    Compact blob -> high score.
    Thin/elongated crease -> lower score.
    """
    aspect_score = min(1.0, 1.8 / max(aspect, 1e-6))
    extent_score = min(1.0, extent / 0.65)

    return float(
        0.55 * aspect_score
        + 0.45 * extent_score
    )


def _find_best_component(
    candidate_bool,
    anomaly_map,
    glove_area,
    reference_strength,
):
    """
    Filter and rank connected components.

    Unlike the previous implementation, localisation is NOT based on
    whichever region has the largest area. Each region is independently
    scored using chromatic strength, affected area and compactness.
    """
    cleaned = _clean_binary_mask(candidate_bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, 8
    )

    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < MIN_REGION_AREA_PX:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        bbox_area = max(w * h, 1)
        extent = float(area / bbox_area)

        long_side = max(w, h)
        short_side = max(min(w, h), 1)
        aspect = float(long_side / short_side)

        if extent < MIN_EXTENT:
            continue

        if aspect > MAX_ASPECT_RATIO:
            continue

        component_pixels = labels == label
        mean_anomaly = float(
            anomaly_map[component_pixels].mean()
        )

        strength_score = min(
            1.0,
            mean_anomaly / max(reference_strength, 1e-6),
        )

        area_score = min(
            1.0,
            (area / max(glove_area, 1))
            / REFERENCE_AREA_FRACTION,
        )

        shape_score = _component_shape_score(
            extent, aspect
        )

        component_score = (
            COMPONENT_SCORE_WEIGHTS["strength"] * strength_score
            + COMPONENT_SCORE_WEIGHTS["area"] * area_score
            + COMPONENT_SCORE_WEIGHTS["shape"] * shape_score
        )

        components.append({
            "label": label,
            "area": area,
            "bounding_box": (x, y, w, h),
            "extent": extent,
            "aspect_ratio": aspect,
            "mean_anomaly": mean_anomaly,
            "strength_score": float(strength_score),
            "area_score": float(area_score),
            "shape_score": float(shape_score),
            "component_score": float(component_score),
        })

    if not components:
        return None, None, []

    components.sort(
        key=lambda item: item["component_score"],
        reverse=True,
    )

    best = components[0]

    best_mask = np.zeros_like(cleaned)
    best_mask[labels == best["label"]] = 255

    return best, best_mask, components


# ============================================================
# FEATURE MODES
# ============================================================

def _detect_high_saturation_glove(
    hsv,
    interior_bool,
):
    """
    High-saturation glove mode (e.g. blue nitrile).

    Detect broad HUE changes instead of brightness changes.
    This is much less sensitive to shadows and folds than full LAB
    delta-E.
    """
    hue = hsv[:, :, 0].astype(np.float32)

    # Broad smoothing suppresses narrow wrinkle/crease colour changes.
    hue_smooth = cv2.GaussianBlur(
        hue,
        (0, 0),
        sigmaX=GAUSSIAN_SIGMA,
        sigmaY=GAUSSIAN_SIGMA,
    )

    reference_hue = float(
        np.median(hue_smooth[interior_bool])
    )

    hue_deviation = _circular_hue_distance(
        hue_smooth,
        reference_hue,
    )

    threshold = _robust_threshold(
        hue_deviation[interior_bool],
        NITRILE_MIN_HUE_DEVIATION,
    )

    candidate_bool = (
        (hue_deviation > threshold)
        & interior_bool
    )

    return {
        "mode": "HSV hue deviation",
        "candidate_bool": candidate_bool,
        "anomaly_map": hue_deviation,
        "threshold": threshold,
        "reference_value": reference_hue,
        "reference_strength": NITRILE_REFERENCE_HUE_DEVIATION,
    }


def _detect_pale_glove(
    lab,
    interior_bool,
):
    """
    Pale glove mode (e.g. latex in the current dataset).

    The current pale-glove discoloration samples are yellow patches.
    LAB b* directly measures the blue <-> yellow axis, so a positive
    b* shift isolates the defect much better than full delta-E and
    largely ignores lighting/shadow changes.

    This is intentionally documented as a dataset-specific assumption.
    If future pale-glove discoloration samples use a different colour,
    this feature should be generalised to chroma-direction analysis.
    """
    b_channel = lab[:, :, 2].astype(np.float32)

    b_smooth = cv2.GaussianBlur(
        b_channel,
        (0, 0),
        sigmaX=GAUSSIAN_SIGMA,
        sigmaY=GAUSSIAN_SIGMA,
    )

    reference_b = float(
        np.median(b_smooth[interior_bool])
    )

    # Positive = more yellow than the glove's dominant colour.
    b_shift = b_smooth - reference_b

    threshold = _robust_threshold(
        b_shift[interior_bool],
        PALE_MIN_B_SHIFT,
    )

    candidate_bool = (
        (b_shift > threshold)
        & interior_bool
    )

    # Negative shifts are not candidates in this mode.
    anomaly_map = np.maximum(
        b_shift,
        0.0,
    )

    return {
        "mode": "LAB b* positive colour shift",
        "candidate_bool": candidate_bool,
        "anomaly_map": anomaly_map,
        "threshold": threshold,
        "reference_value": reference_b,
        "reference_strength": PALE_REFERENCE_B_SHIFT,
    }


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_discoloration(processed: dict, segmentation: dict) -> dict:
    """
    Detect a discoloration region on the segmented glove.

    Required evaluate.py contract:
        defect_name
        detected
        detection_score
        algorithm
        bounding_box
        mask
        measurements
    """
    result = {
        "defect_name": "discoloration",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": (
            "Material-adaptive chromatic deviation "
            "+ connected-component scoring"
        ),
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }

    glove_mask = segmentation.get("glove_mask")
    glove_area = int(segmentation.get("glove_area", 0))

    if glove_mask is None or glove_area <= 0:
        return result

    mask_bool = glove_mask > 0

    if not np.any(mask_bool):
        return result

    interior_bool = _erode_glove_mask(
        mask_bool,
        EDGE_MARGIN_PX,
    )

    hsv = processed["hsv"].astype(np.float32)
    lab = processed["lab"].astype(np.float32)

    median_saturation = float(
        np.median(hsv[:, :, 1][interior_bool])
    )

    # Automatically choose the chromatic feature that matches the glove.
    if median_saturation >= SATURATION_MODE_THRESHOLD:
        feature = _detect_high_saturation_glove(
            hsv,
            interior_bool,
        )
    else:
        feature = _detect_pale_glove(
            lab,
            interior_bool,
        )

    best, best_mask, components = _find_best_component(
        feature["candidate_bool"],
        feature["anomaly_map"],
        glove_area,
        feature["reference_strength"],
    )

    if best is None:
        result["measurements"] = {
            "area_pct": 0.0,
            "num_regions": 0,
            "mode": feature["mode"],
            "threshold": round(
                float(feature["threshold"]), 3
            ),
            "median_saturation": round(
                median_saturation, 2
            ),
        }
        return result

    detection_score = float(
        np.clip(
            best["component_score"],
            0.0,
            1.0,
        )
    )

    area_pct = (
        100.0 * best["area"]
        / max(glove_area, 1)
    )

    result["detected"] = (
        detection_score
        >= LOCAL_DETECTION_THRESHOLD
    )
    result["detection_score"] = detection_score
    result["bounding_box"] = best["bounding_box"]
    result["mask"] = best_mask

    result["measurements"] = {
        "area_pct": round(area_pct, 2),
        "num_regions": len(components),
        "mode": feature["mode"],
        "threshold": round(
            float(feature["threshold"]), 3
        ),
        "reference_value": round(
            float(feature["reference_value"]), 3
        ),
        "median_saturation": round(
            median_saturation, 2
        ),
        "mean_anomaly": round(
            best["mean_anomaly"], 3
        ),
        "extent": round(
            best["extent"], 3
        ),
        "aspect_ratio": round(
            best["aspect_ratio"], 3
        ),
        "strength_score": round(
            best["strength_score"], 3
        ),
        "area_score": round(
            best["area_score"], 3
        ),
        "shape_score": round(
            best["shape_score"], 3
        ),
    }

    return result


# ============================================================
# OPTIONAL QUICK TEST
# ============================================================

if __name__ == "__main__":
    import os

    from preprocessing import (
        load_image,
        preprocess_image,
    )
    from segmentation import segment_glove

    candidate_folders = [
        os.path.join(
            "datasets", "nitrile", "discoloration"
        ),
        os.path.join(
            "datasets", "latex", "discoloration"
        ),
        os.path.join(
            "datasets", "cotton", "discoloration"
        ),
        # Compatibility with an older folder spelling:
        os.path.join(
            "dataset", "nitrile", "discoloration"
        ),
        os.path.join(
            "dataset", "latex", "discoloration"
        ),
        os.path.join(
            "dataset", "cotton", "discoloration"
        ),
    ]

    sample = None

    for folder in candidate_folders:
        if not os.path.isdir(folder):
            continue

        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                sample = os.path.join(folder, fname)
                break

        if sample is not None:
            break

    if sample is None:
        print(
            "No discoloration image found under "
            "datasets/ or dataset/."
        )
    else:
        processed = preprocess_image(
            load_image(sample)
        )
        segmentation = segment_glove(
            processed
        )
        output = detect_discoloration(
            processed,
            segmentation,
        )

        print(f"Image: {sample}")
        print(
            "Detected:",
            output["detected"],
        )
        print(
            "Score:",
            round(
                output["detection_score"],
                3,
            ),
        )
        print(
            "Bounding box:",
            output["bounding_box"],
        )
        print(
            "Measurements:",
            output["measurements"],
        )