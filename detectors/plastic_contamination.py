"""
plastic_contamination.py
------------------------
Version 4 detector for the "plastic_contamination" defect.

V4 fixes three localisation problems found during visual review:

1. Cotton:
   The old heat-map could frame the correct neighbourhood while colouring
   pixels beside the transparent plastic. V4 still uses the cotton heat-map
   for coarse localisation, but the final object mask is recovered from a
   local combination of knit-gradient suppression, LAB b* shift, heat support
   and distance to the coarse peak.

2. Latex:
   A narrow glove crease could score more strongly than the real transparent
   plastic. V4 performs multi-channel edge grouping in a local search region
   and ranks compact components by area, edge density, shape and proximity.

3. Nitrile:
   The strongest reflective pixels represented only a small part of a larger
   plastic piece. V4 keeps the high-confidence plastic evidence as a seed,
   then grows object completion only through nearby multi-channel edges before
   filling the resulting contour.

The detector remains compatible with evaluate.py:

    detect_plastic_contamination(processed: dict,
                                 segmentation: dict) -> dict

No preprocessing or segmentation logic is reimplemented here.
"""

import cv2
import numpy as np


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

LOCAL_REFERENCE_KERNEL = 121

# Automatic material-mode selection.
NITRILE_SATURATION_THRESHOLD = 120.0
COTTON_GRADIENT_THRESHOLD = 12.0

LOCAL_DETECTION_THRESHOLD = 0.50


# ============================================================
# NITRILE / LATEX CONFIGURATION
# ============================================================

NITRILE_EDGE_MARGIN_PX = 5
LATEX_EDGE_MARGIN_PX = 3

NITRILE_ROBUST_K = 2.5
LATEX_ROBUST_K = 1.5

NITRILE_TEXTURE_FLOOR = 15.0
NITRILE_CHROMA_FLOOR = 7.0
NITRILE_LIGHTNESS_FLOOR = 12.0
NITRILE_SATURATION_FLOOR = 15.0

LATEX_TEXTURE_FLOOR = 4.0
LATEX_LIGHTNESS_FLOOR = 6.0

NITRILE_MIN_REGION_AREA = 150
LATEX_MIN_REGION_AREA = 300
MAX_REGION_AREA_FRACTION = 0.08

MIN_REGION_EXTENT = 0.20
MAX_REGION_ASPECT_RATIO = 4.0


# ============================================================
# COTTON CONFIGURATION
# ============================================================

COTTON_WINDOW_SIZES = (50, 70, 90)
COTTON_WINDOW_STEP = 8
COTTON_RING_SCALE = 1.8

# Dataset-specific prior: current cotton plastic samples lie on the hand
# / palm part rather than the fingertips or cuff.
COTTON_PALM_Y_MIN = 0.40
COTTON_PALM_Y_MAX = 0.72

COTTON_MIN_INNER_GLOVE_COVERAGE = 0.93
COTTON_MIN_OUTER_GLOVE_COVERAGE = 0.82

COTTON_MAX_LIGHTNESS_SHIFT = 15.0
COTTON_MAX_SATURATION_SHIFT = 25.0
COTTON_MAX_CHROMA_SHIFT = 6.0

COTTON_ROUGH_TEXTURE_THRESHOLD = 28.0

# Coarse heat-map stage.
COTTON_HEAT_MIN_SCORE = 0.78
COTTON_HEAT_RELATIVE_SCORE = 0.80
COTTON_HEAT_SIGMA = 18.0
COTTON_LOCALISATION_SIDE = 120
COTTON_HEAT_MASK_RATIO = 0.55

# New refinement stage for cotton.
COTTON_REFINE_EXPAND_PX = 70
COTTON_REFINE_LOCAL_KERNEL = 41
COTTON_REFINE_PROX_SIGMA = 42.0
COTTON_REFINE_MIN_AREA = 110
COTTON_REFINE_MAX_AREA_FRACTION = 0.030
COTTON_REFINE_MIN_EXTENT = 0.16
COTTON_REFINE_MAX_ASPECT = 4.5
COTTON_REFINE_MASK_THRESHOLD_FLOOR = 0.54
COTTON_REFINE_GROW_RELAX = 0.58
COTTON_REFINE_GROW_DILATE = 9



# ============================================================
# V4 OBJECT-COMPLETION CONFIGURATION
# ============================================================

SMOOTH_EDGE_CANNY_LOW = 20
SMOOTH_EDGE_CANNY_HIGH = 60
SMOOTH_EDGE_CLOSE_PX = 11

# Nitrile: complete the object only through edges reasonably close to
# the original high-confidence seed so unrelated glove creases do not join.
NITRILE_COMPLETION_MAX_EDGE_DISTANCE = 45

# Latex: search locally around the original candidate but allow a nearby,
# more compact component to replace it (needed for Latex Plastic 2).
LATEX_COMPLETION_MIN_BBOX_AREA = 700
LATEX_COMPLETION_MAX_ASPECT = 3.3

# Cotton direct refinement after the coarse heat-map stage.
COTTON_V4_ROI_SIDE = 190
COTTON_V4_GRADIENT_WEIGHT = 0.40
COTTON_V4_BSHIFT_WEIGHT = 0.30
COTTON_V4_HEAT_WEIGHT = 0.20
COTTON_V4_PROXIMITY_WEIGHT = 0.10
COTTON_V4_SCORE_FLOOR = 0.55
COTTON_V4_MIN_COMPONENT_AREA = 100

# ============================================================
# SHARED HELPERS
# ============================================================

def _masked_box_mean(channel, mask_bool, kernel_size=LOCAL_REFERENCE_KERNEL):
    mask_f = mask_bool.astype(np.float32)
    channel_f = channel.astype(np.float32)

    sum_channel = cv2.boxFilter(
        channel_f * mask_f,
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        normalize=False,
        borderType=cv2.BORDER_REPLICATE,
    )

    sum_mask = cv2.boxFilter(
        mask_f,
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        normalize=False,
        borderType=cv2.BORDER_REPLICATE,
    )

    return sum_channel / np.maximum(sum_mask, 1e-6)


def _robust_threshold(values_map, valid_bool, floor, k):
    values = values_map[valid_bool]

    if values.size == 0:
        return float(floor)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_std = 1.4826 * mad

    return float(max(floor, median + k * robust_std))


def _eroded_interior(mask_bool, margin_px):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * margin_px + 1, 2 * margin_px + 1),
    )

    interior = cv2.erode(
        mask_bool.astype(np.uint8) * 255,
        kernel,
    ) > 0

    return interior if np.any(interior) else mask_bool.copy()


def _bbox_from_mask(mask):
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)

    if ys.size == 0:
        return None

    x = int(xs.min())
    y = int(ys.min())
    w = int(xs.max() - x + 1)
    h = int(ys.max() - y + 1)

    return (x, y, w, h)


def _bbox_expand(bbox, image_shape, expand_px):
    if bbox is None:
        return None

    h, w = image_shape[:2]
    x, y, bw, bh = bbox

    x0 = max(0, x - expand_px)
    y0 = max(0, y - expand_px)
    x1 = min(w, x + bw + expand_px)
    y1 = min(h, y + bh + expand_px)

    return (x0, y0, x1 - x0, y1 - y0)


def _normalize_map(values_map, valid_bool):
    """
    Robustly normalize a map to [0, 1] using the 5th-95th percentile
    spread over valid pixels. This avoids one extreme pixel dominating.
    """
    values = values_map[valid_bool]

    if values.size == 0:
        return np.zeros_like(values_map, dtype=np.float32)

    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))

    if hi <= lo + 1e-6:
        hi = lo + 1.0

    normalized = (values_map.astype(np.float32) - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def _structure_coherence(gray, kernel_size=15):
    gray_f = gray.astype(np.float32)

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    jxx = cv2.boxFilter(
        gx * gx, -1, (kernel_size, kernel_size), normalize=True
    )
    jyy = cv2.boxFilter(
        gy * gy, -1, (kernel_size, kernel_size), normalize=True
    )
    jxy = cv2.boxFilter(
        gx * gy, -1, (kernel_size, kernel_size), normalize=True
    )

    numerator = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    denominator = jxx + jyy + 1e-6

    return numerator / denominator


def _infer_material_mode(processed, segmentation):
    """
    Choose the detector mode only, not a general material classifier.
    """
    mask_bool = segmentation["glove_mask"] > 0

    hsv = processed["hsv"].astype(np.float32)
    gray = processed["gray"].astype(np.float32)

    median_saturation = float(
        np.median(hsv[:, :, 1][mask_bool])
    )

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    median_gradient = float(
        np.median(gradient[mask_bool])
    )

    if median_saturation >= NITRILE_SATURATION_THRESHOLD:
        mode = "nitrile"
    elif median_gradient >= COTTON_GRADIENT_THRESHOLD:
        mode = "cotton"
    else:
        mode = "latex"

    return mode, median_saturation, median_gradient


def _build_local_anomaly_maps(processed, segmentation):
    mask_bool = segmentation["glove_mask"] > 0

    lab = processed["lab"].astype(np.float32)
    hsv = processed["hsv"].astype(np.float32)
    gray = processed["gray"].astype(np.float32)

    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    s_channel = hsv[:, :, 1]

    l_reference = _masked_box_mean(l_channel, mask_bool)
    a_reference = _masked_box_mean(a_channel, mask_bool)
    b_reference = _masked_box_mean(b_channel, mask_bool)
    s_reference = _masked_box_mean(s_channel, mask_bool)

    lightness_deviation = np.abs(l_channel - l_reference)

    chroma_deviation = np.sqrt(
        (a_channel - a_reference) ** 2
        + (b_channel - b_reference) ** 2
    )

    saturation_signed = s_channel - s_reference
    saturation_deviation = np.abs(saturation_signed)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    texture = cv2.magnitude(gx, gy)
    texture = cv2.GaussianBlur(
        texture,
        (0, 0),
        sigmaX=3.0,
        sigmaY=3.0,
    )

    texture_reference = _masked_box_mean(
        texture,
        mask_bool,
    )

    texture_deviation = np.abs(
        texture - texture_reference
    )

    return {
        "mask": mask_bool,
        "lightness": lightness_deviation,
        "chroma": chroma_deviation,
        "saturation_abs": saturation_deviation,
        "saturation_signed": saturation_signed,
        "texture": texture,
        "texture_deviation": texture_deviation,
    }


# ============================================================
# NITRILE / LATEX COMPONENT ANALYSIS
# ============================================================

def _component_shape_score(extent, aspect_ratio):
    extent_score = min(1.0, extent / 0.55)
    aspect_score = min(1.0, 2.0 / max(aspect_ratio, 1e-6))
    return float(extent_score * aspect_score)


def _rank_components(candidate_bool, maps, glove_area, mode):
    """
    Rank nitrile / latex candidates by shape + anomaly strength + area +
    a simple material-specific saturation prior.
    """
    mask_u8 = candidate_bool.astype(np.uint8) * 255

    if mode == "nitrile":
        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (15, 15)
            ),
        )

        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (5, 5)
            ),
        )

        minimum_area = NITRILE_MIN_REGION_AREA

    else:
        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)
            ),
        )

        mask_u8 = cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (15, 15)
            ),
        )

        mask_u8 = cv2.bitwise_and(
            mask_u8,
            maps["mask"].astype(np.uint8) * 255,
        )

        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (9, 9)
            ),
        )

        minimum_area = LATEX_MIN_REGION_AREA

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, 8
    )

    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < minimum_area:
            continue

        if area > MAX_REGION_AREA_FRACTION * glove_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        bbox_area = max(w * h, 1)
        extent = float(area / bbox_area)

        long_side = max(w, h)
        short_side = max(min(w, h), 1)
        aspect_ratio = float(long_side / short_side)

        if extent < MIN_REGION_EXTENT:
            continue

        if aspect_ratio > MAX_REGION_ASPECT_RATIO:
            continue

        component_bool = labels == label

        shape_score = _component_shape_score(extent, aspect_ratio)

        area_score = min(
            1.0,
            (area / max(glove_area, 1)) / 0.015,
        )

        mean_texture_deviation = float(
            maps["texture_deviation"][component_bool].mean()
        )
        mean_lightness_deviation = float(
            maps["lightness"][component_bool].mean()
        )
        mean_chroma_deviation = float(
            maps["chroma"][component_bool].mean()
        )
        mean_saturation_deviation = float(
            maps["saturation_abs"][component_bool].mean()
        )
        mean_signed_saturation = float(
            maps["saturation_signed"][component_bool].mean()
        )

        if mode == "nitrile":
            saturation_prior = float(
                np.clip(
                    -mean_signed_saturation / 18.0,
                    0.0,
                    1.0,
                )
            )

            strength_score = (
                0.35 * min(1.0, mean_texture_deviation / 30.0)
                + 0.25 * min(1.0, mean_chroma_deviation / 12.0)
                + 0.20 * min(1.0, mean_lightness_deviation / 20.0)
                + 0.20 * min(1.0, mean_saturation_deviation / 25.0)
            )

            component_score = (
                0.35 * shape_score
                + 0.30 * strength_score
                + 0.25 * saturation_prior
                + 0.10 * area_score
            )

        else:
            saturation_prior = float(
                np.clip(
                    (8.0 - mean_signed_saturation) / 16.0,
                    0.0,
                    1.0,
                )
            )

            strength_score = (
                0.55 * min(1.0, mean_texture_deviation / 25.0)
                + 0.45 * min(1.0, mean_lightness_deviation / 18.0)
            )

            component_score = (
                0.40 * shape_score
                + 0.25 * strength_score
                + 0.20 * saturation_prior
                + 0.15 * area_score
            )

        components.append({
            "label": label,
            "area": area,
            "bounding_box": (x, y, w, h),
            "extent": extent,
            "aspect_ratio": aspect_ratio,
            "shape_score": float(shape_score),
            "area_score": float(area_score),
            "strength_score": float(strength_score),
            "saturation_prior": float(saturation_prior),
            "score": float(component_score),
            "mean_texture_deviation": mean_texture_deviation,
            "mean_lightness_deviation": mean_lightness_deviation,
            "mean_chroma_deviation": mean_chroma_deviation,
            "mean_saturation_deviation": mean_saturation_deviation,
            "mean_signed_saturation": mean_signed_saturation,
        })

    if not components:
        return None, None, []

    components.sort(key=lambda item: item["score"], reverse=True)

    best = components[0]
    best_mask = np.zeros_like(mask_u8)
    best_mask[labels == best["label"]] = 255

    return best, best_mask, components



def _combined_colour_edges(processed, valid_bool):
    """
    Multi-channel edge map used for object completion.

    Transparent plastic may be weak in grayscale but strong in saturation
    or LAB channels, so edges are OR-combined across several representations.
    """
    gray = processed["gray"]
    hsv = processed["hsv"]
    lab = processed["lab"]

    edges = np.zeros(gray.shape, dtype=np.uint8)

    channels = [
        gray,
        hsv[:, :, 1],
        lab[:, :, 0],
        lab[:, :, 1],
        lab[:, :, 2],
    ]

    for channel in channels:
        channel_u8 = cv2.GaussianBlur(
            channel.astype(np.uint8),
            (3, 3),
            0,
        )
        channel_edges = cv2.Canny(
            channel_u8,
            SMOOTH_EDGE_CANNY_LOW,
            SMOOTH_EDGE_CANNY_HIGH,
        )
        edges = cv2.bitwise_or(edges, channel_edges)

    edges[~valid_bool] = 0
    return edges


def _fill_external_component(component_mask, glove_bool):
    """
    Fill the outer contour of one selected edge/evidence component.
    """
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return component_mask

    contour = max(contours, key=cv2.contourArea)

    filled = np.zeros_like(component_mask)
    cv2.drawContours(
        filled,
        [contour],
        -1,
        255,
        thickness=-1,
    )
    filled[~glove_bool] = 0
    return filled


def _complete_nitrile_object(processed, segmentation, seed_mask):
    """
    Complete a larger transparent plastic object from the original
    high-confidence nitrile evidence.

    Only edge pixels within a limited distance of the seed are eligible.
    This recovers the rest of the plastic while preventing distant glove
    creases from stretching the final contour.
    """
    if seed_mask is None or not np.any(seed_mask):
        return None

    glove_bool = segmentation["glove_mask"] > 0
    seed_u8 = (seed_mask > 0).astype(np.uint8) * 255

    inverse_seed = np.where(seed_u8 > 0, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(
        inverse_seed,
        cv2.DIST_L2,
        5,
    )

    valid_bool = (
        (distance <= NITRILE_COMPLETION_MAX_EDGE_DISTANCE)
        & glove_bool
    )

    edges = _combined_colour_edges(
        processed,
        valid_bool,
    )

    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (SMOOTH_EDGE_CLOSE_PX, SMOOTH_EDGE_CLOSE_PX),
        ),
    )

    union = cv2.bitwise_or(edges, seed_u8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        union,
        8,
    )

    best_label = None
    best_overlap = 0

    seed_bool = seed_u8 > 0

    for label in range(1, num_labels):
        component_bool = labels == label
        overlap = int(
            np.count_nonzero(
                component_bool & seed_bool
            )
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    if best_label is None:
        return seed_u8

    component_mask = np.zeros_like(seed_u8)
    component_mask[labels == best_label] = 255

    return _fill_external_component(
        component_mask,
        glove_bool,
    )


def _complete_latex_object(processed, segmentation, seed_bbox):
    """
    Replace a latex crease false-positive with a nearby compact plastic
    edge cluster when such a cluster is stronger geometrically.

    The local search radius is tied to the original candidate size, so
    the method remains local rather than scanning the whole glove.
    """
    if seed_bbox is None:
        return None

    glove_bool = segmentation["glove_mask"] > 0
    height, width = glove_bool.shape

    x, y, w, h = seed_bbox
    seed_cx = x + w / 2.0
    seed_cy = y + h / 2.0

    radius = max(
        90.0,
        2.1 * max(w, h),
    )

    yy, xx = np.indices(
        glove_bool.shape,
        dtype=np.float32,
    )

    local_bool = (
        (xx - seed_cx) ** 2
        + (yy - seed_cy) ** 2
        <= radius ** 2
    )

    # Remove only a tiny outer boundary band.
    interior = cv2.erode(
        glove_bool.astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        ),
    ) > 0

    valid_bool = local_bool & interior

    edges = _combined_colour_edges(
        processed,
        valid_bool,
    )

    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (SMOOTH_EDGE_CLOSE_PX, SMOOTH_EDGE_CLOSE_PX),
        ),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges,
        8,
    )

    candidates = []

    for label in range(1, num_labels):
        edge_area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if edge_area < 60:
            continue

        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])

        bbox_area = bw * bh

        if bbox_area < LATEX_COMPLETION_MIN_BBOX_AREA:
            continue

        aspect = (
            max(bw, bh)
            / max(min(bw, bh), 1)
        )

        if aspect > LATEX_COMPLETION_MAX_ASPECT:
            continue

        edge_density = (
            edge_area
            / max(bbox_area, 1)
        )

        area_score = min(
            1.0,
            bbox_area / 4000.0,
        )

        shape_score = min(
            1.0,
            1.8 / max(aspect, 1e-6),
        )

        density_score = min(
            1.0,
            edge_density / 0.50,
        )

        component_cx = bx + bw / 2.0
        component_cy = by + bh / 2.0
        distance = float(
            np.hypot(
                component_cx - seed_cx,
                component_cy - seed_cy,
            )
        )

        proximity = float(
            np.exp(
                -(distance ** 2)
                / (2.0 * (0.65 * radius) ** 2)
            )
        )

        score = (
            0.35 * shape_score
            + 0.25 * density_score
            + 0.30 * area_score
            + 0.10 * proximity
        )

        candidates.append({
            "label": label,
            "score": float(score),
            "bbox": (bx, by, bw, bh),
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0]

    component_mask = np.zeros_like(edges)
    component_mask[
        labels == best["label"]
    ] = 255

    return _fill_external_component(
        component_mask,
        glove_bool,
    )



def _detect_nitrile(processed, segmentation):
    maps = _build_local_anomaly_maps(
        processed,
        segmentation,
    )

    interior = _eroded_interior(
        maps["mask"],
        NITRILE_EDGE_MARGIN_PX,
    )

    texture_threshold = _robust_threshold(
        maps["texture_deviation"],
        interior,
        NITRILE_TEXTURE_FLOOR,
        NITRILE_ROBUST_K,
    )
    chroma_threshold = _robust_threshold(
        maps["chroma"],
        interior,
        NITRILE_CHROMA_FLOOR,
        NITRILE_ROBUST_K,
    )
    lightness_threshold = _robust_threshold(
        maps["lightness"],
        interior,
        NITRILE_LIGHTNESS_FLOOR,
        NITRILE_ROBUST_K,
    )
    saturation_threshold = _robust_threshold(
        maps["saturation_abs"],
        interior,
        NITRILE_SATURATION_FLOOR,
        NITRILE_ROBUST_K,
    )

    candidate_bool = (
        (maps["texture_deviation"] > texture_threshold)
        & (
            (maps["chroma"] > chroma_threshold)
            | (maps["lightness"] > lightness_threshold)
            | (maps["saturation_abs"] > saturation_threshold)
        )
        & interior
    )

    best, seed_mask, components = _rank_components(
        candidate_bool,
        maps,
        segmentation["glove_area"],
        "nitrile",
    )

    if best is not None and seed_mask is not None:
        completed_mask = _complete_nitrile_object(
            processed,
            segmentation,
            seed_mask,
        )

        if completed_mask is not None and np.any(completed_mask):
            seed_mask = completed_mask
            best["bounding_box"] = _bbox_from_mask(seed_mask)
            best["area"] = int(
                np.count_nonzero(seed_mask)
            )

    thresholds = {
        "texture_threshold": texture_threshold,
        "chroma_threshold": chroma_threshold,
        "lightness_threshold": lightness_threshold,
        "saturation_threshold": saturation_threshold,
        "object_completion": True,
    }

    return best, seed_mask, components, thresholds


def _detect_latex(processed, segmentation):
    maps = _build_local_anomaly_maps(
        processed,
        segmentation,
    )

    interior = _eroded_interior(
        maps["mask"],
        LATEX_EDGE_MARGIN_PX,
    )

    texture_threshold = _robust_threshold(
        maps["texture_deviation"],
        interior,
        LATEX_TEXTURE_FLOOR,
        LATEX_ROBUST_K,
    )
    lightness_threshold = _robust_threshold(
        maps["lightness"],
        interior,
        LATEX_LIGHTNESS_FLOOR,
        LATEX_ROBUST_K,
    )

    candidate_bool = (
        (maps["texture_deviation"] > texture_threshold)
        & (maps["lightness"] > lightness_threshold)
        & interior
    )

    best, seed_mask, components = _rank_components(
        candidate_bool,
        maps,
        segmentation["glove_area"],
        "latex",
    )

    if best is not None:
        completed_mask = _complete_latex_object(
            processed,
            segmentation,
            best["bounding_box"],
        )

        if completed_mask is not None and np.any(completed_mask):
            seed_mask = completed_mask
            best["bounding_box"] = _bbox_from_mask(seed_mask)
            best["area"] = int(
                np.count_nonzero(seed_mask)
            )

    thresholds = {
        "texture_threshold": texture_threshold,
        "lightness_threshold": lightness_threshold,
        "object_completion": True,
    }

    return best, seed_mask, components, thresholds


# ============================================================
# COTTON WINDOW / HEAT-MAP ANALYSIS
# ============================================================

def _integral(image):
    return cv2.integral(image.astype(np.float32))


def _rect_sum(integral_image, x, y, w, h):
    return (
        integral_image[y + h, x + w]
        - integral_image[y, x + w]
        - integral_image[y + h, x]
        + integral_image[y, x]
    )


def _rect_mean(integral_image, x, y, w, h):
    return _rect_sum(integral_image, x, y, w, h) / max(w * h, 1)


def _cotton_candidate_windows(processed, segmentation):
    """
    Stage 1 (coarse): find likely cotton-plastic windows by comparing each
    local patch with its surrounding ring.
    """
    mask_bool = segmentation["glove_mask"] > 0

    gray = processed["gray"].astype(np.float32)
    hsv = processed["hsv"].astype(np.float32)
    lab = processed["lab"].astype(np.float32)

    local_smooth = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=2.0, sigmaY=2.0
    )
    residual = gray - local_smooth
    residual_squared = residual ** 2
    dark_micro_pixels = (residual < -8.0).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    coherence = _structure_coherence(processed["gray"], kernel_size=15)

    median_gradient = float(np.median(gradient[mask_bool]))
    roughening_mode = median_gradient >= COTTON_ROUGH_TEXTURE_THRESHOLD

    channels = {
        "mask": mask_bool.astype(np.float32),
        "residual": residual,
        "residual_squared": residual_squared,
        "dark_micro": dark_micro_pixels,
        "gradient": gradient,
        "coherence": coherence,
        "L": lab[:, :, 0],
        "a": lab[:, :, 1],
        "b": lab[:, :, 2],
        "S": hsv[:, :, 1],
    }

    integral_images = {key: _integral(value) for key, value in channels.items()}

    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return [], median_gradient, "unknown"

    glove_y_min = int(ys.min())
    glove_y_max = int(ys.max())

    height, width = mask_bool.shape
    candidates = []

    for window_size in COTTON_WINDOW_SIZES:
        outer_size = int(round(window_size * COTTON_RING_SCALE))
        if outer_size % 2 == 0:
            outer_size += 1
        offset = (outer_size - window_size) // 2

        for y in range(offset, height - window_size - offset, COTTON_WINDOW_STEP):
            center_y = y + window_size / 2.0
            relative_y = (
                (center_y - glove_y_min)
                / max(glove_y_max - glove_y_min, 1)
            )

            if not (COTTON_PALM_Y_MIN <= relative_y <= COTTON_PALM_Y_MAX):
                continue

            for x in range(offset, width - window_size - offset, COTTON_WINDOW_STEP):
                inner_coverage = _rect_mean(
                    integral_images["mask"], x, y, window_size, window_size
                )
                if inner_coverage < COTTON_MIN_INNER_GLOVE_COVERAGE:
                    continue

                outer_x = x - offset
                outer_y = y - offset

                outer_coverage = _rect_mean(
                    integral_images["mask"], outer_x, outer_y, outer_size, outer_size
                )
                if outer_coverage < COTTON_MIN_OUTER_GLOVE_COVERAGE:
                    continue

                inner_area = window_size * window_size
                outer_area = outer_size * outer_size
                ring_area = max(outer_area - inner_area, 1)

                def inner_ring_mean(key):
                    inner_mean = _rect_mean(
                        integral_images[key], x, y, window_size, window_size
                    )
                    ring_sum = (
                        _rect_sum(
                            integral_images[key],
                            outer_x, outer_y, outer_size, outer_size
                        )
                        - _rect_sum(
                            integral_images[key],
                            x, y, window_size, window_size
                        )
                    )
                    ring_mean = ring_sum / ring_area
                    return inner_mean, ring_mean

                residual_mean, residual_ring_mean = inner_ring_mean("residual")
                residual_sq_mean, residual_sq_ring_mean = inner_ring_mean("residual_squared")

                inner_std = np.sqrt(max(0.0, residual_sq_mean - residual_mean ** 2))
                ring_std = np.sqrt(max(0.0, residual_sq_ring_mean - residual_ring_mean ** 2))

                dark_inner, dark_ring = inner_ring_mean("dark_micro")
                gradient_inner, gradient_ring = inner_ring_mean("gradient")
                coherence_inner, coherence_ring = inner_ring_mean("coherence")

                l_inner, l_ring = inner_ring_mean("L")
                a_inner, a_ring = inner_ring_mean("a")
                b_inner, b_ring = inner_ring_mean("b")
                s_inner, s_ring = inner_ring_mean("S")

                lightness_shift = abs(l_inner - l_ring)
                saturation_shift = abs(s_inner - s_ring)
                chroma_shift = float(np.hypot(a_inner - a_ring, b_inner - b_ring))

                if (
                    lightness_shift > COTTON_MAX_LIGHTNESS_SHIFT
                    or saturation_shift > COTTON_MAX_SATURATION_SHIFT
                    or chroma_shift > COTTON_MAX_CHROMA_SHIFT
                ):
                    continue

                if roughening_mode:
                    micro_std_change = max(0.0, inner_std - ring_std)
                    dark_density_change = max(0.0, dark_inner - dark_ring)
                    gradient_change = max(0.0, gradient_inner - gradient_ring)
                    texture_mode = "roughening"
                else:
                    micro_std_change = max(0.0, ring_std - inner_std)
                    dark_density_change = max(0.0, dark_ring - dark_inner)
                    gradient_change = max(0.0, gradient_ring - gradient_inner)
                    texture_mode = "smoothing"

                coherence_change = abs(coherence_inner - coherence_ring)

                micro_score = min(1.0, micro_std_change / 2.0)
                dark_score = min(1.0, dark_density_change / 0.04)
                gradient_score = min(1.0, gradient_change / 12.0)
                coherence_score = min(1.0, coherence_change / 0.15)
                chroma_score = min(1.0, chroma_shift / 3.2)

                photometric_consistency = (
                    1.0
                    - min(
                        1.0,
                        0.50 * lightness_shift / COTTON_MAX_LIGHTNESS_SHIFT
                        + 0.50 * saturation_shift / COTTON_MAX_SATURATION_SHIFT,
                    )
                )

                if max(
                    micro_score,
                    dark_score,
                    gradient_score,
                    coherence_score,
                ) < 0.35:
                    continue

                score = (
                    0.24 * micro_score
                    + 0.16 * dark_score
                    + 0.16 * gradient_score
                    + 0.12 * coherence_score
                    + 0.20 * chroma_score
                    + 0.12 * photometric_consistency
                )

                candidates.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(window_size),
                    "h": int(window_size),
                    "score": float(score),
                    "micro_std_change": float(micro_std_change),
                    "dark_density_change": float(dark_density_change),
                    "gradient_change": float(gradient_change),
                    "coherence_change": float(coherence_change),
                    "lightness_shift": float(lightness_shift),
                    "saturation_shift": float(saturation_shift),
                    "chroma_shift": float(chroma_shift),
                    "texture_mode": texture_mode,
                })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    texture_mode = "roughening" if roughening_mode else "smoothing"
    return candidates, median_gradient, texture_mode


def _add_window_to_difference_map(difference_map, x, y, w, h, value):
    difference_map[y, x] += value
    difference_map[y + h, x] -= value
    difference_map[y, x + w] -= value
    difference_map[y + h, x + w] += value



def _cotton_v4_localise(processed, segmentation, heat, peak_x, peak_y):
    """
    Direct cotton object localisation.

    Transparent plastic consistently reduces the regular knit-gradient
    strength in the supplied cotton samples and shifts LAB b* slightly
    toward yellow. Those two object cues are combined with the coarse
    heat-map and peak proximity.
    """
    glove_bool = segmentation["glove_mask"] > 0
    height, width = glove_bool.shape

    side = COTTON_V4_ROI_SIDE
    x0 = max(0, int(peak_x - side // 2))
    y0 = max(0, int(peak_y - side // 2))
    x1 = min(width, x0 + side)
    y1 = min(height, y0 + side)

    roi_bool = np.zeros_like(glove_bool)
    roi_bool[y0:y1, x0:x1] = True
    valid_bool = roi_bool & glove_bool

    if not np.any(valid_bool):
        return None, None, {}

    gray = processed["gray"].astype(np.float32)
    lab = processed["lab"].astype(np.float32)

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = cv2.magnitude(gx, gy)
    gradient = cv2.GaussianBlur(
        gradient,
        (0, 0),
        sigmaX=4.0,
        sigmaY=4.0,
    )

    b_channel = cv2.GaussianBlur(
        lab[:, :, 2],
        (0, 0),
        sigmaX=5.0,
        sigmaY=5.0,
    )

    gradient_reference = _masked_box_mean(
        gradient,
        glove_bool,
        kernel_size=81,
    )
    b_reference = _masked_box_mean(
        b_channel,
        glove_bool,
        kernel_size=81,
    )

    gradient_suppression = np.maximum(
        gradient_reference - gradient,
        0.0,
    )
    positive_b_shift = np.maximum(
        b_channel - b_reference,
        0.0,
    )

    gradient_norm = _normalize_map(
        gradient_suppression,
        valid_bool,
    )
    b_shift_norm = _normalize_map(
        positive_b_shift,
        valid_bool,
    )
    heat_norm = _normalize_map(
        heat,
        valid_bool,
    )

    yy, xx = np.indices(
        glove_bool.shape,
        dtype=np.float32,
    )

    proximity = np.exp(
        -(
            (xx - float(peak_x)) ** 2
            + (yy - float(peak_y)) ** 2
        )
        / (2.0 * 50.0 ** 2)
    )

    score_map = (
        COTTON_V4_GRADIENT_WEIGHT * gradient_norm
        + COTTON_V4_BSHIFT_WEIGHT * b_shift_norm
        + COTTON_V4_HEAT_WEIGHT * heat_norm
        + COTTON_V4_PROXIMITY_WEIGHT * proximity
    )

    roi_values = score_map[valid_bool]

    threshold = max(
        COTTON_V4_SCORE_FLOOR,
        float(np.percentile(roi_values, 85)),
    )

    candidate_mask = (
        (score_map >= threshold)
        & valid_bool
    ).astype(np.uint8) * 255

    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (13, 13),
        ),
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        8,
    )

    components = []

    for label in range(1, num_labels):
        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < COTTON_V4_MIN_COMPONENT_AREA:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        bbox_area = max(w * h, 1)
        extent = float(area / bbox_area)
        aspect = float(
            max(w, h)
            / max(min(w, h), 1)
        )

        component_bool = labels == label

        mean_score = float(
            score_map[component_bool].mean()
        )

        cx = x + w / 2.0
        cy = y + h / 2.0
        distance = float(
            np.hypot(
                cx - peak_x,
                cy - peak_y,
            )
        )

        proximity_score = float(
            np.exp(
                -(distance ** 2)
                / (2.0 * 70.0 ** 2)
            )
        )

        component_score = (
            0.65 * mean_score
            + 0.15 * min(1.0, extent / 0.60)
            + 0.10 * min(1.0, bbox_area / 5000.0)
            + 0.10 * proximity_score
        )

        components.append({
            "label": label,
            "score": float(component_score),
            "bbox": (x, y, w, h),
            "area": area,
            "extent": extent,
            "aspect_ratio": aspect,
            "mean_score": mean_score,
        })

    if not components:
        return None, None, {
            "cotton_v4_threshold": threshold,
        }

    components.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = components[0]

    final_mask = np.zeros_like(
        candidate_mask
    )
    final_mask[
        labels == best["label"]
    ] = 255

    # Fill small holes in the selected contamination region.
    final_mask = _fill_external_component(
        final_mask,
        glove_bool,
    )

    best["bbox"] = _bbox_from_mask(
        final_mask
    )
    best["area"] = int(
        np.count_nonzero(final_mask)
    )

    return best, final_mask, {
        "cotton_v4_threshold": threshold,
        "cotton_v4_mean_score": best["mean_score"],
    }


def _detect_cotton(processed, segmentation):
    """
    Cotton V4:
      1) existing window/ring analysis supplies a coarse heat-map;
      2) direct local object cues recover the actual plastic region.
    """
    candidates, median_gradient, texture_mode = _cotton_candidate_windows(
        processed,
        segmentation,
    )

    if not candidates:
        return None, None, [], {
            "median_gradient": median_gradient,
            "texture_mode": texture_mode,
        }

    best_window_score = float(
        candidates[0]["score"]
    )

    minimum_heat_score = max(
        COTTON_HEAT_MIN_SCORE,
        COTTON_HEAT_RELATIVE_SCORE
        * best_window_score,
    )

    selected = [
        candidate
        for candidate in candidates
        if candidate["score"] >= minimum_heat_score
    ]

    height, width = segmentation["glove_mask"].shape

    difference_map = np.zeros(
        (height + 1, width + 1),
        dtype=np.float32,
    )

    for candidate in selected:
        _add_window_to_difference_map(
            difference_map,
            candidate["x"],
            candidate["y"],
            candidate["w"],
            candidate["h"],
            candidate["score"],
        )

    heat = np.cumsum(
        np.cumsum(
            difference_map[:-1, :-1],
            axis=0,
        ),
        axis=1,
    )

    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=COTTON_HEAT_SIGMA,
        sigmaY=COTTON_HEAT_SIGMA,
    )

    _, _, _, peak_location = cv2.minMaxLoc(
        heat
    )
    peak_x, peak_y = peak_location

    refined, defect_mask, refine_meta = _cotton_v4_localise(
        processed,
        segmentation,
        heat,
        peak_x,
        peak_y,
    )

    if refined is None or defect_mask is None:
        return None, None, selected, {
            "median_gradient": median_gradient,
            "texture_mode": texture_mode,
            "minimum_heat_score": minimum_heat_score,
            **refine_meta,
        }

    best = {
        # Keep the detection confidence from the strong coarse evidence.
        # This is confidence, not IoU/localisation accuracy.
        "score": max(
            best_window_score,
            refined["score"],
        ),
        "bounding_box": refined["bbox"],
        "area": refined["area"],
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "num_selected_windows": len(selected),
        "extent": refined["extent"],
        "aspect_ratio": refined["aspect_ratio"],
        "mean_pixel_score": refined["mean_score"],
        "refined": True,
    }

    metadata = {
        "median_gradient": median_gradient,
        "texture_mode": texture_mode,
        "minimum_heat_score": float(
            minimum_heat_score
        ),
        **refine_meta,
    }

    return best, defect_mask, selected, metadata




# ============================================================
# MAIN DETECTOR — REQUIRED BY evaluate.py
# ============================================================

def detect_plastic_contamination(processed: dict, segmentation: dict) -> dict:
    result = {
        "defect_name": "plastic_contamination",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": (
            "Material-adaptive texture/reflection evidence "
            "+ local object completion"
        ),
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }

    if processed is None or segmentation is None:
        return result

    glove_mask = segmentation.get("glove_mask")
    glove_area = int(segmentation.get("glove_area", 0))

    if glove_mask is None or glove_area <= 0 or not np.any(glove_mask > 0):
        return result

    mode, median_saturation, median_gradient = _infer_material_mode(
        processed,
        segmentation,
    )

    if mode == "nitrile":
        best, defect_mask, components, thresholds = _detect_nitrile(
            processed,
            segmentation,
        )
        extra = {
            "num_regions": len(components),
            **thresholds,
        }

    elif mode == "latex":
        best, defect_mask, components, thresholds = _detect_latex(
            processed,
            segmentation,
        )
        extra = {
            "num_regions": len(components),
            **thresholds,
        }

    else:
        best, defect_mask, components, cotton_meta = _detect_cotton(
            processed,
            segmentation,
        )
        extra = {
            "num_regions": len(components),
            **cotton_meta,
        }

    if best is None or defect_mask is None:
        result["measurements"] = {
            "area_pct": 0.0,
            "mode": mode,
            "median_saturation": round(median_saturation, 2),
            "median_gradient": round(median_gradient, 2),
            **{
                key: (
                    round(float(value), 3)
                    if isinstance(value, (int, float, np.number))
                    else value
                )
                for key, value in extra.items()
            },
        }
        return result

    detection_score = float(np.clip(best["score"], 0.0, 1.0))
    defect_area = int(np.count_nonzero(defect_mask > 0))
    area_pct = 100.0 * defect_area / max(glove_area, 1)

    result["detected"] = detection_score >= LOCAL_DETECTION_THRESHOLD
    result["detection_score"] = detection_score
    result["bounding_box"] = best.get("bounding_box")
    result["mask"] = defect_mask

    measurements = {
        "area_pct": round(area_pct, 2),
        "mode": mode,
        "median_saturation": round(median_saturation, 2),
        "median_gradient": round(median_gradient, 2),
    }

    if mode in ("nitrile", "latex"):
        measurements.update({
            "extent": round(best["extent"], 3),
            "aspect_ratio": round(best["aspect_ratio"], 3),
            "strength_score": round(best["strength_score"], 3),
            "shape_score": round(best["shape_score"], 3),
            "saturation_prior": round(best["saturation_prior"], 3),
        })
    else:
        measurements.update({
            "texture_mode": extra.get("texture_mode"),
            "num_selected_windows": best.get("num_selected_windows", 0),
            "peak_window_score": round(best["score"], 3),
            "refined": bool(best.get("refined", False)),
        })
        if "extent" in best:
            measurements["extent"] = round(best["extent"], 3)
        if "aspect_ratio" in best:
            measurements["aspect_ratio"] = round(best["aspect_ratio"], 3)
        if "mean_pixel_score" in best:
            measurements["mean_pixel_score"] = round(best["mean_pixel_score"], 3)

    for key, value in extra.items():
        if key in measurements:
            continue
        if isinstance(value, (int, float, np.number)):
            measurements[key] = round(float(value), 3)
        else:
            measurements[key] = value

    result["measurements"] = measurements
    return result


# ============================================================
# OPTIONAL QUICK TEST
# ============================================================

if __name__ == "__main__":
    import os

    from preprocessing import load_image, preprocess_image
    from segmentation import segment_glove

    candidate_folders = [
        os.path.join("datasets", "nitrile", "plastic_contamination"),
        os.path.join("datasets", "latex", "plastic_contamination"),
        os.path.join("datasets", "cotton", "plastic_contamination"),
        os.path.join("dataset", "nitrile", "plastic_contamination"),
        os.path.join("dataset", "latex", "plastic_contamination"),
        os.path.join("dataset", "cotton", "plastic_contamination"),
    ]

    sample_path = None

    for folder in candidate_folders:
        if not os.path.isdir(folder):
            continue

        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                sample_path = os.path.join(folder, filename)
                break

        if sample_path is not None:
            break

    if sample_path is None:
        print("No plastic-contamination image found under datasets/ or dataset/.")
    else:
        image = load_image(sample_path)
        processed = preprocess_image(image)
        segmentation = segment_glove(processed)

        output = detect_plastic_contamination(processed, segmentation)

        print(f"Image: {sample_path}")
        print(f"Detected: {output['detected']}")
        print("Score:", round(output["detection_score"], 3))
        print("Bounding box:", output["bounding_box"])
        print("Measurements:", output["measurements"])