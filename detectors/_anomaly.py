"""Shared colour-anomaly processing for the dirty, stain and spotting detectors."""

import cv2
import numpy as np


def glove_interior(glove_mask, margin_ratio=0.09, min_margin=10, max_margin=40):
    """Remove a scale-aware margin from the glove silhouette.

    Parameters
    ----------
    glove_mask : numpy.ndarray
        Binary glove mask (0/255) from segmentation.py.
    margin_ratio : float
        Margin as a fraction of the glove's equivalent radius.
    min_margin, max_margin : int
        Absolute clamps in pixels.

    Returns
    -------
    numpy.ndarray
        Binary interior mask (0/255).
    """
    if glove_mask is None:
        return None

    binary = (glove_mask > 0).astype(np.uint8)
    area = int(binary.sum())
    if area <= 0:
        return np.zeros_like(binary, dtype=np.uint8)

    equivalent_radius = np.sqrt(area / np.pi)
    margin = int(np.clip(margin_ratio * equivalent_radius, min_margin, max_margin))

    # Use distance thresholding for scale-independent erosion.
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    interior = (distance > margin).astype(np.uint8) * 255

    # Fall back to lighter erosion when the normal margin removes too much.
    if int(np.count_nonzero(interior)) < 0.25 * area:
        interior = (distance > max(3, margin // 3)).astype(np.uint8) * 255

    return interior


def robust_background(channel, mask, scale=8, ksize=11, passes=2):
    """Estimate a smooth, defect-free version of one LAB channel.

    Parameters
    ----------
    channel : numpy.ndarray
        Single LAB channel as float32.
    mask : numpy.ndarray
        Binary interior mask (0/255).
    scale : int
        Downsampling factor used for the median filter.
    ksize : int
        Median kernel size at the reduced scale (odd).
    passes : int
        Number of re-estimation passes.

    Returns
    -------
    numpy.ndarray
        Smooth background estimate, same shape as `channel`, float32.
    """
    selected = mask > 0
    if not selected.any():
        return np.full_like(channel, float(np.median(channel)), dtype=np.float32)

    values = channel[selected]
    median_value = float(np.median(values))
    highlight_cap = float(np.percentile(values, 97.0))

    # Clip highlights and mask the background so neither biases the estimate.
    filled = np.minimum(channel.astype(np.float32), highlight_cap)
    filled[~selected] = median_value

    height, width = channel.shape[:2]
    small_h = max(8, height // scale)
    small_w = max(8, width // scale)

    working = filled
    background = filled
    for _ in range(max(1, passes)):
        small = cv2.resize(working, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small = np.clip(small, 0, 255).astype(np.uint8)
        smoothed = cv2.medianBlur(small, ksize).astype(np.float32)
        background = cv2.resize(smoothed, (width, height), interpolation=cv2.INTER_LINEAR)
        background = cv2.GaussianBlur(background, (0, 0), scale)
        # Lift dark pixels so large defects do not bias the next pass.
        working = np.maximum(filled, background)

    return background


def anomaly_fields(processed, interior_mask):
    """Build lightness, chroma and edge maps for anomaly detection.

    Parameters
    ----------
    processed : dict
        Preprocessed image channels.
    interior_mask : numpy.ndarray
        Binary glove interior mask.

    Returns
    -------
    dict
        dark_abs  : how many L units darker than the local background
        dark_rel  : the same darkening as a fraction of the local
                    background, which removes exposure dependence
        chroma    : Euclidean a/b distance from the local background
        gradient  : Sobel magnitude of L, used later to measure how
                    sharply a region's edge is defined
        L, background_L : the lightness channel and its estimate
    """
    lab = processed["lab"].astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    background_L = robust_background(L, interior_mask, passes=2)
    background_A = robust_background(A, interior_mask, passes=1)
    background_B = robust_background(B, interior_mask, passes=1)

    dark_abs = np.clip(background_L - L, 0.0, None)
    dark_rel = dark_abs / (background_L + 1e-3)
    chroma = np.sqrt((A - background_A) ** 2 + (B - background_B) ** 2)

    smoothed_L = cv2.GaussianBlur(L, (0, 0), 1.2)
    gx = cv2.Sobel(smoothed_L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed_L, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    return {
        "dark_abs": dark_abs,
        "dark_rel": dark_rel,
        "chroma": chroma,
        "gradient": gradient,
        "L": L,
        "background_L": background_L,
    }


def candidate_mask(fields, interior_mask,
                   dark_abs_min=12.0, dark_rel_min=0.10, chroma_min=9.0,
                   open_radius=2, close_radius=2):
    """Threshold anomaly maps and clean the result morphologically.

    Parameters
    ----------
    fields : dict
        Anomaly maps from :func:`anomaly_fields`.
    interior_mask : numpy.ndarray
        Binary glove interior mask.
    dark_abs_min, dark_rel_min, chroma_min : float
        Lightness and chroma thresholds.
    open_radius, close_radius : int
        Morphological kernel radii.

    Returns
    -------
    numpy.ndarray
        Binary candidate mask.
    """
    dark_hit = (fields["dark_abs"] > dark_abs_min) & (fields["dark_rel"] > dark_rel_min)
    colour_hit = fields["chroma"] > chroma_min

    candidate = ((dark_hit | colour_hit) & (interior_mask > 0)).astype(np.uint8) * 255

    if open_radius > 0:
        size = 2 * open_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    if close_radius > 0:
        size = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    return candidate


def describe_blobs(candidate, fields, interior_mask, min_area=20):
    """Describe connected candidate regions using shape and contrast features.

    Parameters
    ----------
    candidate : numpy.ndarray
        Binary candidate mask.
    fields : dict
        Anomaly maps from :func:`anomaly_fields`.
    interior_mask : numpy.ndarray
        Binary glove interior mask.
    min_area : int
        Minimum connected-component area.

    Returns
    -------
    list
        Feature dictionaries for candidate regions.
    """
    blobs = []
    if candidate is None:
        return blobs

    glove_area = max(1, int(np.count_nonzero(interior_mask)))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)

    dark_rel = fields["dark_rel"]
    gradient = fields["gradient"]
    L = fields["L"]

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])

        region = (labels == index)

        # Measure shape from each region's own contour.
        patch = region[y:y + h, x:x + w].astype(np.uint8) * 255
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        circularity = float(4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        circularity = min(circularity, 1.0)

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = float(area / hull_area) if hull_area > 0 else 0.0

        # Pad the patch so boundary measurements are not clipped.
        height, width = region.shape[:2]
        px0, py0 = max(0, x - 1), max(0, y - 1)
        px1, py1 = min(width, x + w + 1), min(height, y + h + 1)
        padded = region[py0:py1, px0:px1].astype(np.uint8) * 255
        dilated = cv2.dilate(padded, np.ones((3, 3), np.uint8))
        ring = (dilated > 0) & (padded == 0)
        ring_full = np.zeros(region.shape, dtype=bool)
        ring_full[py0:py1, px0:px1] = ring

        depth = float(dark_rel[region].mean())
        edge_sharp = float(gradient[ring_full].mean()) if ring_full.any() else 0.0
        edge_sharp = edge_sharp / (255.0 * depth + 1e-3)

        # Edge contact rejects regions not surrounded by glove material.
        if ring_full.any():
            edge_contact = float(np.mean(interior_mask[ring_full] == 0))
        else:
            edge_contact = 0.0

        values = L[region]
        texture = float(values.std() / (values.mean() + 1e-3))

        blobs.append({
            "area": area,
            "area_frac": area / glove_area,
            "bbox": (x, y, w, h),
            "centroid": (float(centroids[index][0]), float(centroids[index][1])),
            "circularity": circularity,
            "elongation": max(w, h) / max(1.0, min(w, h)),
            "solidity": solidity,
            "mean_dark": depth,
            "edge_sharp": edge_sharp,
            "edge_contact": edge_contact,
            "texture": texture,
        })

    return blobs


def accept_blobs(blobs, max_edge_contact=0.30):
    """Keep regions whose boundary is mostly surrounded by glove material.

    Parameters
    ----------
    blobs : list
        Region feature dictionaries.
    max_edge_contact : float
        Maximum allowed boundary contact ratio.

    Returns
    -------
    list
        Accepted region dictionaries.
    """
    return [b for b in blobs if b["edge_contact"] < max_edge_contact]


def cloud_statistics(blobs, defect_mask, fields, interior_mask):
    """Summarise accepted regions with aggregate shape and colour statistics.

    Parameters
    ----------
    blobs : list
        Accepted region feature dictionaries.
    defect_mask : numpy.ndarray
        Binary mask containing accepted regions.
    fields : dict
        Anomaly maps from :func:`anomaly_fields`.
    interior_mask : numpy.ndarray
        Binary glove interior mask.

    Returns
    -------
    dict
        Aggregate measurements used by the detectors.
    """
    glove_area = max(1, int(np.count_nonzero(interior_mask)))
    equivalent_radius = float(np.sqrt(glove_area / np.pi))

    if not blobs:
        return {
            "count": 0, "area_frac": 0.0, "median_area": 0.0,
            "max_area_frac": 0.0, "circularity": 0.0, "elongation": 0.0,
            "solidity": 0.0, "mean_dark": 0.0, "ring_dark": 0.0,
            "isolation": 0.0, "texture": 0.0, "edge_sharp": 0.0,
            "dispersion": 0.0, "chromatic_fraction": 0.0, "chroma_p90": 0.0,
            "peak_dark": 0.0,
            "glove_area": glove_area,
            "equivalent_radius": equivalent_radius,
        }

    dark_rel = fields["dark_rel"]
    height, width = interior_mask.shape[:2]

    ring_values = []
    for blob in blobs:
        x, y, w, h = blob["bbox"]
        pad = max(4, int(0.4 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        # Exclude all accepted regions so dense clusters do not measure themselves.
        ring = (defect_mask[y0:y1, x0:x1] == 0) & (interior_mask[y0:y1, x0:x1] > 0)
        if ring.any():
            ring_values.append(float(dark_rel[y0:y1, x0:x1][ring].mean()))

    ring_dark = float(np.median(ring_values)) if ring_values else 0.0
    areas = np.array([b["area"] for b in blobs], dtype=float)
    centroids = np.array([b["centroid"] for b in blobs], dtype=float)
    mean_dark = float(np.median([b["mean_dark"] for b in blobs]))

    # Use the same chroma scale for candidate selection and confidence.
    flagged = defect_mask > 0
    if flagged.any():
        chroma_values = fields["chroma"][flagged]
        chromatic_fraction = float(np.mean(chroma_values > CHROMA_EVIDENCE))
        chroma_p90 = float(np.percentile(chroma_values, 90))
    else:
        chromatic_fraction = 0.0
        chroma_p90 = 0.0
    if flagged.any():
        peak_dark = float(np.percentile(dark_rel[flagged], 90))
    else:
        peak_dark = 0.0

    return {
        "count": len(blobs),
        "area_frac": float(areas.sum() / glove_area),
        "median_area": float(np.median(areas)),
        "max_area_frac": float(areas.max() / glove_area),
        "circularity": float(np.median([b["circularity"] for b in blobs])),
        "elongation": float(np.median([b["elongation"] for b in blobs])),
        "solidity": float(np.median([b["solidity"] for b in blobs])),
        "mean_dark": mean_dark,
        "ring_dark": ring_dark,
        "isolation": float(mean_dark / (ring_dark + 1e-3)),
        "texture": float(np.median([b["texture"] for b in blobs])),
        "edge_sharp": float(np.median([b["edge_sharp"] for b in blobs])),
        "dispersion": float(np.sqrt(centroids.var(axis=0).sum()) / (equivalent_radius + 1e-6)),
        "chromatic_fraction": chromatic_fraction,
        "chroma_p90": chroma_p90,
        "peak_dark": peak_dark,
        "glove_area": glove_area,
        "equivalent_radius": equivalent_radius,
    }


# Chroma distance required for colour evidence.
CHROMA_EVIDENCE = 8.0

# Range over which colour evidence increases confidence.
CHROMATIC_RAMP = (0.02, 0.20)

# Minimum score multiplier when colour evidence is absent.
SHADOW_FLOOR = 0.65


def chromatic_confidence(stats):
    """Return a soft confidence multiplier based on chroma evidence.

    Parameters
    ----------
    stats : dict
        Aggregate measurements including chromatic fraction.

    Returns
    -------
    float
        Multiplier in [SHADOW_FLOOR, 1.0] to apply to a raw score.
    """
    confidence = ramp(stats["chromatic_fraction"], *CHROMATIC_RAMP)
    return SHADOW_FLOOR + (1.0 - SHADOW_FLOOR) * confidence


def ramp(value, low, high):
    """Map a measurement onto 0..1 using a linear ramp.

    Parameters
    ----------
    value : float
        Measurement to scale.
    low, high : float
        Ramp endpoints.

    Returns
    -------
    float
        Clipped scaled value.
    """
    if high == low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def combine(weighted_terms):
    """Combine weighted evidence terms into a single 0..1 score.

    Parameters
    ----------
    weighted_terms : list
        List of ``(weight, value)`` pairs.

    Returns
    -------
    float
        Clipped weighted mean.
    """
    total_weight = sum(weight for weight, _ in weighted_terms)
    if total_weight <= 0:
        return 0.0
    total = sum(weight * value for weight, value in weighted_terms)
    return float(np.clip(total / total_weight, 0.0, 1.0))


def blobs_to_mask(blobs, candidate, shape):
    """Rebuild a binary mask containing only accepted regions.

    Parameters
    ----------
    blobs : list
        Accepted region dictionaries.
    candidate : numpy.ndarray
        Original candidate mask.
    shape : tuple
        Output image shape.

    Returns
    -------
    numpy.ndarray
        Binary mask of accepted regions.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not blobs:
        return mask
    for blob in blobs:
        x, y, w, h = blob["bbox"]
        patch = candidate[y:y + h, x:x + w]
        mask[y:y + h, x:x + w] = np.maximum(mask[y:y + h, x:x + w], patch)
    return mask


def union_bbox(blobs):
    """Return the smallest box containing all accepted regions.

    Parameters
    ----------
    blobs : list
        Region dictionaries containing bounding boxes.

    Returns
    -------
    tuple or None
        Bounding box, or ``None`` when no regions are supplied.
    """
    if not blobs:
        return None
    xs0 = min(b["bbox"][0] for b in blobs)
    ys0 = min(b["bbox"][1] for b in blobs)
    xs1 = max(b["bbox"][0] + b["bbox"][2] for b in blobs)
    ys1 = max(b["bbox"][1] + b["bbox"][3] for b in blobs)
    return (int(xs0), int(ys0), int(xs1 - xs0), int(ys1 - ys0))


def prepare(processed, segmentation, **candidate_kwargs):
    """Run the shared anomaly-processing pipeline used by each detector.

    Parameters
    ----------
    processed : dict
        Output of preprocessing.
    segmentation : dict
        Output of glove segmentation.
    **candidate_kwargs
        Optional candidate-mask thresholds.

    Returns
    -------
    tuple(dict or None, numpy.ndarray, list, numpy.ndarray)
        Fields, candidate mask, blob table and interior mask.
    """
    interior = glove_interior(segmentation["glove_mask"])
    if interior is None or int(np.count_nonzero(interior)) < 500:
        empty = np.zeros(processed["gray"].shape[:2], dtype=np.uint8)
        return None, empty, [], empty

    fields = anomaly_fields(processed, interior)
    candidate = candidate_mask(fields, interior, **candidate_kwargs)
    blobs = describe_blobs(candidate, fields, interior)
    return fields, candidate, blobs, interior
