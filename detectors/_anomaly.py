"""Shared colour-anomaly processing for the dirty, stain and spotting detectors.

Dirty, stain and spotting are all foreign material sitting on an otherwise
uniform glove surface, and all three are darker than the glove. They share the
first question - which pixels do not look like glove material? - and differ only
in the second - what shape is that material? Separating the two questions means
the hard part (telling contamination apart from shading, wrinkles and
segmentation leakage) is solved and justified once here, and the three detectors
stay small enough to read.

Not a detector. The leading underscore keeps it out of DETECTOR_REGISTRY.
"""

import cv2
import numpy as np


# A glove is curved, so the last few millimetres before its silhouette are always
# in self-shadow, and the segmentation boundary itself sits on a colour ramp
# between glove and background. Both look exactly like "a dark region on the
# glove" to a colour test, and in testing they were the single largest source of
# false positives. The margin scales with the glove's own equivalent radius
# rather than the image size, so it does not assume a fixed framing.
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


# Flat-field illumination model: what would this pixel look like if the glove
# were clean here? Comparing against a LOCAL estimate rather than one global
# average is what makes the detectors tolerant of uneven lighting, which the
# brief explicitly requires.
#
# A median filter is used rather than the two more obvious alternatives, both of
# which were tried first and failed:
#   - morphological closing takes a local maximum, so one specular highlight
#     propagates across its whole neighbourhood and makes the entire glove read
#     as "too dark" by comparison. This drove some test images to 99% anomaly
#     area.
#   - a Gaussian blur has the opposite failure: large dark defects drag the
#     estimate down and so conceal themselves.
# A median ignores both as long as they occupy less than half the window.
#
# Two further robustness measures: highlights are clipped before filtering so the
# brightest speculars cannot bias the estimate at all, and a second pass lifts
# pixels darker than the first estimate up to it, because a large dirty patch can
# occupy more than half a window and defeat a single median pass.
#
# Filtering runs on a downsampled copy: the window must be large compared with
# the defects, and a large median at full resolution is far too slow for an
# interactive GUI.
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


# Splitting the LAB difference into a lightness term and a chroma term keeps the
# two cues independent, so a rust-coloured speck that is barely darker than pale
# latex is still caught by chroma, and a black ink streak on blue nitrile is
# still caught by lightness.
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
        "lab": lab,
        "dark_abs": dark_abs,
        "dark_rel": dark_rel,
        "chroma": chroma,
        "gradient": gradient,
        "L": L,
        "background_L": background_L,
    }


# Two decisions matter here.
#
# First, the darkening test is a CONJUNCTION of an absolute and a relative
# threshold. The relative term alone fires on dark gloves, where a few L units is
# a large fraction; the absolute term alone fires on bright gloves under uneven
# light. Requiring both keeps the test stable across nitrile, latex and cotton.
#
# Second, the thresholds are fixed constants rather than a per-image adaptive
# rule such as Otsu. Otsu always returns a split, so on a perfectly clean glove it
# manufactures a "defect" out of ordinary shading. Since the system must be able
# to answer "no defect here", an absolute physical threshold is the correct
# choice and a data-driven one is not.
#
# The opening removes wrinkle and crease lines, which are one or two pixels wide
# and survive thresholding on latex and nitrile; a defect wide enough to matter
# survives it.
def background_reference(processed, glove_mask):
    """
    Median LAB colour of whatever segmentation decided was NOT glove.

    Used to recognise segmentation leakage. When the glove mask over-reaches
    and swallows a patch of the photographic background, that patch is darker
    and differently coloured than the glove, so every colour test flags it as a
    large, confident defect. It is not one.

    A region that matches the background colour almost certainly IS background,
    so this reference lets such regions be discarded without re-running or
    second-guessing segmentation - it uses segmentation's own output as the
    reference. Real contamination sits on the glove and does not match the
    backdrop.

    The mask is dilated before sampling so the colour ramp at the glove edge is
    excluded from the estimate.
    """
    outside = glove_mask == 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    grown = cv2.dilate((glove_mask > 0).astype(np.uint8), kernel)
    outside = outside & (grown == 0)
    if np.count_nonzero(outside) < 500:
        return None
    lab = processed["lab"].astype(np.float32)
    return np.array([
        float(np.median(lab[:, :, 0][outside])),
        float(np.median(lab[:, :, 1][outside])),
        float(np.median(lab[:, :, 2][outside])),
    ], dtype=np.float32)


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


# Every downstream decision is made from this feature table, so each measurement
# is chosen to separate the three defects from each other and from the benign
# structures that survive thresholding.
#
# edge_contact does most of the false-positive suppression. Surface contamination
# is SURROUNDED by glove material: a stain has clean glove on every side. Two of
# the three things that survive thresholding do not have that property - forearm
# skin that segmentation failed to cut away runs off the bottom of the analysed
# area, and the shading band along a curved finger runs along its silhouette.
# Both therefore have most of their perimeter on the boundary, and one ratio
# separates them from real defects without modelling skin colour or redoing any
# segmentation.
def describe_blobs(candidate, fields, interior_mask, min_area=20, background_lab=None):
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

        # Distance from this region's own colour to the photographic
        # background. Small means the region is almost certainly background
        # that segmentation included by mistake, not contamination.
        if background_lab is None:
            bg_distance = 1e6
        else:
            lab = fields["lab"]
            region_lab = np.array([
                float(np.median(lab[:, :, 0][region])),
                float(np.median(lab[:, :, 1][region])),
                float(np.median(lab[:, :, 2][region])),
            ], dtype=np.float32)
            bg_distance = float(np.linalg.norm(region_lab - background_lab))

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
            "bg_distance": bg_distance,
            "texture": texture,
        })

    return blobs


def accept_blobs(blobs, max_edge_contact=0.30,
                 compact_area_frac=0.020, compact_edge_contact=0.60,
                 min_bg_distance=14.0):
    """
    Drop regions that are not surrounded by glove material.

    The structures this rejects - forearm skin the segmentation failed to cut,
    and the shading band along a curved finger - are all LARGE regions that run
    along the boundary. A small compact defect can also touch the boundary
    heavily without being either of those, simply because it sits inside a
    narrow structure: a smudge on a fingertip is only a few millimetres from
    the silhouette on three sides, and a single flat threshold discards it.

    The limit is therefore relaxed for regions below `compact_area_frac` of the
    glove. Size is what separates the two cases: a skin band is a substantial
    fraction of the glove, a fingertip smudge is not.
    """
    kept = []
    for blob in blobs:
        # Regions whose colour matches the photographic backdrop are
        # segmentation leakage, not contamination.
        if blob.get("bg_distance", 1e6) < min_bg_distance:
            continue
        limit = (compact_edge_contact
                 if blob["area_frac"] < compact_area_frac
                 else max_edge_contact)
        if blob["edge_contact"] < limit:
            kept.append(blob)
    return kept


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


# Shadow rejection constants.
#
# A cast shadow scales the illumination reaching a surface, so it moves lightness
# but leaves the surface hue essentially untouched: its a/b deviation stays near
# zero. Foreign material has its own reflectance and does shift a/b - brown
# liquid, rust-coloured specks, black ink pulling blue nitrile towards neutral.
#
# a/b distance at which a pixel counts as genuinely discoloured rather than merely
# shaded. Matches the chroma_min used to build the candidate mask, so the two
# tests are expressed in the same units.
CHROMA_EVIDENCE = 8.0

# Ramp over which colour evidence is treated as convincing. Placed from the
# measured distributions: clear shadow cases score near zero (bunched cotton
# 0.000, creased latex 0.065) and strongly coloured defects score 0.6-0.96, but an
# overlap band near 0.09-0.18 contains BOTH the weakest true positive (small brown
# marks on pale latex, 0.089) and the strongest shadow case (folded cotton, 0.099).
CHROMATIC_RAMP = (0.02, 0.20)

# How much of the score a region with no colour evidence at all retains. Because of
# the overlap above, the test is applied as a soft multiplier rather than a hard
# veto: a veto set tight enough to catch the folded-cotton case would also reject
# the true stain, i.e. it would remove more true positives than false ones. The
# floor is also high enough that genuinely achromatic contamination - grey dust on
# a white cotton glove - can still be detected on shape evidence alone.
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
    """Smallest box containing every accepted region, or None."""
    if not blobs:
        return None
    xs0 = min(b["bbox"][0] for b in blobs)
    ys0 = min(b["bbox"][1] for b in blobs)
    xs1 = max(b["bbox"][0] + b["bbox"][2] for b in blobs)
    ys1 = max(b["bbox"][1] + b["bbox"][3] for b in blobs)
    return (int(xs0), int(ys0), int(xs1 - xs0), int(ys1 - ys0))


def localisation_blobs(blobs, cuff_fraction=0.88):
    """
    Regions eligible to carry the reported bounding box.

    On knit cotton the ribbed cuff is a different structure from the glove
    body: it is darker, coarser, and it traps grime, so it thresholds into one
    of the largest regions on the image. On a heavily soiled glove that single
    cuff region outweighs the real soiling on the fingers and palm, and the box
    is drawn around the wrist while the visible dirt sits outside it.

    The cuff is excluded from box selection only - never from scoring. Two
    reasons. A soiled cuff is genuinely soiled, so removing it from the score
    would be wrong and it costs a true positive when tried. And the cuff is
    already another detector's subject (incomplete_beading), so it is not this
    detector's job to describe it. Detection therefore sees the whole glove
    while the box points at the part a user needs to look at.

    If every region falls in the cuff band the full set is returned, so a glove
    soiled only at the wrist still gets a box.
    """
    if not blobs:
        return blobs
    top = min(b["bbox"][1] for b in blobs)
    bottom = max(b["bbox"][1] + b["bbox"][3] for b in blobs)
    limit = top + cuff_fraction * (bottom - top)
    eligible = [b for b in blobs if b["centroid"][1] < limit]
    return eligible if eligible else blobs


def dominant_bbox(blobs, equivalent_radius, gap_ratio=0.10):
    """
    Box around the strongest concentration of defect evidence.

    A union box over every accepted region is a poor way to show a user where a
    defect is. Scattered weak responses - weave texture, residual shading -
    stretch it across most of the glove, so the box ends up highlighting
    everything and therefore nothing, and can easily fail to sit on the actual
    mark at all.

    Instead, regions closer together than `gap_ratio` of the glove radius are
    grouped, and the group carrying the most evidence is boxed. Evidence is
    weighted by the SQUARE of each region's contrast, so a small, deeply
    coloured mark outranks a large, faint one: a fingertip smudge at 0.39 mean
    darkening beats a broad shading band at 0.26 covering ten times the area.
    Contrast is what makes something a defect rather than a gradient, so it is
    what the box should follow.

    The full extent stays visible through the returned mask, which the GUI fills
    semi-transparently; this only decides where the rectangle goes.
    """
    if not blobs:
        return None
    if len(blobs) == 1:
        return tuple(int(v) for v in blobs[0]["bbox"])

    gap = max(8.0, gap_ratio * equivalent_radius)

    # Single-link grouping on bounding-box separation.
    groups = []
    for blob in blobs:
        x, y, w, h = blob["bbox"]
        placed = False
        for group in groups:
            for other in group:
                ox, oy, ow, oh = other["bbox"]
                dx = max(0, max(ox - (x + w), x - (ox + ow)))
                dy = max(0, max(oy - (y + h), y - (oy + oh)))
                if (dx * dx + dy * dy) ** 0.5 <= gap:
                    group.append(blob)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append([blob])

    def evidence(group):
        # Area weighted by the square of contrast. Neither term alone works:
        # pure area follows residual shading and cuffs, which are broad; pure
        # contrast follows whichever single region is darkest, which on a knit
        # glove can be the cuff rather than the mark. This is a compromise, and
        # on images where a large low-contrast structure survives thresholding
        # it still loses to that structure - see LIMITATIONS.
        return sum(b["area"] * (b["mean_dark"] ** 2) for b in group)

    return union_bbox(max(groups, key=evidence))


def background_guarded_interior(processed, glove_mask, tolerance=25.0):
    """
    Build the analysed region, excluding pixels that are photographic background.

    Segmentation sometimes over-reaches and swallows a strip of the backdrop -
    on cotton_dirty_1 the mask extends so far past the glove's right edge that
    the true silhouette sits roughly a hundred pixels INSIDE the mask. The
    glove's own dark rim is then interior rather than boundary, so neither the
    distance-transform margin nor the perimeter-contact test can remove it, and
    it thresholds into a single region ten times the size of the real defect.
    Because that region is large, it dominates any area-weighted localisation
    and the reported box lands on the glove edge instead of on the mark.

    The fix is to remove background-coloured pixels from the analysed region
    BEFORE thresholding, so the spurious rim never forms a region at all.
    Segmentation's own output supplies the reference colour, so this does not
    re-run or second-guess segmentation - it only declines to analyse pixels
    that segmentation itself would call background if it looked again.

    After masking, a small opening removes speckle and only the largest
    connected component is kept, so the analysed region stays a single glove
    rather than a scatter of fragments.

    Parameters
    ----------
    processed : dict
        Output of preprocessing.preprocess_image().
    glove_mask : numpy.ndarray
        Binary glove mask (0/255) from segmentation.py.
    tolerance : float
        LAB distance below which a pixel counts as background. Set from the
        measured gap between the backdrop and genuine contamination; at 25 the
        spurious rim disappears while every true defect survives, and raising
        it further starts to erode real marks.

    Returns
    -------
    numpy.ndarray
        Binary interior mask (0/255), already margin-eroded.
    """
    background_lab = background_reference(processed, glove_mask)
    mask = (glove_mask > 0).astype(np.uint8) * 255

    if background_lab is not None:
        lab = processed["lab"].astype(np.float32)
        distance = np.sqrt(((lab - background_lab.reshape(1, 1, 3)) ** 2).sum(axis=2))
        mask[distance < tolerance] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8) * 255

    return glove_interior(mask)


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
    interior = background_guarded_interior(processed, segmentation["glove_mask"])
    if interior is None or int(np.count_nonzero(interior)) < 500:
        empty = np.zeros(processed["gray"].shape[:2], dtype=np.uint8)
        return None, empty, [], empty

    fields = anomaly_fields(processed, interior)
    candidate = candidate_mask(fields, interior, **candidate_kwargs)
    background_lab = background_reference(processed, segmentation["glove_mask"])
    blobs = describe_blobs(candidate, fields, interior, background_lab=background_lab)
    return fields, candidate, blobs, interior
