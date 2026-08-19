"""
detectors/_anomaly.py
---------------------
Shared colour-anomaly front-end for the three surface-contamination
defects owned by Ka Sin: dirty, stain and spotting.

This is a HELPER module, not a detector. It is deliberately prefixed
with an underscore so evaluate.py's DETECTOR_REGISTRY never picks it
up as a defect.

Why a shared front-end
----------------------
Dirty, stain and spotting are all "foreign material sitting on an
otherwise uniform glove surface". They therefore share the same first
question - *which pixels do not look like the glove material?* - and
differ only in the second question - *what shape and contrast does
that foreign material have?*

Splitting the problem this way means the expensive, error-prone part
(separating true surface contamination from shading, wrinkles and
segmentation leakage) is written, tuned and justified once, and the
three detectors stay small and readable.

Pipeline
--------
    glove_mask (from segmentation.py)
        v
    1. glove_interior()      distance-transform margin - drop the dark
                             self-shadow rim at the silhouette edge
        v
    2. robust_background()   flat-field estimate of L, a, b from the
                             glove itself (illumination model)
        v
    3. anomaly_fields()      per-pixel darkening + colour-shift maps
        v
    4. candidate_mask()      threshold + morphology (wrinkle removal)
        v
    5. describe_blobs()      per-region shape/contrast feature table

Nothing here re-implements resizing, denoising or background removal -
those belong to preprocessing.py and segmentation.py.
"""

import cv2
import numpy as np


# ============================================================
# 1. INTERIOR MASK
# ============================================================

def glove_interior(glove_mask, margin_ratio=0.09, min_margin=10, max_margin=40):
    """
    Shrink the glove mask away from its own silhouette edge.

    A glove is a curved object, so the last few millimetres before the
    silhouette are always in self-shadow, and the segmentation boundary
    itself sits on a colour ramp between glove and background. Both
    effects look exactly like "a dark region on the glove" to a colour
    anomaly test, and in practice they were the single largest source
    of false positives.

    The margin is derived from the glove's own equivalent radius rather
    than from the image size, so it scales with how much of the frame
    the glove occupies instead of assuming a fixed framing.

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

    # Distance transform gives, for every glove pixel, its distance to
    # the nearest non-glove pixel. Thresholding it is an exact,
    # shape-independent erosion by `margin` pixels.
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    interior = (distance > margin).astype(np.uint8) * 255

    # Safety net: if the glove is thin enough that the margin removes
    # nearly everything, back off to a light erosion so the detector
    # still has something to work with.
    if int(np.count_nonzero(interior)) < 0.25 * area:
        interior = (distance > max(3, margin // 3)).astype(np.uint8) * 255

    return interior


# ============================================================
# 2. ILLUMINATION BACKGROUND
# ============================================================

def robust_background(channel, mask, scale=8, ksize=11, passes=2):
    """
    Estimate the smooth, defect-free version of one LAB channel.

    This is a flat-field / illumination model: it answers "what would
    this pixel look like if the glove were clean here?". Comparing each
    pixel against a *local* estimate instead of one global average is
    what makes the detectors tolerant of uneven lighting, which the
    brief explicitly requires.

    Robustness comes from three choices:

    - A median filter, not a blur or a morphological closing. A closing
      takes a local maximum, so a single specular highlight propagates
      across its whole neighbourhood and makes the entire glove look
      "too dark" by comparison. A plain blur has the opposite problem:
      large dark defects drag the estimate down and hide themselves. A
      median ignores both as long as they occupy less than half the
      window.
    - Highlight clipping before filtering, so the brightest specular
      pixels cannot bias the estimate at all.
    - A second pass in which pixels darker than the first estimate are
      lifted up to it. Large dirty patches can occupy more than half of
      a window and defeat a single median pass; after one pass they no
      longer pull the estimate down.

    The filtering is done on a downsampled copy because the window has
    to be large compared with the defects, and a large median on a full
    resolution image is far too slow for an interactive GUI.

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

    # Clip speculars, and neutralise everything outside the glove so
    # the background colour never leaks into the estimate.
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
        # Lift dark pixels to the current estimate so the next pass is
        # not dragged down by large defects.
        working = np.maximum(filled, background)

    return background


# ============================================================
# 3. ANOMALY FIELDS
# ============================================================

def anomaly_fields(processed, interior_mask):
    """
    Build the per-pixel maps the three detectors reason about.

    All three defects are foreign material that is *darker* than the
    glove, and some of them are also a different *hue*. Splitting the
    LAB difference into a lightness term and a chroma term keeps those
    two cues independent, so a rust-coloured speck that is barely
    darker than pale latex is still caught by the chroma term, and a
    black ink streak on blue nitrile is still caught by the lightness
    term.

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


# ============================================================
# 4. CANDIDATE MASK
# ============================================================

def candidate_mask(fields, interior_mask,
                   dark_abs_min=12.0, dark_rel_min=0.10, chroma_min=9.0,
                   open_radius=2, close_radius=2):
    """
    Threshold the anomaly maps into a binary candidate mask.

    Two design decisions matter here.

    First, the darkening test is a *conjunction* of an absolute and a
    relative threshold. The relative term alone would fire on dark
    gloves, where a few L units is a large fraction; the absolute term
    alone would fire on bright gloves under uneven light. Requiring
    both keeps the test stable across nitrile, latex and cotton.

    Second, the thresholds are fixed constants rather than a
    per-image adaptive rule such as Otsu. Otsu always returns a split,
    so on a perfectly clean glove it manufactures a "defect" out of
    ordinary shading. Since the system has to be able to answer "no
    defect here", an absolute physical threshold is the correct choice
    and a data-driven one is not.

    The morphological opening removes wrinkle and crease lines, which
    are one or two pixels wide and survive thresholding on latex and
    nitrile; a defect wide enough to matter survives it.
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


# ============================================================
# 5. BLOB DESCRIPTION
# ============================================================

def describe_blobs(candidate, fields, interior_mask, min_area=20):
    """
    Turn the candidate mask into a table of measured regions.

    Every downstream decision is made from these numbers, so each one
    is chosen to separate the three defects from each other and from
    the benign structures that survive thresholding:

    area_frac    fraction of the glove the region covers
    circularity  4*pi*A / P^2 - near 1 for a round speck, low for a
                 streak or a ragged smear
    elongation   major/minor axis ratio of the bounding box
    solidity     area / convex-hull area - a compact speck fills its
                 hull, a ragged dirt smear does not
    mean_dark    average relative darkening inside the region, i.e.
                 how strong the contamination is
    edge_sharp   mean L-gradient on the region's boundary, normalised
                 by its depth - an ink stain has a hard edge, a dust
                 smear fades out
    texture      standard deviation of L inside the region, normalised
                 - granular dirt is speckled, a liquid stain is flat
    edge_contact fraction of the region's perimeter that lies on the
                 edge of the analysed area rather than on clean glove

    `edge_contact` deserves a note, because it does most of the work in
    suppressing false positives. Surface contamination is *surrounded*
    by glove material: a stain has clean glove on every side of it. Two
    of the three things that survive thresholding do not have that
    property. Forearm skin that segmentation failed to cut away runs
    off the bottom of the analysed area, and the shading band along a
    curved finger runs along its silhouette. Both therefore have most
    of their perimeter on the boundary, and a single ratio separates
    them from real defects without needing to model skin colour or
    re-do any segmentation.
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

        # Perimeter and convex hull from the region's own contour.
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

        # Boundary ring of the region, for edge sharpness and edge
        # contact. The patch is taken with a one-pixel pad so the ring
        # is complete instead of being clipped at the bounding box.
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

        # How much of the region's perimeter sits on the edge of the
        # analysed area instead of on clean glove.
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


# ============================================================
# 6. REGION ACCEPTANCE AND CLOUD STATISTICS
# ============================================================

def accept_blobs(blobs, max_edge_contact=0.30):
    """
    Drop regions that are not surrounded by glove material.

    See the `edge_contact` note in describe_blobs. The threshold is
    deliberately generous: a defect that happens to sit near the edge
    of the hand still has most of its perimeter on clean glove, so it
    survives, while forearm skin and silhouette shading do not.
    """
    return [b for b in blobs if b["edge_contact"] < max_edge_contact]


def cloud_statistics(blobs, defect_mask, fields, interior_mask):
    """
    Summarise the accepted regions as a whole.

    Two of these numbers do most of the classification work.

    `ring_dark` is the average darkening of the glove immediately
    *around* the accepted regions, with the regions themselves and all
    their neighbours excluded. It answers the question that separates
    dirt from spots: is this region an isolated mark on clean glove, or
    one speck inside a broader soiled patch? Dirt darkens its own
    surroundings, so its ring is dark; an ink spot does not.

    `isolation` is the ratio of the regions' own darkening to that ring
    value. It is high for a discrete, well-defined mark and low for
    granular dirt that fades into a smear, and unlike an absolute
    darkness threshold it does not change when the exposure does.

    `chromatic_fraction` is the third key number and it exists to
    separate contamination from shadow. A cast shadow scales lightness
    down but leaves the surface hue essentially unchanged, so its a/b
    deviation from the local background stays near zero. Foreign
    material does change hue - brown coffee, rust-coloured spots, black
    ink that pulls blue nitrile towards neutral. Measuring the
    *fraction* of flagged pixels that show a real colour shift, rather
    than the average shift, keeps the number meaningful when a defect
    has both a strongly coloured core and a soft grey edge.
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
        # Exclude every accepted region, not just this one, so a dense
        # cluster of specks does not measure itself.
        ring = (defect_mask[y0:y1, x0:x1] == 0) & (interior_mask[y0:y1, x0:x1] > 0)
        if ring.any():
            ring_values.append(float(dark_rel[y0:y1, x0:x1][ring].mean()))

    ring_dark = float(np.median(ring_values)) if ring_values else 0.0
    areas = np.array([b["area"] for b in blobs], dtype=float)
    centroids = np.array([b["centroid"] for b in blobs], dtype=float)
    mean_dark = float(np.median([b["mean_dark"] for b in blobs]))

    # Colour evidence over all flagged pixels. CHROMA_EVIDENCE is the
    # a/b shift at which a pixel counts as genuinely discoloured rather
    # than merely shaded; it is the same scale as the chroma_min used
    # when the candidate mask was built.
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


# ============================================================
# 7. SHADOW REJECTION
# ============================================================

# a/b distance at which a pixel counts as genuinely discoloured rather
# than merely shaded. Matches the chroma_min used to build the
# candidate mask so the two tests speak the same units.
CHROMA_EVIDENCE = 8.0

# Ramp over which colour evidence is considered convincing.
CHROMATIC_RAMP = (0.02, 0.20)

# How much of the score a region with no colour evidence at all keeps.
# Not zero, because grey dust on a white cotton glove is genuinely
# close to achromatic and should still be able to register; low enough
# that a pure shadow cannot reach the 0.5 decision threshold on its own.
SHADOW_FLOOR = 0.65


def chromatic_confidence(stats):
    """
    How confident we are that the flagged regions are material, not shade.

    This is the single test that separates real surface contamination
    from the fold, crease and pose shadows that dominate the remaining
    false positives. The physical argument is standard: a cast shadow
    scales the illumination reaching a surface, which moves lightness
    but leaves the surface's own hue essentially untouched, so its a/b
    deviation from the local background stays near zero. Foreign
    material on the glove has its own reflectance and therefore does
    shift a/b - brown liquid, rust-coloured specks, or black ink that
    pulls blue nitrile towards neutral.

    The separation is real but not clean. Measured per detector on the
    group's own dataset, the clearest shadow cases score near zero
    (a bunched cotton glove at 0.000, a creased latex glove at 0.065)
    while strongly coloured defects score 0.6-0.96. In between sits an
    overlap band around 0.09-0.18 that contains both the weakest true
    positive (small brown marks on pale latex, 0.089) and the strongest
    shadow case (a folded cotton glove, 0.099).

    Because of that overlap the test is applied as a soft multiplier
    rather than a hard veto, and the floor is set high enough that a
    defect with genuinely achromatic contamination - grey dust on white
    cotton - can still be detected on shape evidence alone. Treating it
    as a veto would remove more true positives than false ones, which
    is why it is not one.

    Returns
    -------
    float
        Multiplier in [SHADOW_FLOOR, 1.0] to apply to a raw score.
    """
    confidence = ramp(stats["chromatic_fraction"], *CHROMATIC_RAMP)
    return SHADOW_FLOOR + (1.0 - SHADOW_FLOOR) * confidence


# ============================================================
# HELPERS SHARED BY THE THREE DETECTORS
# ============================================================

def ramp(value, low, high):
    """
    Map a measurement onto 0..1 with a linear ramp between two points.

    Used instead of a hard if/else so that a region sitting just under
    a threshold produces a low-but-non-zero score rather than
    disappearing. This is what lets detection_score behave like a
    confidence value in the GUI rather than a disguised boolean.
    """
    if high == low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def combine(weighted_terms):
    """
    Combine weighted evidence terms into a single 0..1 score.

    `weighted_terms` is a list of (weight, value) pairs. A plain
    weighted mean is used rather than a product so that one weak cue
    cannot veto an otherwise obvious defect.
    """
    total_weight = sum(weight for weight, _ in weighted_terms)
    if total_weight <= 0:
        return 0.0
    total = sum(weight * value for weight, value in weighted_terms)
    return float(np.clip(total / total_weight, 0.0, 1.0))


def blobs_to_mask(blobs, candidate, shape):
    """Rebuild a binary mask containing only the accepted regions."""
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


def prepare(processed, segmentation, **candidate_kwargs):
    """
    Run steps 1-5 in one call. Every detector starts with this.

    Returns
    -------
    tuple(dict or None, numpy.ndarray, list, numpy.ndarray)
        fields, candidate mask, blob table, interior mask.
        `fields` is None when the glove mask is unusable.
    """
    interior = glove_interior(segmentation["glove_mask"])
    if interior is None or int(np.count_nonzero(interior)) < 500:
        empty = np.zeros(processed["gray"].shape[:2], dtype=np.uint8)
        return None, empty, [], empty

    fields = anomaly_fields(processed, interior)
    candidate = candidate_mask(fields, interior, **candidate_kwargs)
    blobs = describe_blobs(candidate, fields, interior)
    return fields, candidate, blobs, interior
