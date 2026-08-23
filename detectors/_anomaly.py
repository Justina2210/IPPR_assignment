import cv2
import numpy as np


# Shadow/colour-ramp at the silhouette boundary was the single largest source of false positives; margin scales with glove radius, not image size.
def glove_interior(glove_mask, margin_ratio=0.09, min_margin=10, max_margin=40):
    """Remove a scale-aware margin from the glove silhouette."""
    if glove_mask is None:
        return None

    binary = (glove_mask > 0).astype(np.uint8)
    area = int(binary.sum())
    if area <= 0:
        return np.zeros_like(binary, dtype=np.uint8)

    equivalent_radius = np.sqrt(area / np.pi)
    margin = int(np.clip(margin_ratio * equivalent_radius, min_margin, max_margin))

    # Distance thresholding for scale-independent erosion.
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    interior = (distance > margin).astype(np.uint8) * 255

    # Fall back to lighter erosion when the normal margin removes too much.
    if int(np.count_nonzero(interior)) < 0.25 * area:
        interior = (distance > max(3, margin // 3)).astype(np.uint8) * 255

    return interior


# Median filter, not morph closing/Gaussian blur (both tried, both let one highlight or one large defect bias their own neighbourhood's estimate); downsampled for speed.
def robust_background(channel, mask, scale=8, ksize=11, passes=2):
    """Estimate a smooth, defect-free version of one LAB channel via iterative downsample/median-blur passes."""
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


# Splitting into lightness + chroma keeps the cues independent - catches both a pale rust speck (chroma) and a black streak on blue nitrile (lightness).
def anomaly_fields(processed, interior_mask):
    """Build lightness/chroma/edge maps for anomaly detection; dark_rel normalises by local background to remove exposure dependence."""
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


def background_reference(processed, glove_mask):
    """Median LAB colour of whatever segmentation decided was NOT glove, used to detect segmentation leakage (a background patch wrongly included otherwise reads as a confident defect)."""
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


# Conjunction of absolute+relative darkening thresholds (not Otsu, which always finds a split even on a clean glove) keeps the test stable across materials; opening removes 1-2px wrinkle lines.
def candidate_mask(fields, interior_mask,
                   dark_abs_min=12.0, dark_rel_min=0.10, chroma_min=9.0,
                   open_radius=2, close_radius=2):
    """Threshold anomaly maps (dark AND chroma) into a candidate mask, cleaned morphologically."""
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


# edge_contact does most of the FP suppression: real contamination is surrounded by glove material, while segmentation-leaked skin/shading runs along the boundary.
def describe_blobs(candidate, fields, interior_mask, min_area=20, background_lab=None):
    """Describe connected candidate regions using shape and contrast features."""
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

        # Distance to background colour; small means likely segmentation-leaked background, not contamination.
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
    """Drop regions not surrounded by glove material; the edge-contact limit is relaxed for small regions so a fingertip smudge touching the boundary isn't discarded like a large leaked skin/shading band."""
    kept = []
    for blob in blobs:
        # Regions whose colour matches the photographic backdrop are segmentation leakage, not contamination.
        if blob.get("bg_distance", 1e6) < min_bg_distance:
            continue
        limit = (compact_edge_contact
                 if blob["area_frac"] < compact_area_frac
                 else max_edge_contact)
        if blob["edge_contact"] < limit:
            kept.append(blob)
    return kept


def cloud_statistics(blobs, defect_mask, fields, interior_mask):
    """Summarise accepted regions into aggregate shape/colour statistics used by the detectors."""
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


# Cast shadows shift lightness but leave hue (a/b) near zero; foreign material shifts a/b too. Matches candidate_mask's chroma_min.
CHROMA_EVIDENCE = 8.0

# Ramp from measured distributions: shadows score ~0-0.10 (bunched cotton 0.000, creased latex 0.065), true positives 0.09-0.96 (weakest 0.089) - the two overlap around 0.09-0.18.
CHROMATIC_RAMP = (0.02, 0.20)

# Soft multiplier, not a hard veto (a veto tight enough for the shadow overlap would also cut real stains); floor stays high enough that achromatic contamination (grey dust) still scores on shape alone.
SHADOW_FLOOR = 0.65


def chromatic_confidence(stats):
    """Soft confidence multiplier in [SHADOW_FLOOR, 1.0] based on chroma evidence."""
    confidence = ramp(stats["chromatic_fraction"], *CHROMATIC_RAMP)
    return SHADOW_FLOOR + (1.0 - SHADOW_FLOOR) * confidence


def ramp(value, low, high):
    """Map a measurement onto 0..1 via a linear ramp between low and high, clipped."""
    if high == low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def combine(weighted_terms):
    """Combine weighted (weight, value) evidence terms into a single clipped 0..1 score."""
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


def localisation_blobs(blobs, cuff_fraction=0.88):
    """Regions eligible to carry the reported box - excludes the cuff band (own detector's subject, dominates a soiled-cuff image) from box selection only, never from scoring."""
    if not blobs:
        return blobs
    top = min(b["bbox"][1] for b in blobs)
    bottom = max(b["bbox"][1] + b["bbox"][3] for b in blobs)
    limit = top + cuff_fraction * (bottom - top)
    eligible = [b for b in blobs if b["centroid"][1] < limit]
    return eligible if eligible else blobs


def dominant_bbox(blobs, equivalent_radius, gap_ratio=0.10):
    """Box around the strongest concentration of defect evidence (grouped by proximity, ranked by area x contrast^2), not a union box over everything scattered."""
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
        # Area x contrast^2: pure area follows shading/cuffs, pure contrast follows the darkest region (often the cuff on knit).
        return sum(b["area"] * (b["mean_dark"] ** 2) for b in group)

    return union_bbox(max(groups, key=evidence))


def background_guarded_interior(processed, glove_mask, tolerance=25.0):
    """Interior mask excluding photographic-background-coloured pixels first, so segmentation over-reach doesn't threshold into one large false region."""
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
    """Run the shared anomaly pipeline: background-guarded interior mask -> fields -> candidate mask -> blobs."""
    interior = background_guarded_interior(processed, segmentation["glove_mask"])
    if interior is None or int(np.count_nonzero(interior)) < 500:
        empty = np.zeros(processed["gray"].shape[:2], dtype=np.uint8)
        return None, empty, [], empty

    fields = anomaly_fields(processed, interior)
    candidate = candidate_mask(fields, interior, **candidate_kwargs)
    background_lab = background_reference(processed, segmentation["glove_mask"])
    blobs = describe_blobs(candidate, fields, interior, background_lab=background_lab)
    return fields, candidate, blobs, interior
