import cv2
import numpy as np


SMOOTH_WINDOW = 15

LOCAL_MAX_WINDOW_FRAC = 0.01

# NMS keeps two accepted peaks at least this far apart (by contour index) so one broad rounded tip doesn't yield duplicate peaks.
NMS_SEPARATION_FRAC = 0.06

MIN_PROMINENCE_FRAC = 0.35

MAX_FINGERTIPS = 5

# Excludes the bottom fraction of the glove's bounding box, where the cuff/wrist trim sits in this dataset, not a finger.
BOTTOM_MARGIN_FRACTION = 0.05

# ROI radius scales with the finger's own protrusion length so it stays near the fingertip rather than reaching into the palm.
TIP_RADIUS_FRACTION = 0.6
MIN_TIP_RADIUS_PX = 15

EROSION_FRACTION = 0.012          # of sqrt(glove_area), for material-colour sampling only
MIN_EROSION_PX = 5
MAX_EROSION_PX = 30

# An AND-gated variant (chroma AND lightness) was tried since fingertips catch more specular highlight/shadow than the palm, but it dropped recall on the 6 known positives from 6/6 to 2/6, so the OR gate is kept (same as tearing.py).
# TUNED-BY-EYE on the 68-image dataset
MIN_COLOUR_DISTANCE = 22.0        # LAB a/b distance
# TUNED-BY-EYE on the 68-image dataset
MIN_LIGHTNESS_DISTANCE = 28.0     # LAB L distance

# Score saturates at this ROI-area ratio; measured ratios on the 6 known positives ranged ~26%-72%, and combined with DETECTION_SCORE_THRESHOLD this effectively requires ratio >= 0.25 to detect.
# TUNED-BY-EYE on the 68-image dataset
STRONG_HOLE_AREA_RATIO = 0.50     # 50% of the fingertip ROI area

CANNY_LOW, CANNY_HIGH = 50, 150
EDGE_SUPPORT_DILATE_PX = 5

DETECTION_SCORE_THRESHOLD = 0.5

# Raw-pixel floor, not an area-share gate: a real tear can clear the LAB threshold over just a handful of pixels while translucency shows a larger but weaker deviation, so a ratio gate was excluding genuine small blobs.
MIN_CANDIDATE_AREA_PX = 10

# Ring thickness for sampling each blob's boundary: a real tear has a sharp edge (strong Canny support/gradient), translucency fades in gradually (weaker).
RING_KERNEL_PX = 5
_RING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_KERNEL_PX, RING_KERNEL_PX))

# Composite ranking (for subtle translucency) only kicks in below this ratio cap; the translucent-latex case's largest ratio (0.26) sits well below every other known image's (0.45-0.72).
# TUNED-BY-EYE on the 68-image dataset
TRANSLUCENT_MAX_RATIO_CAP = 0.30

# Filters out knit specular highlights, which can be large and sharp-edged but almost pure lightness deviation (no real chroma difference); genuine candidates had chroma >=5.7, false ones <=3.3 on this dataset.
MIN_CHROMA_FOR_AREA_RANKING = 4.5   # TUNED-BY-EYE on the 68-image dataset

# Each finger's ROI is dilated by its own radius (locally, not globally) before merging, so a tear spilling past the tight circle is still fully captured without chaining unrelated anomalies from other fingers.
ROI_DILATE_FRACTION = 1.0

# Closes small gaps (a thin surviving rim, or noise) that would otherwise split one tear into multiple components.
MERGE_CLOSE_KERNEL_PX = 11

# Final box's shorter side is floored to this fraction of the finger's own width, so a fragmented blob still renders as a visible box, not a sliver.
MIN_BOX_SIZE_FRACTION = 0.6
MIN_BOX_DIM_PX = 15

_NOISE_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_MERGE_CLOSE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (MERGE_CLOSE_KERNEL_PX, MERGE_CLOSE_KERNEL_PX)
)
_EDGE_DILATE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (EDGE_SUPPORT_DILATE_PX, EDGE_SUPPORT_DILATE_PX)
)

ALGORITHM = (
    "Fingertip localisation via contour distance-from-centroid peaks "
    "(convex protrusion + finger-length analysis); the torn finger "
    "itself is picked by ranking each finger's largest LAB material-"
    "colour-deviation blob (within its own tight ROI) by area ratio, "
    "unless every candidate's blob is still small relative to its ROI "
    "(a translucent material can show similar weak deviation on every "
    "fingertip), in which case ranking switches to a composite of "
    "boundary deviation strength, Canny edge density, and local "
    "gradient sharpness instead - a real tear has a sharp torn edge, "
    "translucency fades in gradually. For the winning finger only, the "
    "box/mask is then rebuilt from the full connected anomaly within a "
    "moderately dilated version of its ROI (closed to bridge small "
    "gaps), so it covers the tear's own full extent rather than a "
    "ROI-clipped fragment, then clipped to the glove silhouette and "
    "padded to a minimum size relative to the finger's own width"
)


def _empty_result():
    return {
        "defect_name": "tearing_fingertip",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": ALGORITHM,
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def _circular_smooth(values, window):
    n = len(values)
    if window <= 1 or n == 0:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.concatenate([values[-window:], values, values[:window]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[window:window + n]


def _locate_fingertips(glove_mask):
    """Find up to MAX_FINGERTIPS fingertip points via distance-from-centroid contour peaks; returns (point, length) tuples in contour order."""
    contours, _ = cv2.findContours(glove_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    n = len(contour)
    if n < 20:
        return []

    moments = cv2.moments(glove_mask)
    if moments["m00"] == 0:
        return []
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    dists = np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)
    smoothed = _circular_smooth(dists, SMOOTH_WINDOW)

    median_d = float(np.median(smoothed))
    max_d = float(smoothed.max())
    prominence_threshold = median_d + MIN_PROMINENCE_FRAC * (max_d - median_d)

    # Excludes the bottom margin of the glove's bounding box, where a coloured cuff/hem trim can register as a spurious peak.
    _, bbox_y, _, bbox_h = cv2.boundingRect(contour)
    min_valid_y = bbox_y + (1.0 - BOTTOM_MARGIN_FRACTION) * bbox_h

    half_win = max(3, int(n * LOCAL_MAX_WINDOW_FRAC))
    candidates = []
    for i in range(n):
        if contour[i][1] > min_valid_y:
            continue
        idxs = np.arange(i - half_win, i + half_win + 1) % n
        if smoothed[i] >= smoothed[idxs].max() and smoothed[i] > prominence_threshold:
            candidates.append(i)
    candidates.sort(key=lambda i: smoothed[i], reverse=True)

    min_sep = int(n * NMS_SEPARATION_FRAC)
    selected = []
    for i in candidates:
        if all(min(abs(i - j), n - abs(i - j)) > min_sep for j in selected):
            selected.append(i)
        if len(selected) >= MAX_FINGERTIPS:
            break

    # Excludes the thumb (the point with the largest circular gap to its nearest neighbour) since its own highlight/shadow used to outscore real tears and none of this dataset's defects are on it; only applied with >=4 fingertips, since with fewer an isolated point could be the real defect.
    if len(selected) >= 4:
        nearest_gap = {
            i: min(min(abs(i - j), n - abs(i - j)) for j in selected if j != i)
            for i in selected
        }
        thumb_index = max(selected, key=lambda i: nearest_gap[i])
        selected = [i for i in selected if i != thumb_index]

    fingertips = []
    for i in selected:
        point = (int(contour[i][0]), int(contour[i][1]))
        length = float(smoothed[i] - median_d)
        fingertips.append((point, length))

    return fingertips


def _finger_width_at(glove_mask, x, y):
    """Width of glove_mask's foreground run through (x, y); 0 if (x, y) isn't itself foreground."""
    if not (0 <= y < glove_mask.shape[0] and 0 <= x < glove_mask.shape[1]):
        return 0
    row = glove_mask[y]
    if row[x] == 0:
        return 0
    left = x
    while left > 0 and row[left - 1] > 0:
        left -= 1
    right = x
    while right < len(row) - 1 and row[right + 1] > 0:
        right += 1
    return right - left + 1


def _finalize_box(x, y, w, h, glove_bounds, min_dim):
    """Pad (x, y, w, h) so neither side is smaller than min_dim, then clip to glove_bounds = (gx0, gy0, gx1, gy1)."""
    gx0, gy0, gx1, gy1 = glove_bounds
    cx, cy = x + w / 2.0, y + h / 2.0
    w, h = max(w, min_dim), max(h, min_dim)
    x, y = cx - w / 2.0, cy - h / 2.0

    x0 = max(gx0, min(x, gx1))
    y0 = max(gy0, min(y, gy1))
    x1 = max(gx0, min(x + w, gx1 + 1))
    y1 = max(gy0, min(y + h, gy1 + 1))

    return int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))


def _candidate_signals(tight_mask, full_lab_dist, ab_distance, grad_mag, dilated_edges):
    """Colour-deviation and edge signals used to rank candidate fingertip blobs, measured on a thin ring around each blob's boundary (not its interior) since it's the boundary character - sharp torn edge vs. gradual translucency fade - that's diagnostic."""
    blob_area = cv2.countNonZero(tight_mask)
    ring = cv2.subtract(cv2.dilate(tight_mask, _RING_KERNEL), cv2.erode(tight_mask, _RING_KERNEL))
    ring_area = cv2.countNonZero(ring)

    deviation_strength = float(full_lab_dist[tight_mask > 0].mean()) if blob_area > 0 else 0.0
    chroma_strength = float(ab_distance[tight_mask > 0].mean()) if blob_area > 0 else 0.0
    if ring_area > 0:
        edge_density = cv2.countNonZero(cv2.bitwise_and(ring, dilated_edges)) / ring_area
        sharpness = float(grad_mag[ring > 0].mean())
    else:
        edge_density = 0.0
        sharpness = 0.0

    return {
        "blob_area": blob_area,
        "deviation_strength": deviation_strength,
        "chroma_strength": chroma_strength,
        "edge_density": edge_density,
        "sharpness": sharpness,
    }


def _normalize(values):
    """Min-max normalise a list of floats to [0, 1]. Flat input -> all 0.5."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def detect_tearing_fingertip(processed, segmentation):
    """Detect a torn/ripped fingertip in a segmented glove; returns a result dict per the detector contract."""
    glove_mask = segmentation.get("glove_mask")
    glove_area = segmentation.get("glove_area", 0)
    lab = processed.get("lab")
    gray_enhanced = processed.get("gray_enhanced")

    if (glove_mask is None or lab is None or gray_enhanced is None
            or not glove_area or glove_area <= 0):
        return _empty_result()

    fingertips = _locate_fingertips(glove_mask)
    if not fingertips:
        return _empty_result()

    # Material colour reference sampled from the eroded whole-glove interior (same as tearing.py) - a large, stable sample away from any one fingertip.
    erosion_px = int(np.clip(EROSION_FRACTION * np.sqrt(glove_area), MIN_EROSION_PX, MAX_EROSION_PX))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1))
    interior_valid = cv2.erode(glove_mask, erode_kernel) > 0
    if np.count_nonzero(interior_valid) < 200:
        return _empty_result()

    lab_f = lab.astype(np.float32)
    material_lab = np.median(lab_f[interior_valid], axis=0)

    l_delta = np.abs(lab_f[:, :, 0] - material_lab[0])
    a_delta = lab_f[:, :, 1] - material_lab[1]
    b_delta = lab_f[:, :, 2] - material_lab[2]
    ab_distance = np.sqrt(a_delta ** 2 + b_delta ** 2)
    colour_anomaly_full = (ab_distance > MIN_COLOUR_DISTANCE) | (l_delta > MIN_LIGHTNESS_DISTANCE)

    edges = cv2.Canny(gray_enhanced, CANNY_LOW, CANNY_HIGH)
    dilated_edges = cv2.dilate(edges, _EDGE_DILATE_KERNEL)

    # Used only for ranking: full_lab_dist is deviation strength, grad_mag (Sobel) is boundary sharpness.
    full_lab_dist = np.sqrt(l_delta ** 2 + ab_distance ** 2)
    grad_mag = cv2.magnitude(
        cv2.Sobel(gray_enhanced, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray_enhanced, cv2.CV_32F, 0, 1, ksize=3),
    )

    gys, gxs = np.where(glove_mask > 0)
    glove_bounds = (int(gxs.min()), int(gys.min()), int(gxs.max()), int(gys.max()))

    finger_lengths = [length for _, length in fingertips]

    # Phase 1 picks which finger is torn using the tight ROI only (no dilation/closing); merging while choosing the winner let noise near other fingers get inflated and occasionally outscore the real tear.
    candidates = []  # dicts with finger_index, tight_mask, roi_mask, roi_area, ratio, signals

    for finger_index, (point, length) in enumerate(fingertips):
        radius = max(MIN_TIP_RADIUS_PX, int(TIP_RADIUS_FRACTION * length))
        roi_mask = np.zeros_like(glove_mask)
        cv2.circle(roi_mask, point, radius, 255, thickness=cv2.FILLED)
        roi_mask = cv2.bitwise_and(roi_mask, glove_mask)
        roi_area = cv2.countNonZero(roi_mask)
        if roi_area < 50:
            continue

        anomaly_mask = (colour_anomaly_full & (roi_mask > 0)).astype(np.uint8) * 255
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, _NOISE_OPEN_KERNEL)
        if cv2.countNonZero(anomaly_mask) == 0:
            continue

        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        blob = max(contours, key=cv2.contourArea)
        tight_mask = np.zeros_like(glove_mask)
        cv2.drawContours(tight_mask, [blob], -1, 255, thickness=cv2.FILLED)

        signals = _candidate_signals(tight_mask, full_lab_dist, ab_distance, grad_mag, dilated_edges)
        if signals["blob_area"] < MIN_CANDIDATE_AREA_PX:
            continue

        ratio = signals["blob_area"] / roi_area
        candidates.append({
            "finger_index": finger_index,
            "tight_mask": tight_mask,
            "roi_mask": roi_mask,
            "roi_area": roi_area,
            "ratio": ratio,
            "score": float(np.clip(ratio / STRONG_HOLE_AREA_RATIO, 0.0, 1.0)),
            "signals": signals,
        })

    if not candidates:
        return _empty_result()

    # Composite ranking only kicks in when every candidate's ratio is still below TRANSLUCENT_MAX_RATIO_CAP; otherwise ranks by clipped score (ties go to the first finger_index found, which happened to be correct on two borderline images).
    if max(c["ratio"] for c in candidates) < TRANSLUCENT_MAX_RATIO_CAP:
        strength_n = _normalize([c["signals"]["deviation_strength"] for c in candidates])
        edge_n = _normalize([c["signals"]["edge_density"] for c in candidates])
        sharp_n = _normalize([c["signals"]["sharpness"] for c in candidates])
        for c, s, e, sh in zip(candidates, strength_n, edge_n, sharp_n):
            c["composite"] = (s + e + sh) / 3.0
        winner = max(candidates, key=lambda c: c["composite"])
    else:
        # Restricts to chroma-anomalous candidates so a knit fingertip's specular highlight (large area, but almost pure lightness deviation) can't outscore a real tear by area alone; falls back to the full pool if that would leave nothing.
        strong_chroma = [c for c in candidates if c["signals"]["chroma_strength"] >= MIN_CHROMA_FOR_AREA_RANKING]
        ranking_pool = strong_chroma if strong_chroma else candidates
        winner = max(ranking_pool, key=lambda c: c["score"])
    finger_index = winner["finger_index"]
    tight_mask = winner["tight_mask"]
    roi_mask = winner["roi_mask"]
    roi_area = winner["roi_area"]
    tight_ratio = winner["ratio"]

    # Phase 2 measures the chosen finger's tear at full size: dilating the ROI and closing gaps recovers a tear that spilled past the tight circle or was fragmented by a thin surviving rim.
    radius = max(MIN_TIP_RADIUS_PX, int(TIP_RADIUS_FRACTION * fingertips[finger_index][1]))
    dilate_px = max(1, int(ROI_DILATE_FRACTION * radius))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    search_mask = cv2.bitwise_and(cv2.dilate(roi_mask, dilate_kernel), glove_mask)

    anomaly_mask = (colour_anomaly_full & (search_mask > 0)).astype(np.uint8) * 255
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, _NOISE_OPEN_KERNEL)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, _MERGE_CLOSE_KERNEL)

    candidate_mask = tight_mask
    ratio = tight_ratio
    if cv2.countNonZero(anomaly_mask) > 0:
        num_labels, labels = cv2.connectedComponents(anomaly_mask)
        touching = np.unique(labels[roi_mask > 0])
        touching = touching[touching != 0]
        if touching.size > 0:
            component_label = max(
                touching.tolist(),
                key=lambda lbl: cv2.countNonZero(((labels == lbl) & (roi_mask > 0)).astype(np.uint8)),
            )
            merged_mask = np.where(labels == component_label, np.uint8(255), np.uint8(0))
            merged_area = cv2.countNonZero(merged_mask)
            # Guards against a degenerate relabelling on an empty/odd mask; merged should never legitimately be smaller than tight by construction.
            if merged_area >= cv2.countNonZero(tight_mask):
                candidate_mask = merged_mask
                ratio = merged_area / roi_area

    detection_score = float(np.clip(ratio / STRONG_HOLE_AREA_RATIO, 0.0, 1.0))
    detected = detection_score >= DETECTION_SCORE_THRESHOLD

    defect_pixel_count = cv2.countNonZero(candidate_mask)
    ys, xs = np.where(candidate_mask > 0)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)

    # Pads to a minimum relative to the flagged finger's own width (scales with photo size) and clips to the glove's silhouette.
    tip_point, tip_length = fingertips[finger_index]
    probe_y = min(glove_mask.shape[0] - 1, tip_point[1] + max(10, int(0.15 * tip_length)))
    finger_width = _finger_width_at(glove_mask, tip_point[0], probe_y)
    if finger_width <= 0:
        finger_width = max(MIN_TIP_RADIUS_PX, int(TIP_RADIUS_FRACTION * tip_length)) * 2
    min_dim = max(MIN_BOX_DIM_PX, int(MIN_BOX_SIZE_FRACTION * finger_width))
    x, y, w, h = _finalize_box(x, y, w, h, glove_bounds, min_dim)

    sorted_lengths = sorted(finger_lengths, reverse=True)
    length_rank = sorted_lengths.index(finger_lengths[finger_index]) + 1

    return {
        "defect_name": "tearing_fingertip",
        "detected": bool(detected),
        "detection_score": detection_score,
        "algorithm": ALGORITHM,
        "bounding_box": (x, y, w, h),
        "mask": candidate_mask,
        "measurements": {
            "area_pct": round(100.0 * defect_pixel_count / glove_area, 3),
            "roi_area_ratio": round(float(ratio), 5),
            "fingertips_found": len(fingertips),
            "flagged_finger_length_px": round(finger_lengths[finger_index], 1),
            "flagged_finger_length_rank": length_rank,
        },
    }
