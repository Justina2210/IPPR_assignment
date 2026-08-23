import cv2
import numpy as np

# Erode inward first: excludes the anti-aliased edge/cuff ring, which looks like a colour deviation unrelated to tearing.
EROSION_FRACTION = 0.012          # of sqrt(glove_area)
MIN_EROSION_PX = 5
MAX_EROSION_PX = 30

# Fixed (not Otsu) cutoff: a smooth glove-surface lighting gradient makes Otsu over-flag ordinary shading.
# TUNED-BY-EYE on the 68-image dataset
MIN_COLOUR_DISTANCE = 22.0        # LAB a/b distance
# TUNED-BY-EYE on the 68-image dataset
MIN_LIGHTNESS_DISTANCE = 28.0     # LAB L distance

# Below this, the largest anomaly blob is treated as noise (wrinkle highlight, lint), not a tear.
MIN_HOLE_AREA_RATIO = 0.02        # 2% of glove_area

# Score saturates to 1.0 here; the 5 known tearing images measured ~6.4%-10.7%.
# TUNED-BY-EYE on the 68-image dataset
STRONG_HOLE_AREA_RATIO = 0.09     # 9% of glove_area

# Above this, reject outright - more likely a segmentation/lighting artifact than a real tear.
MAX_HOLE_AREA_RATIO = 0.25         # 25% of glove_area

# Must contain this many (dilated) Canny edge pixels to count as a real torn edge, not a soft gradient.
MIN_EDGE_SUPPORT_PX = 6
CANNY_LOW, CANNY_HIGH = 50, 150
EDGE_SUPPORT_DILATE_PX = 5

# Overall detected/not-detected cutoff on the 0-1 score.
DETECTION_SCORE_THRESHOLD = 0.5

_NOISE_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_EDGE_DILATE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (EDGE_SUPPORT_DILATE_PX, EDGE_SUPPORT_DILATE_PX)
)

ALGORITHM = (
    "Largest connected LAB material-colour-deviation blob inside the "
    "eroded glove interior, cross-checked against Canny edges from "
    "gray_enhanced"
)


def _empty_result():
    return {
        "defect_name": "tearing",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": ALGORITHM,
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def detect_tearing(processed, segmentation):
    """Detect tearing (rips/holes) in a segmented glove; returns a result dict per the detector contract."""
    glove_mask = segmentation.get("glove_mask")
    glove_area = segmentation.get("glove_area", 0)
    lab = processed.get("lab")
    gray_enhanced = processed.get("gray_enhanced")

    if (glove_mask is None or lab is None or gray_enhanced is None
            or not glove_area or glove_area <= 0):
        return _empty_result()

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

    colour_anomaly = ab_distance > MIN_COLOUR_DISTANCE
    lightness_anomaly = l_delta > MIN_LIGHTNESS_DISTANCE
    anomaly_mask = ((colour_anomaly | lightness_anomaly) & interior_valid).astype(np.uint8) * 255
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, _NOISE_OPEN_KERNEL)

    if cv2.countNonZero(anomaly_mask) == 0:
        return _empty_result()

    # Largest blob only: a real tear is ~2x the runner-up, wrinkle shading breaks into several smaller blobs.
    contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty_result()

    best_contour = max(contours, key=cv2.contourArea)
    best_area = cv2.contourArea(best_contour)
    best_ratio = best_area / glove_area

    if best_ratio < MIN_HOLE_AREA_RATIO or best_ratio > MAX_HOLE_AREA_RATIO:
        return _empty_result()

    candidate_mask = np.zeros_like(glove_mask)
    cv2.drawContours(candidate_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    edges = cv2.Canny(gray_enhanced, CANNY_LOW, CANNY_HIGH)
    dilated_edges = cv2.dilate(edges, _EDGE_DILATE_KERNEL)
    if cv2.countNonZero(cv2.bitwise_and(candidate_mask, dilated_edges)) < MIN_EDGE_SUPPORT_PX:
        return _empty_result()  # no sharp torn edge nearby -> likely a soft shading/glare artifact

    detection_score = float(np.clip(best_ratio / STRONG_HOLE_AREA_RATIO, 0.0, 1.0))
    detected = detection_score >= DETECTION_SCORE_THRESHOLD

    defect_pixel_count = cv2.countNonZero(candidate_mask)
    ys, xs = np.where(candidate_mask > 0)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)

    return {
        "defect_name": "tearing",
        "detected": bool(detected),
        "detection_score": detection_score,
        "algorithm": ALGORITHM,
        "bounding_box": (x, y, w, h),
        "mask": candidate_mask,
        "measurements": {
            "area_pct": round(100.0 * defect_pixel_count / glove_area, 3),
            "hole_area_ratio": round(float(best_ratio), 5),
        },
    }
