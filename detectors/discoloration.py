"""
discoloration.py
-----------------
Detector for the "discoloration" defect in the Glove Defect Detection System.

Discoloration = a patch of the glove whose colour deviates noticeably from
the glove's own dominant colour. Unlike a small isolated spot (spotting.py)
or a hard dark blob (dirty.py / stain.py), discoloration is typically a
softer, larger, patchy shift in colour caused by fading, chemical exposure,
or uneven material colouring.

Algorithm (documented per guideline Section 2): "LAB colour deviation +
connected components"
----------------------------------------------------------------------
1. Work only inside segmentation["glove_mask"] (background is masked out
   with cv2.bitwise_and-equivalent boolean indexing, so background colour
   never contributes).
2. Estimate the glove's own expected colour as the MEDIAN LAB value over
   all glove pixels. The median is used (rather than the mean) so that a
   discoloration patch covering a minority of the glove does not pull the
   reference colour toward itself.
   Assumption / limitation: this only holds if the discoloration affects
   less than half the visible glove surface. If a discoloration covers
   most of the glove, the median itself may be biased and the detector
   may under-report -- documented here per the guideline's requirement to
   flag tuning assumptions.
3. Compute per-pixel LAB colour distance (Euclidean delta-E) from that
   reference colour.
4. Threshold the distance map using Otsu's method (same adaptive approach
   used in segmentation.py) with a floor (MIN_DELTA_E) so that ordinary
   LAB noise on a uniformly-coloured glove is never flagged.
5. Clean the candidate mask (morphological open/close) and drop connected
   components smaller than MIN_REGION_AREA_PX (camera noise / compression
   artifacts, not real discoloration).
6. Score the result using three sub-scores, combined with the same
   0.5 / 0.3 / 0.2 (colour / area / region) weighting scheme used across
   all three of Justina's detectors, per the architecture doc's worked
   example of how detection_score should be composed:
     - colour_score  : how strongly the flagged pixels deviate in colour
     - area_score    : how much of the glove surface is affected
     - region_score  : how consolidated (vs. speckled/noisy) the patches are

Thresholds (MIN_DELTA_E, MIN_REGION_AREA_PX, REFERENCE_DELTA_E,
REFERENCE_AREA_FRACTION) were set by inspection since no annotated
discoloration dataset was available at development time. This is a
limitation to state explicitly in the report's Critical Analysis section,
and these values should be revisited once evaluate.py has been run against
the full labelled dataset.
"""

import cv2
import numpy as np


# ============================================================
# CONFIGURATION / THRESHOLDS
# (document per Detector_Development_Guideline Section 2)
# ============================================================

# Minimum LAB colour distance (delta-E) from the glove's own median colour
# for a pixel to be a discoloration *candidate*, before Otsu refines it.
# Floor chosen so normal fabric/rubber shading variation isn't flagged.
MIN_DELTA_E = 8.0

# Any connected discoloration region smaller than this (in px, on the
# 1000px-long-side resized image) is treated as noise, not a real patch.
MIN_REGION_AREA_PX = 40

# Mean delta-E that should map to a "clearly discoloured" colour_score of
# ~1.0. Tuned by eye -- documented limitation, see module docstring.
REFERENCE_DELTA_E = 35.0

# Fraction of glove_area treated as "large, obvious discoloration" for
# normalising area_score to ~1.0.
REFERENCE_AREA_FRACTION = 0.15

DETECTION_SCORE_WEIGHTS = {"colour": 0.5, "area": 0.3, "region": 0.2}

# Final cutoff detect_discoloration uses for its own "detected" flag.
# evaluate.py re-applies its own DETECTION_THRESHOLD (0.5) on top of this,
# so this is only this detector's own opinion, kept in step with that
# shared threshold for consistency.
LOCAL_DETECTION_THRESHOLD = 0.5


# ============================================================
# HELPERS
# ============================================================

def _otsu_mask(distance_map, floor):
    """Adaptive Otsu threshold on a non-negative distance map, with a floor."""
    max_val = float(distance_map.max())
    if max_val <= 1e-6:
        return np.zeros(distance_map.shape, dtype=bool)
    scaled = np.clip(distance_map / max_val * 255.0, 0, 255).astype(np.uint8)
    otsu_level, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(floor, (otsu_level / 255.0) * max_val)
    return distance_map > threshold


def _clean_candidate_mask(mask_bool):
    """Morphological clean-up + minimum-area connected-component filtering."""
    mask_u8 = mask_bool.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    output = np.zeros_like(cleaned)
    kept_regions = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= MIN_REGION_AREA_PX:
            output[labels == label] = 255
            kept_regions += 1
    return output, kept_regions


def _bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
    return (x, y, w, h)


# ============================================================
# MAIN DETECTOR (contract required by evaluate.py)
# ============================================================

def detect_discoloration(processed: dict, segmentation: dict) -> dict:
    """
    Detect discoloration on the glove surface.

    Parameters
    ----------
    processed : dict
        Output of preprocess_image() -- uses the 'lab' representation.
    segmentation : dict
        Output of segment_glove() -- uses 'glove_mask' and 'glove_area'.

    Returns
    -------
    dict
        Standard detector-contract result: defect_name, detected,
        detection_score, algorithm, bounding_box, mask, measurements.
    """
    glove_mask = segmentation["glove_mask"]
    glove_area = segmentation["glove_area"]

    result = {
        "defect_name": "discoloration",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": "LAB colour deviation + connected components",
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }

    if glove_area <= 0 or glove_mask is None:
        return result

    lab = processed["lab"].astype(np.float32)
    mask_bool = glove_mask > 0

    glove_lab_pixels = lab[mask_bool]
    if glove_lab_pixels.shape[0] == 0:
        return result

    reference_lab = np.median(glove_lab_pixels, axis=0)

    l_delta = lab[:, :, 0] - reference_lab[0]
    a_delta = lab[:, :, 1] - reference_lab[1]
    b_delta = lab[:, :, 2] - reference_lab[2]
    delta_e = np.sqrt(l_delta ** 2 + a_delta ** 2 + b_delta ** 2)

    delta_e_glove = np.where(mask_bool, delta_e, 0.0)

    candidate_bool = _otsu_mask(delta_e_glove, MIN_DELTA_E) & mask_bool
    defect_mask, num_regions = _clean_candidate_mask(candidate_bool)

    defect_area = int(np.count_nonzero(defect_mask))

    if defect_area == 0:
        result["measurements"] = {
            "area_pct": 0.0,
            "mean_delta_e": 0.0,
            "num_regions": 0,
        }
        return result

    area_pct = 100.0 * defect_area / glove_area
    mean_delta_e = float(delta_e[defect_mask > 0].mean())

    colour_score = min(1.0, mean_delta_e / REFERENCE_DELTA_E)
    area_score = min(1.0, (defect_area / glove_area) / REFERENCE_AREA_FRACTION)
    # Fewer, larger regions -> higher region_score; many tiny specks
    # (more likely noise) pull the score down.
    region_score = 1.0 / (1.0 + 0.15 * max(0, num_regions - 1))

    detection_score = (
        DETECTION_SCORE_WEIGHTS["colour"] * colour_score
        + DETECTION_SCORE_WEIGHTS["area"] * area_score
        + DETECTION_SCORE_WEIGHTS["region"] * region_score
    )
    detection_score = float(min(1.0, max(0.0, detection_score)))

    result["detected"] = detection_score >= LOCAL_DETECTION_THRESHOLD
    result["detection_score"] = detection_score
    result["bounding_box"] = _bbox_from_mask(defect_mask)
    result["mask"] = defect_mask
    result["measurements"] = {
        "area_pct": round(area_pct, 2),
        "mean_delta_e": round(mean_delta_e, 2),
        "num_regions": num_regions,
        "colour_score": round(colour_score, 3),
        "area_score": round(area_score, 3),
        "region_score": round(region_score, 3),
    }

    return result


# ============================================================
# OPTIONAL QUICK TEST
# ============================================================

if __name__ == "__main__":
    import os
    from preprocessing import load_image, preprocess_image
    from segmentation import segment_glove

    candidate_folders = [
        os.path.join("dataset", "nitrile", "discoloration"),
        os.path.join("dataset", "latex", "discoloration"),
        os.path.join("dataset", "cotton", "discoloration"),
    ]
    sample = None
    for folder in candidate_folders:
        if os.path.isdir(folder):
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    sample = os.path.join(folder, fname)
                    break
        if sample:
            break

    if sample is None:
        print("No sample discoloration image found under dataset/.")
    else:
        processed = preprocess_image(load_image(sample))
        seg = segment_glove(processed)
        out = detect_discoloration(processed, seg)
        print(f"Image: {sample}")
        print(f"Detected: {out['detected']}  Score: {out['detection_score']:.3f}")
        print(f"Measurements: {out['measurements']}")