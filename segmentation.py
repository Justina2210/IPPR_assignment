"""
segmentation.py
---------------
Shared segmentation module for the Glove Defect Detection System.

This module is responsible for separating the glove foreground from the
background and, when needed, removing forearm regions that were captured
in the image. The method is adaptive so it can work with different
photo backgrounds such as turquoise, green, or slightly brighter/darker
versions with small shadows.

The main output is a binary glove mask where:
- glove pixels = 255 (white)
- background = 0 (black)
- forearm = 0 if reliably removed
"""

import cv2
import numpy as np


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _as_bgr_image(image):
    """Return a BGR image from either a processed dictionary or a NumPy array."""
    if isinstance(image, dict):
        if "original" in image:
            return image["original"]
        if "denoised" in image:
            return image["denoised"]
        raise ValueError("Dictionary image input must contain 'original' or 'denoised'.")
    return image


# ============================================================
# BACKGROUND ESTIMATION
# ============================================================

def estimate_background(image):
    """
    Estimate the dominant background colour in an adaptive way.

    The method samples the border region of the image because most
    backgrounds are uniform and appear around the glove edges.

    Parameters
    ----------
    image : numpy.ndarray or dict
        Input image or processed dictionary.

    Returns
    -------
    numpy.ndarray
        Estimated background colour in HSV format, shape (3,).
    """
    bgr = _as_bgr_image(image)
    if bgr is None or bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]

    # Sample pixels from the image border. This is usually where the
    # background is visible, and it is less affected by the glove itself.
    margin = max(10, min(height, width) // 12)
    border_mask = np.zeros((height, width), dtype=np.uint8)
    border_mask[:margin, :] = 1
    border_mask[-margin:, :] = 1
    border_mask[:, :margin] = 1
    border_mask[:, -margin:] = 1

    border_pixels = hsv[border_mask == 1]
    if border_pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Use median values across border pixels to be robust to shadows and
    # small local variations in the background.
    background_hsv = np.median(border_pixels.reshape(-1, 3), axis=0)
    return np.clip(background_hsv, 0, 255).astype(np.uint8)


# ============================================================
# FOREGROUND MASK CREATION
# ============================================================

def create_foreground_mask(image):
    """
    Create a rough foreground mask by comparing the image to the
    estimated background colour.

    The comparison is done in HSV space because background changes are
    often more consistent in the hue/saturation domain than in pure BGR.

    Parameters
    ----------
    image : numpy.ndarray or dict
        Image or processed dictionary.

    Returns
    -------
    numpy.ndarray
        Binary mask with foreground pixels as 255 and background as 0.
    """
    bgr = _as_bgr_image(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bg_hsv = estimate_background(bgr)

    # Calculate difference from estimated background.
    bg_hsv = bg_hsv.astype(np.int16)
    hsv_int = hsv.astype(np.int16)

    hue_delta = np.abs(hsv_int[:, :, 0] - bg_hsv[0])
    hue_delta = np.minimum(hue_delta, 180 - hue_delta)

    sat_delta = np.abs(hsv_int[:, :, 1] - bg_hsv[1])
    val_delta = np.abs(hsv_int[:, :, 2] - bg_hsv[2])

    # Weighted difference so hue changes contribute noticeably, but value
    # and saturation differences still matter.
    difference = (
        hue_delta * 0.7 +
        sat_delta * 0.2 +
        val_delta * 0.1
    )

    # Adaptive threshold: choose threshold relative to the image's
    # distribution of colour differences.
    median_value = float(np.median(difference))
    std_value = float(np.std(difference))
    threshold = max(25.0, median_value + 1.4 * std_value)

    foreground = (difference > threshold).astype(np.uint8) * 255

    return foreground


# ============================================================
# MASK CLEANING
# ============================================================

def clean_mask(mask):
    """
    Clean the binary mask using morphological operations.

    This removes small noise while preserving the glove silhouette.
    """
    if mask is None:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


# ============================================================
# LARGEST COMPONENT
# ============================================================

def keep_largest_component(mask):
    """
    Keep the largest connected component from the binary mask.

    This helps remove isolated background fragments and small noisy blobs.
    """
    if mask is None:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for label in range(2, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label

    filtered = np.zeros_like(mask)
    filtered[labels == largest_label] = 255
    return filtered


# ============================================================
# CUFF DETECTION / FOREARM REMOVAL
# ============================================================

def detect_cuff_line(processed, mask):
    """
    Detect a cuff line when a forearm is visible.

    This is a conservative check: if there is no reliable cuff boundary,
    the function returns False and the original mask is kept.
    """
    if processed is None or mask is None:
        return False, None

    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        return False, None

    row_counts = mask.sum(axis=1) // 255
    if row_counts.max() == 0:
        return False, None

    bottom_start = max(0, int(height * 0.35))
    bottom_rows = row_counts[bottom_start:]

    if bottom_rows.size == 0:
        return False, None

    # Look for a strong drop near the bottom of the glove. This usually
    # indicates where the forearm area begins or where the cuff boundary is.
    for index in range(len(bottom_rows) - 2, 0, -1):
        current = bottom_rows[index]
        previous = bottom_rows[index - 1]
        next_row = bottom_rows[index + 1] if index + 1 < len(bottom_rows) else 0

        if previous > 0 and current <= max(5, previous * 0.55) and next_row <= current:
            cuff_y = bottom_start + index
            return True, cuff_y

    # If no clear cuff line is found, assume no reliable forearm removal.
    return False, None


def remove_forearm(mask, cuff_y):
    """
    Remove the area below the detected cuff line.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary glove mask.
    cuff_y : int
        Y coordinate of the cuff boundary.

    Returns
    -------
    numpy.ndarray
        Mask with area below the cuff removed.
    """
    if mask is None or cuff_y is None:
        return mask

    output = mask.copy()
    height = output.shape[0]
    if cuff_y >= height:
        return output

    output[cuff_y:, :] = 0
    return output


# ============================================================
# MAIN SEGMENTATION
# ============================================================

def segment_glove(processed):
    """
    Run the full shared segmentation pipeline.

    Flow:
        Preprocessed image
            ↓
        Estimate background colour
            ↓
        Create foreground mask
            ↓
        Morphological cleaning
            ↓
        Keep largest component
            ↓
        Optional cuff/forearm removal
            ↓
        Final glove mask

    Parameters
    ----------
    processed : dict
        Output of preprocess_image() in preprocessing.py.

    Returns
    -------
    dict
        Contains final segmentation results.
    """
    if processed is None:
        raise ValueError("Processed image data is required for segmentation.")

    original = processed.get("original")
    if original is None:
        raise ValueError("Processed dictionary must include 'original'.")

    raw_mask = create_foreground_mask(processed)
    cleaned = clean_mask(raw_mask)
    main_component = keep_largest_component(cleaned)

    cuff_detected = False
    cuff_y = None

    if main_component is not None:
        cuff_detected, cuff_y = detect_cuff_line(processed, main_component)
        glove_mask = main_component if not cuff_detected else remove_forearm(main_component, cuff_y)
    else:
        glove_mask = np.zeros_like(raw_mask)

    glove_area = int(np.count_nonzero(glove_mask > 0))

    return {
        "raw_mask": raw_mask,
        "glove_mask": glove_mask,
        "glove_area": glove_area,
        "cuff_detected": cuff_detected,
        "cuff_y": cuff_y,
    }


# ============================================================
# OPTIONAL QUICK TEST
# ============================================================

def _demo():
    """Small demonstration for local testing."""
    try:
        from preprocessing import load_image, preprocess_image
    except ImportError:
        print("Unable to import preprocessing.py. Make sure it is in the same folder.")
        return

    sample_candidates = [
        "dataset/nitrile/discoloration/nitrile_discoloration_1.jpeg",
        "dataset/latex/discoloration/latex_discoloration_1.jpeg",
        "dataset/cotton/oversize/cotton_oversize_1.jpeg",
    ]

    found = None
    for path in sample_candidates:
        try:
            image = load_image(path)
            found = path
            break
        except ValueError:
            continue

    if found is None:
        print("No sample image found for segmentation demo.")
        return

    processed = preprocess_image(load_image(found))
    result = segment_glove(processed)

    print(f"Demo image: {found}")
    print(f"Glove mask area: {result['glove_area']}")
    print(f"Cuff detected: {result['cuff_detected']}")
    print(f"Cuff y: {result['cuff_y']}")


if __name__ == "__main__":
    _demo()
