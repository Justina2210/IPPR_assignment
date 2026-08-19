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

    LAB is used (rather than HSV) because it separates lightness from
    colour in a way that keeps roughly equal, perceptually meaningful
    distances between similar hues (e.g. a blue glove against a teal
    background), which HSV's circular hue channel does not guarantee.

    Parameters
    ----------
    image : numpy.ndarray or dict
        Input image or processed dictionary.

    Returns
    -------
    dict
        "lab": estimated background colour in LAB space, shape (3,).
    """
    bgr = _as_bgr_image(image)
    if bgr is None or bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]

    # Sample pixels from the image border. This is usually where the
    # background is visible, and it is less affected by the glove itself.
    margin = max(10, min(height, width) // 12)
    border_mask = np.zeros((height, width), dtype=np.uint8)
    border_mask[:margin, :] = 1
    border_mask[-margin:, :] = 1
    border_mask[:, :margin] = 1
    border_mask[:, -margin:] = 1

    border_pixels = lab[border_mask == 1]
    if border_pixels.size == 0:
        return {"lab": np.array([0.0, 0.0, 0.0], dtype=np.float32)}

    # Use median values across border pixels to be robust to shadows and
    # small local variations in the background.
    background_lab = np.median(border_pixels.reshape(-1, 3), axis=0)
    return {"lab": background_lab.astype(np.float32)}


# ============================================================
# FOREGROUND MASK CREATION
# ============================================================

def _otsu_threshold_mask(distance_map, min_threshold=0.0):
    """
    Threshold a non-negative distance map using Otsu's method.

    Otsu automatically finds the valley between two populations
    (background-like vs foreground-like distances) no matter what
    fraction of the image each one occupies. This replaces a fixed
    "median + k*std" rule, which becomes unreliable once the glove
    fills a large part of the frame: mixing foreground and background
    distances together inflates the standard deviation and can push
    a fixed-formula threshold above the *entire* foreground cluster
    (this was the root cause of near-total segmentation failures,
    e.g. a blue glove on a similarly-hued teal background).

    Parameters
    ----------
    distance_map : numpy.ndarray
        Non-negative float distance values (e.g. colour or brightness
        distance from the estimated background).
    min_threshold : float
        A floor so that near-uniform images (Otsu finds ~0) do not
        classify background noise as foreground.

    Returns
    -------
    numpy.ndarray (bool)
        True where distance_map is classified as foreground.
    """
    max_value = float(distance_map.max())
    if max_value <= 1e-6:
        return np.zeros(distance_map.shape, dtype=bool)

    scaled = np.clip(distance_map / max_value * 255.0, 0, 255).astype(np.uint8)
    otsu_level, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(min_threshold, (otsu_level / 255.0) * max_value)

    return distance_map > threshold


def create_foreground_mask(image):
    """
    Create a rough foreground mask by comparing the image to the
    estimated background colour.

    The comparison is done in LAB space because it separates lightness
    from colour, giving more consistent separation than HSV hue for
    glove/background pairs that share a similar hue family (e.g. blue
    glove on a teal mat).

    Two independent signals are combined:
    - Colour distance (a/b channels): catches most colour changes.
    - Lightness distance (L channel): catches pale/white gloves that
      barely differ in colour from a bright background but clearly
      differ in brightness.

    Each signal is thresholded with Otsu's method (see
    `_otsu_threshold_mask`) rather than a fixed formula, so the split
    stays reliable whether the glove occupies 10% or 60% of the frame.

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
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = estimate_background(bgr)
    bg_lab = bg["lab"]

    l_delta = np.abs(lab[:, :, 0] - bg_lab[0])
    a_delta = lab[:, :, 1] - bg_lab[1]
    b_delta = lab[:, :, 2] - bg_lab[2]
    ab_distance = np.sqrt(a_delta ** 2 + b_delta ** 2)

    colour_signal = _otsu_threshold_mask(ab_distance, min_threshold=12.0)
    brightness_signal = _otsu_threshold_mask(l_delta, min_threshold=15.0)

    foreground = (colour_signal | brightness_signal).astype(np.uint8) * 255

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
# HOLE FILLING
# ============================================================

def fill_holes(mask):
    """
    Fill interior holes in the glove silhouette.

    Textured knit gloves and noisy/speckled backgrounds can cause the
    foreground mask to come out as a "swiss cheese" pattern - a mostly
    correct silhouette riddled with small internal holes. A glove is a
    solid object, so any background-coloured pixels found *inside* the
    outer silhouette are treated as noise and filled in.

    This also indirectly protects cuff detection: those holes reduce
    the per-row pixel count unevenly, which was previously enough to
    trigger false "forearm cut" detections on otherwise-good masks.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask (0/255).

    Returns
    -------
    numpy.ndarray
        Mask with interior holes filled.
    """
    if mask is None:
        return mask

    height, width = mask.shape[:2]
    flood_filled = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    # Flood-fill the background starting from the corners (0,0). Only
    # pixels reachable from outside the silhouette without crossing it
    # get filled with 255 in this working copy; the rest are enclosed
    # background pixels, i.e. holes.
    inverted = cv2.bitwise_not(flood_filled)
    cv2.floodFill(inverted, flood_mask, (0, 0), 0)

    filled = cv2.bitwise_or(mask, inverted)
    return filled


# ============================================================
# CUFF DETECTION / FOREARM REMOVAL
# ============================================================

def _smooth_edge_safe(values, kernel_size):
    """
    Moving average that pads with edge values instead of zeros.

    A zero-padded convolution makes the true image border look like an
    artificial drop in width, which previously caused false forearm
    cuts right at the bottom edge of otherwise-good masks. Edge padding
    avoids that artifact.
    """
    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(padded, kernel, mode="valid")


def _color_confirms_forearm(bgr_image, mask, cuff_y, threshold=15.0):
    """
    Confirm a geometric cuff candidate using colour similarity in LAB.

    A forearm (skin) usually looks distinctly different in colour from
    the glove material, while a glove hem/cuff flare that merely
    narrows near the bottom of the frame still looks like the *same*
    material as the rest of the glove. This compares the mean a/b
    (colour-only, lightness-independent) LAB values of:
    - a reference band (20%-45% down the glove, clearly glove material)
    - the candidate region below the detected geometric cuff line

    If the two regions are colour-similar, the geometric cut is treated
    as a false positive (e.g. hem flare) and rejected.

    Parameters
    ----------
    bgr_image : numpy.ndarray
        BGR image resized to match `mask`'s dimensions.
    mask : numpy.ndarray
        Binary glove mask (0/255).
    cuff_y : int
        Y coordinate of the candidate geometric cuff line.
    threshold : float
        Minimum a/b colour distance required to confirm a forearm.

    Returns
    -------
    bool
        True if the candidate region's colour is distinct enough from
        the glove material to confirm a real forearm cut.
    """
    height, width = mask.shape[:2]
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)

    ref_band = mask[int(height * 0.20):int(height * 0.45), :] > 0
    ref_pixels = lab[int(height * 0.20):int(height * 0.45), :][ref_band]

    cand_band = mask[cuff_y:, :] > 0
    cand_pixels = lab[cuff_y:, :][cand_band]

    if ref_pixels.size == 0 or cand_pixels.size == 0:
        return False

    ref_mean = ref_pixels.reshape(-1, 3).mean(axis=0)
    cand_mean = cand_pixels.reshape(-1, 3).mean(axis=0)
    dist_ab = float(np.linalg.norm(ref_mean[1:] - cand_mean[1:]))

    return dist_ab > threshold


def detect_cuff_line(processed, mask):
    """
    Detect a cuff line when a forearm is visible.

    This is a conservative, two-stage check: if there is no reliable
    cuff boundary, the function returns False and the original mask is
    kept.

    Stage 1 - geometric "plateau" detection:
    A real forearm exits through the bottom edge of the photo and holds
    a near-constant width there (a cylindrical arm), unlike the hand or
    a glove hem, which vary in width. The bottom rows of the mask are
    checked for a sufficiently long, sufficiently narrow, sufficiently
    flat plateau of foreground width, then extended upward from that
    plateau using a fixed reference width.

    Stage 2 - colour confirmation:
    Because some gloves flare or fray near the hem in a way that can
    geometrically resemble a plateau, the candidate region is compared
    in LAB colour space against the glove material higher up. A region
    that looks like the *same* material (e.g. more of the glove) is
    rejected; a region that looks distinctly different (e.g. skin under
    a glove) confirms the cut.

    Parameters
    ----------
    processed : dict or None
        Output of preprocess_image(), used to obtain the BGR image for
        the colour-confirmation stage. If None or missing 'original',
        the geometric result is returned without colour confirmation.
    mask : numpy.ndarray
        Binary glove mask (0/255).

    Returns
    -------
    tuple(bool, int or None)
        Whether a cuff line was confirmed, and its y-coordinate.
    """
    if mask is None:
        return False, None

    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        return False, None

    row_counts = (mask.sum(axis=1) // 255).astype(np.float32)
    if row_counts.max() <= 0:
        return False, None

    # Smooth the row-count profile first. A single noisy row (caused by a
    # small unfilled hole or jagged edge pixel) can otherwise look exactly
    # like a sharp "drop" and trigger a false forearm cut on an otherwise
    # good mask. Edge-safe padding avoids treating the real image border
    # as an artificial drop.
    kernel_size = max(3, (min(height, width) // 100) | 1)  # odd, scales with image
    smoothed = _smooth_edge_safe(row_counts, kernel_size)
    peak = smoothed.max()
    if peak <= 0:
        return False, None

    # A real forearm always exits through the bottom image edge, so
    # require some real foreground there as a cheap guard.
    border_band = smoothed[-max(3, kernel_size):]
    if border_band.max() < max(10, 0.05 * width):
        return False, None

    ratio_max = 0.62        # forearm must be visibly narrower than the hand/palm
    variability_max = 0.05  # forearm width is nearly constant (cylindrical arm);
                             # a glove hem/cuff flaring toward its edge is not
    window_frac = 0.12
    window_len = max(15, int(height * window_frac))
    if window_len >= height:
        return False, None

    window = smoothed[height - window_len:]
    window_mean = window.mean()
    window_std = window.std()

    if window_mean <= 0:
        return False, None
    if window_mean > ratio_max * peak:
        return False, None
    if (window_std / window_mean) > variability_max:
        return False, None

    # Extend upward from the window using a FIXED reference (the window's
    # own mean), not an adaptive local mean, so the walk can't drift up a
    # gradual hand taper.
    bottom_start = max(0, int(height * 0.35))
    start = height - window_len
    extend_tol = 0.15
    for index in range(height - window_len - 1, bottom_start - 1, -1):
        value = smoothed[index]
        if value <= 0 or abs(value - window_mean) / window_mean > extend_tol:
            break
        start = index

    cuff_y = start

    # Stage 2: colour confirmation, when a BGR image is available.
    bgr = None
    if isinstance(processed, dict):
        bgr = processed.get("original")
        if bgr is None:
            bgr = processed.get("denoised")
    elif processed is not None:
        bgr = processed

    if bgr is not None:
        if bgr.shape[:2] != (height, width):
            bgr = cv2.resize(bgr, (width, height))
        if not _color_confirms_forearm(bgr, mask, cuff_y):
            return False, None

    return True, cuff_y


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
    if main_component is not None:
        main_component = fill_holes(main_component)

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