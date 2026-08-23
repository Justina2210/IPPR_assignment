import cv2
import numpy as np


def _as_bgr_image(image):
    """Return a BGR image from either a processed dictionary or a NumPy array."""
    if isinstance(image, dict):
        if "original" in image:
            return image["original"]
        if "denoised" in image:
            return image["denoised"]
        raise ValueError("Dictionary image input must contain 'original' or 'denoised'.")
    return image


def estimate_background(image):
    """Estimate background colour as the median LAB value across the image border (robust to shadows)."""
    bgr = _as_bgr_image(image)
    if bgr is None or bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    # LAB, not HSV: keeps hue distances comparable for similar-hue glove/background pairs.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]

    margin = max(10, min(height, width) // 12)
    border_mask = np.zeros((height, width), dtype=np.uint8)
    border_mask[:margin, :] = 1
    border_mask[-margin:, :] = 1
    border_mask[:, :margin] = 1
    border_mask[:, -margin:] = 1

    border_pixels = lab[border_mask == 1]
    if border_pixels.size == 0:
        return {"lab": np.array([0.0, 0.0, 0.0], dtype=np.float32)}

    background_lab = np.median(border_pixels.reshape(-1, 3), axis=0)
    return {"lab": background_lab.astype(np.float32)}


def _otsu_threshold_mask(distance_map, min_threshold=0.0):
    """Otsu-threshold a distance map into foreground/background."""
    # Otsu, not a fixed cutoff: a fixed formula fails once the glove fills most of the frame.
    max_value = float(distance_map.max())
    if max_value <= 1e-6:
        return np.zeros(distance_map.shape, dtype=bool)

    scaled = np.clip(distance_map / max_value * 255.0, 0, 255).astype(np.uint8)
    otsu_level, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(min_threshold, (otsu_level / 255.0) * max_value)

    return distance_map > threshold


def create_foreground_mask(image):
    """Foreground mask from LAB colour + lightness distance to the estimated background, Otsu-thresholded and OR'd."""
    bgr = _as_bgr_image(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = estimate_background(bgr)
    bg_lab = bg["lab"]

    l_delta = np.abs(lab[:, :, 0] - bg_lab[0])
    a_delta = lab[:, :, 1] - bg_lab[1]
    b_delta = lab[:, :, 2] - bg_lab[2]
    ab_distance = np.sqrt(a_delta ** 2 + b_delta ** 2)

    # Colour distance catches most changes; lightness distance catches pale gloves on bright backgrounds.
    colour_signal = _otsu_threshold_mask(ab_distance, min_threshold=12.0)
    brightness_signal = _otsu_threshold_mask(l_delta, min_threshold=15.0)

    foreground = (colour_signal | brightness_signal).astype(np.uint8) * 255

    return foreground


def clean_mask(mask):
    """Morphological open+close to remove small mask noise while keeping the glove silhouette."""
    if mask is None:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def keep_largest_component(mask):
    """Keep only the largest connected component, dropping isolated background fragments."""
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


def fill_holes(mask):
    """Flood-fill interior holes in the glove silhouette (fixes knit-texture 'swiss cheese' masks)."""
    if mask is None:
        return mask

    # Also protects cuff detection: holes make per-row pixel counts uneven.
    height, width = mask.shape[:2]
    flood_filled = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    # Flood-fill from the corner; unreached background pixels are holes.
    inverted = cv2.bitwise_not(flood_filled)
    cv2.floodFill(inverted, flood_mask, (0, 0), 0)

    filled = cv2.bitwise_or(mask, inverted)
    return filled


def _smooth_edge_safe(values, kernel_size):
    """Moving average with edge padding (not zero padding), so the real image border isn't mistaken for a width drop."""
    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(padded, kernel, mode="valid")


def _color_confirms_forearm(bgr_image, mask, cuff_y, threshold=15.0):
    """Confirm a geometric cuff candidate: distinct LAB colour below the line vs. glove material above it means real forearm/skin."""
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
    """Detect a forearm cuff line via a geometric width-plateau check confirmed by LAB colour; conservative, returns False if unsure."""
    if mask is None:
        return False, None

    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        return False, None

    row_counts = (mask.sum(axis=1) // 255).astype(np.float32)
    if row_counts.max() <= 0:
        return False, None

    # Smooth first so one noisy row doesn't look like a sharp width drop.
    kernel_size = max(3, (min(height, width) // 100) | 1)  # odd, scales with image
    smoothed = _smooth_edge_safe(row_counts, kernel_size)
    peak = smoothed.max()
    if peak <= 0:
        return False, None

    # Guard: a real forearm always exits the bottom edge.
    border_band = smoothed[-max(3, kernel_size):]
    if border_band.max() < max(10, 0.05 * width):
        return False, None

    ratio_max = 0.62        # forearm must be visibly narrower than the hand/palm
    variability_max = 0.05  # forearm width is nearly constant; a glove hem flare is not
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

    # Extend upward using the window's fixed mean, not local, so it can't drift up a hand taper.
    bottom_start = max(0, int(height * 0.35))
    start = height - window_len
    extend_tol = 0.15
    for index in range(height - window_len - 1, bottom_start - 1, -1):
        value = smoothed[index]
        if value <= 0 or abs(value - window_mean) / window_mean > extend_tol:
            break
        start = index

    cuff_y = start

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
    """Zero out the mask below the cuff line."""
    if mask is None or cuff_y is None:
        return mask

    output = mask.copy()
    height = output.shape[0]
    if cuff_y >= height:
        return output

    output[cuff_y:, :] = 0
    return output


def segment_glove(processed):
    """Run the full pipeline: background estimate -> foreground mask -> cleaning -> largest component -> hole fill -> optional forearm removal."""
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
