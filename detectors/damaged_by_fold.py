"""Damaged-by-fold detector.

Finds unusually strong, elongated crease structures inside the glove surface.
It deliberately ignores the outer glove boundary so finger/palm outlines do not
become false fold detections.
"""

import cv2
import numpy as np


def _empty_result():
    return {
        "defect_name": "damaged_by_fold",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": "CLAHE crease enhancement + interior Canny/Hough line analysis",
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def detect_damaged_by_fold(processed: dict, segmentation: dict) -> dict:
    result = _empty_result()
    glove_mask = segmentation.get("glove_mask")
    gray = processed.get("gray_enhanced")
    if glove_mask is None or gray is None or np.count_nonzero(glove_mask) < 500:
        return result

    mask = (glove_mask > 0).astype(np.uint8) * 255
    h, w = mask.shape[:2]
    scale = max(h, w)

    # Erode enough to remove the natural glove outline/finger borders.
    erode_px = max(5, int(round(scale * 0.012)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    interior = cv2.erode(mask, kernel)
    if np.count_nonzero(interior) < 200:
        interior = mask.copy()

    # Dark and bright linear creases can occur depending on illumination.
    k = max(9, int(scale * 0.025) | 1)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morph_kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel)
    crease_response = cv2.max(blackhat, tophat)
    crease_response = cv2.bitwise_and(crease_response, crease_response, mask=interior)

    nz = crease_response[interior > 0]
    if nz.size == 0:
        return result
    thresh_val = max(18, int(np.percentile(nz, 82)))
    _, strong = cv2.threshold(crease_response, thresh_val, 255, cv2.THRESH_BINARY)

    # Edge geometry: folds tend to form long, coherent lines rather than speckle.
    edges = cv2.Canny(gray, 45, 120)
    edges = cv2.bitwise_and(edges, interior)
    candidate_edges = cv2.bitwise_and(edges, strong)

    min_len = max(25, int(scale * 0.10))
    max_gap = max(6, int(scale * 0.018))
    lines = cv2.HoughLinesP(candidate_edges, 1, np.pi / 180.0,
                            threshold=max(18, int(scale * 0.035)),
                            minLineLength=min_len, maxLineGap=max_gap)

    defect_mask = np.zeros_like(mask)
    lengths = []
    if lines is not None:
        thickness = max(4, int(scale * 0.008))
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            length = float(np.hypot(x2 - x1, y2 - y1))
            lengths.append(length)
            cv2.line(defect_mask, (x1, y1), (x2, y2), 255, thickness)

    defect_mask = cv2.bitwise_and(defect_mask, interior)
    total_line_length = float(sum(lengths))
    longest_line = float(max(lengths) if lengths else 0.0)
    line_count = len(lengths)

    # Combine longest crease and accumulated crease evidence.
    longest_signal = np.clip(longest_line / max(scale * 0.30, 1.0), 0.0, 1.0)
    total_signal = np.clip(total_line_length / max(scale * 0.80, 1.0), 0.0, 1.0)
    response_area = np.count_nonzero(strong) / max(np.count_nonzero(interior), 1)
    area_signal = np.clip(response_area / 0.10, 0.0, 1.0)
    score = float(np.clip(0.50 * longest_signal + 0.35 * total_signal + 0.15 * area_signal, 0.0, 1.0))
    detected = score >= 0.50 and longest_line >= min_len

    bbox = None
    if detected and np.count_nonzero(defect_mask):
        ys, xs = np.where(defect_mask > 0)
        x, y = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)
        bbox = (x, y, bw, bh)
    elif not detected:
        defect_mask[:] = 0

    area_pct = 100.0 * np.count_nonzero(defect_mask) / max(np.count_nonzero(mask), 1)
    result.update({
        "detected": bool(detected),
        "detection_score": score,
        "bounding_box": bbox,
        "mask": defect_mask,
        "measurements": {
            "area_pct": round(float(area_pct), 3),
            "crease_line_count": int(line_count),
            "longest_crease_px": round(longest_line, 2),
            "total_crease_length_px": round(total_line_length, 2),
            "strong_crease_area_ratio": round(float(response_area), 4),
        },
    })
    return result
