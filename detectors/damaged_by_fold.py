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
        "algorithm": "multiscale crease enhancement + coherent interior line clusters",
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

    # Dark and bright folds occur at several widths. Taking the maximum of two
    # scales preserves narrow latex creases and broad rolled/folded cotton cuffs.
    responses = []
    for frac in (0.018, 0.045):
        k = max(9, int(scale * frac) | 1)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        responses.append(cv2.max(
            cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morph_kernel),
            cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel),
        ))
    crease_response = cv2.max(responses[0], responses[1])
    crease_response = cv2.bitwise_and(crease_response, crease_response, mask=interior)

    nz = crease_response[interior > 0]
    if nz.size == 0:
        return result
    # A high percentile rejects knitted texture while remaining adaptive to the
    # much smoother latex/nitrile surfaces.
    thresh_val = max(12, int(np.percentile(nz, 88)))
    _, strong = cv2.threshold(crease_response, thresh_val, 255, cv2.THRESH_BINARY)

    # Edge geometry: folds tend to form long, coherent lines rather than speckle.
    median_gray = float(np.median(gray[interior > 0]))
    edges = cv2.Canny(gray, int(max(20, 0.55 * median_gray)),
                      int(min(220, max(60, 1.35 * median_gray))))
    edges = cv2.bitwise_and(edges, interior)
    # Dilate the response slightly before intersecting with Canny: the maximum
    # morphology response lies beside a crease edge, not always on the same pixel.
    support = cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    candidate_edges = cv2.bitwise_and(edges, support)

    min_len = max(25, int(scale * 0.10))
    max_gap = max(6, int(scale * 0.018))
    lines = cv2.HoughLinesP(candidate_edges, 1, np.pi / 180.0,
                            threshold=max(18, int(scale * 0.035)),
                            minLineLength=min_len, maxLineGap=max_gap)

    defect_mask = np.zeros_like(mask)
    segments = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
            segments.append((x1, y1, x2, y2, length, angle))

    # Keep only segments supported by another nearby, similarly oriented segment.
    # This suppresses isolated seams and finger texture while retaining both sides
    # of a physical fold/overlap.
    accepted = []
    near = max(15.0, 0.08 * scale)
    for i, seg in enumerate(segments):
        x1, y1, x2, y2, length, angle = seg
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        supported = length >= 0.20 * scale
        for j, other in enumerate(segments):
            if i == j:
                continue
            ox, oy = (other[0] + other[2]) / 2.0, (other[1] + other[3]) / 2.0
            angle_delta = abs(angle - other[5])
            angle_delta = min(angle_delta, 180.0 - angle_delta)
            if angle_delta <= 18.0 and np.hypot(mx - ox, my - oy) <= near:
                supported = True
                break
        if supported:
            accepted.append(seg)

    thickness = max(5, int(scale * 0.012))
    for x1, y1, x2, y2, _, _ in accepted:
        cv2.line(defect_mask, (x1, y1), (x2, y2), 255, thickness)

    defect_mask = cv2.bitwise_and(defect_mask, interior)
    lengths = [s[4] for s in accepted]
    total_line_length = float(sum(lengths))
    longest_line = float(max(lengths) if lengths else 0.0)
    line_count = len(lengths)

    # Combine longest crease and accumulated crease evidence.
    longest_signal = np.clip(longest_line / max(scale * 0.30, 1.0), 0.0, 1.0)
    total_signal = np.clip(total_line_length / max(scale * 0.80, 1.0), 0.0, 1.0)
    response_area = np.count_nonzero(strong) / max(np.count_nonzero(interior), 1)
    area_signal = np.clip(response_area / 0.10, 0.0, 1.0)
    score = float(np.clip(0.50 * longest_signal + 0.35 * total_signal + 0.15 * area_signal, 0.0, 1.0))
    coherence = len(accepted) / max(len(segments), 1)
    score = float(np.clip(score * (0.75 + 0.25 * coherence), 0.0, 1.0))
    detected = score >= 0.50 and longest_line >= min_len and line_count > 0

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
            "raw_line_count": int(len(segments)),
            "line_coherence": round(float(coherence), 4),
        },
    })
    return result
