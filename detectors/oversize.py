"""
oversize.py
-----------
Detector for the "oversize" glove defect.

Oversize is different from local defects such as discoloration or plastic
contamination: the defect affects the glove's overall fit/geometry rather
than one small patch. Therefore this detector evaluates GLOBAL loose-fit
evidence from the segmented glove.

Algorithm
---------
1. Use segmentation["glove_mask"] only.
2. Measure global silhouette / fit features:
   - contour concavity (1 - solidity)
   - contour roughness (perimeter / convex-hull perimeter)
   - cuff fullness relative to the palm
   - bounding-box width / height (spread/baggy geometry)
   - silhouette extent (especially useful for bulky cotton gloves)
3. For smooth latex/nitrile gloves, also measure loose-fold/wrinkle density
   using low-frequency grayscale shading residuals.
4. Automatically choose a cotton-style or smooth-glove scoring model from
   the glove's texture.
5. Produce:
   - detection_score : global oversize confidence
   - bounding_box    : evidence-region bounding box for the shared evaluator
   - mask            : interpretable oversize evidence
                       * smooth gloves: loose folds + palm/cuff measurement lines
                       * cotton: palm/cuff measurement lines + lower-cuff outline

The full glove bounding box is still stored in measurements['glove_bounding_box'].

Prototype integration
---------------------
The detector also returns measurements['prototype_details'], a JSON-friendly
dictionary containing the explanation, legend and display metrics. The GUI
can render these values beside the output image without writing explanatory
text permanently onto the image.

Important limitation
--------------------
No normal/control gloves were supplied when this detector was developed.
Thresholds were tuned by inspection on the provided oversize samples.
Therefore the method should be described as a heuristic loose-fit detector,
not a calibrated physical glove-size measurement. If normal gloves become
available, the thresholds should be validated against them.

Required shared contract:
    detect_oversize(processed: dict, segmentation: dict) -> dict
"""

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

LOCAL_DETECTION_THRESHOLD = 0.50

# Used only to decide whether strong knit texture is present.
COTTON_MEDIAN_GRADIENT_THRESHOLD = 12.0

# Smooth-glove fold analysis.
FOLD_LOCAL_SIGMA = 12.0
FOLD_DARK_THRESHOLD = 6.0
FOLD_EDGE_MARGIN_PX = 12

# Explainable overlay geometry.
MEASUREMENT_LINE_THICKNESS = 5
PALM_MEASUREMENT_FRACTION = 0.60
CUFF_MEASUREMENT_FRACTION = 0.84
COTTON_LOWER_BOUNDARY_START = 0.70

# Score normalization ranges.
# These were selected by inspection of the supplied oversize samples.
SOLIDITY_HIGH = 0.88
SOLIDITY_RANGE = 0.20

CUFF_RATIO_LOW = 0.65
CUFF_RATIO_RANGE = 0.35

ROUGHNESS_LOW = 1.45
ROUGHNESS_RANGE = 0.45

FOLD_DENSITY_LOW = 0.03
FOLD_DENSITY_RANGE = 0.13

SPREAD_RATIO_LOW = 0.60
SPREAD_RATIO_RANGE = 0.45

COTTON_EXTENT_LOW = 0.55
COTTON_EXTENT_RANGE = 0.16


# ============================================================
# HELPERS
# ============================================================

def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _largest_contour(mask_bool):
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def _bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)

    if ys.size == 0:
        return None

    x = int(xs.min())
    y = int(ys.min())
    w = int(xs.max() - x + 1)
    h = int(ys.max() - y + 1)

    return (x, y, w, h)


def _row_width_profile(mask_bool):
    return mask_bool.sum(axis=1).astype(np.float32)


def _safe_mean(values):
    return float(values.mean()) if values.size else 0.0


def _extract_geometry(mask_bool):
    """
    Extract global geometry / fit measurements from the glove silhouette.
    """
    contour = _largest_contour(mask_bool)

    if contour is None:
        return None

    glove_bbox = _bbox_from_mask(mask_bool)
    x, y, w, h = glove_bbox

    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))

    solidity = (
        contour_area / hull_area
        if hull_area > 1e-6
        else 1.0
    )

    contour_roughness = (
        perimeter / hull_perimeter
        if hull_perimeter > 1e-6
        else 1.0
    )

    bbox_area = max(w * h, 1)
    extent = float(mask_bool.sum() / bbox_area)

    width_height_ratio = float(w / max(h, 1))

    # Row-width measurements are made relative to the glove's own height,
    # so they are independent of absolute image scale.
    row_counts = _row_width_profile(mask_bool)

    y0 = y
    y1 = y + h

    palm_start = int(y0 + 0.45 * h)
    palm_end = int(y0 + 0.70 * h)

    lower_start = int(y0 + 0.80 * h)
    lower_end = y1

    palm_width = _safe_mean(
        row_counts[palm_start:palm_end]
    )

    lower_width = _safe_mean(
        row_counts[lower_start:lower_end]
    )

    cuff_ratio = (
        lower_width / palm_width
        if palm_width > 1e-6
        else 0.0
    )

    return {
        "bounding_box": glove_bbox,
        "solidity": solidity,
        "contour_roughness": contour_roughness,
        "extent": extent,
        "width_height_ratio": width_height_ratio,
        "palm_width": palm_width,
        "lower_width": lower_width,
        "cuff_ratio": cuff_ratio,
    }


def _infer_texture_mode(processed, mask_bool):
    """
    Cotton knit has much stronger fine gradients than smooth latex/nitrile.
    """
    gray = processed["gray"].astype(np.float32)

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = cv2.magnitude(gx, gy)

    median_gradient = float(
        np.median(gradient[mask_bool])
    )

    mode = (
        "cotton"
        if median_gradient >= COTTON_MEDIAN_GRADIENT_THRESHOLD
        else "smooth"
    )

    return mode, median_gradient


def _smooth_glove_fold_evidence(processed, mask_bool):
    """
    Detect broad dark fold/shadow evidence inside smooth gloves.

    A large Gaussian reference removes the glove's slow illumination
    gradient. Pixels substantially darker than their local reference are
    treated as fold/wrinkle evidence.
    """
    gray = processed["gray"].astype(np.float32)

    local_reference = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=FOLD_LOCAL_SIGMA,
        sigmaY=FOLD_LOCAL_SIGMA,
    )

    dark_residual = np.maximum(
        local_reference - gray,
        0.0,
    )

    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * FOLD_EDGE_MARGIN_PX + 1,
            2 * FOLD_EDGE_MARGIN_PX + 1,
        ),
    )

    interior = cv2.erode(
        mask_bool.astype(np.uint8) * 255,
        interior_kernel,
    ) > 0

    if not np.any(interior):
        interior = mask_bool.copy()

    fold_bool = (
        (dark_residual > FOLD_DARK_THRESHOLD)
        & interior
    )

    # Remove tiny isolated texture points while retaining real fold lines.
    fold_mask = fold_bool.astype(np.uint8) * 255

    fold_mask = cv2.morphologyEx(
        fold_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )

    fold_mask = cv2.morphologyEx(
        fold_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
    )

    fold_density = float(
        np.count_nonzero(
            fold_mask[interior] > 0
        )
        / max(
            np.count_nonzero(interior),
            1,
        )
    )

    return fold_density, fold_mask


def _boundary_evidence(mask_bool, thickness=6):
    """
    Create a thin inside-glove boundary band for visualizing that oversize
    is a global silhouette/fit defect rather than a small local patch.
    """
    mask_u8 = mask_bool.astype(np.uint8) * 255

    eroded = cv2.erode(
        mask_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * thickness + 1, 2 * thickness + 1),
        ),
    )

    return cv2.subtract(mask_u8, eroded)



def _row_span(mask_bool, row_y):
    """
    Return the left/right foreground coordinates on one glove row.
    """
    height, width = mask_bool.shape[:2]

    row_y = int(np.clip(row_y, 0, height - 1))
    xs = np.where(mask_bool[row_y])[0]

    if xs.size == 0:
        return None

    return (
        int(xs.min()),
        int(xs.max()),
        int(xs.size),
    )


def _find_nearest_valid_row(mask_bool, target_y, search_radius=30):
    """
    Find the closest row to target_y that still contains glove pixels.
    """
    height = mask_bool.shape[0]

    target_y = int(np.clip(target_y, 0, height - 1))

    for delta in range(search_radius + 1):
        candidates = [target_y - delta, target_y + delta]

        for row_y in candidates:
            if row_y < 0 or row_y >= height:
                continue

            span = _row_span(mask_bool, row_y)

            if span is not None and span[2] > 0:
                return row_y, span

    return None, None


def _measurement_rows(mask_bool, glove_bbox):
    """
    Locate explainable palm and cuff measurement rows.

    These are NOT hidden-hand boundaries. They are simply two horizontal
    cross-sections of the segmented glove used to explain oversize geometry.
    """
    x, y, w, h = glove_bbox

    palm_target = int(y + PALM_MEASUREMENT_FRACTION * h)
    cuff_target = int(y + CUFF_MEASUREMENT_FRACTION * h)

    palm_y, palm_span = _find_nearest_valid_row(
        mask_bool,
        palm_target,
    )

    cuff_y, cuff_span = _find_nearest_valid_row(
        mask_bool,
        cuff_target,
    )

    return {
        "palm_y": palm_y,
        "palm_span": palm_span,
        "cuff_y": cuff_y,
        "cuff_span": cuff_span,
    }


def _draw_measurement_line(mask, row_y, span, thickness=MEASUREMENT_LINE_THICKNESS):
    """
    Draw one horizontal evidence line only over foreground glove pixels.
    """
    if row_y is None or span is None:
        return

    x_left, x_right, _ = span

    half = max(1, thickness // 2)
    y0 = max(0, row_y - half)
    y1 = min(mask.shape[0], row_y + half + 1)

    mask[y0:y1, x_left:x_right + 1] = 255


def _measurement_evidence_mask(mask_bool, glove_bbox):
    """
    Create palm/cuff measurement lines and return their coordinates.
    """
    evidence = np.zeros(
        mask_bool.shape,
        dtype=np.uint8,
    )

    rows = _measurement_rows(
        mask_bool,
        glove_bbox,
    )

    _draw_measurement_line(
        evidence,
        rows["palm_y"],
        rows["palm_span"],
    )

    _draw_measurement_line(
        evidence,
        rows["cuff_y"],
        rows["cuff_span"],
    )

    # Restrict every drawn line to the segmented glove.
    evidence[~mask_bool] = 0

    return evidence, rows


def _cotton_lower_boundary_evidence(mask_bool, glove_bbox, thickness=6):
    """
    Cotton wrinkle texture is naturally strong, so visualize the lower
    silhouette/cuff region instead of highlighting knit texture everywhere.
    """
    boundary = _boundary_evidence(
        mask_bool,
        thickness=thickness,
    )

    _, y, _, h = glove_bbox

    lower_y = int(
        y + COTTON_LOWER_BOUNDARY_START * h
    )

    lower_band = np.zeros(
        mask_bool.shape,
        dtype=np.uint8,
    )

    lower_band[
        max(0, lower_y):,
        :
    ] = 255

    return cv2.bitwise_and(
        boundary,
        lower_band,
    )


def _line_tuple(row_y, span):
    """
    Convert a row/span to a JSON/CSV-friendly tuple.
    """
    if row_y is None or span is None:
        return None

    x_left, x_right, width = span

    return (
        int(x_left),
        int(row_y),
        int(x_right),
        int(row_y),
    )


def _geometry_subscores(geometry):
    concavity_score = _clip01(
        (
            SOLIDITY_HIGH
            - geometry["solidity"]
        )
        / SOLIDITY_RANGE
    )

    cuff_fullness_score = _clip01(
        (
            geometry["cuff_ratio"]
            - CUFF_RATIO_LOW
        )
        / CUFF_RATIO_RANGE
    )

    roughness_score = _clip01(
        (
            geometry["contour_roughness"]
            - ROUGHNESS_LOW
        )
        / ROUGHNESS_RANGE
    )

    spread_score = _clip01(
        (
            geometry["width_height_ratio"]
            - SPREAD_RATIO_LOW
        )
        / SPREAD_RATIO_RANGE
    )

    bulky_extent_score = _clip01(
        (
            geometry["extent"]
            - COTTON_EXTENT_LOW
        )
        / COTTON_EXTENT_RANGE
    )

    return {
        "concavity_score": concavity_score,
        "cuff_fullness_score": cuff_fullness_score,
        "roughness_score": roughness_score,
        "spread_score": spread_score,
        "bulky_extent_score": bulky_extent_score,
    }



def _build_prototype_details(
    mode,
    detected,
    score,
    palm_width,
    cuff_width,
    geometry,
    fold_density,
):
    """
    Build JSON-friendly presentation data for the prototype.

    This function does NOT affect the detector score. It only converts
    measurements that were already calculated into labels, metrics,
    legend items and a short explanation for the GUI.
    """
    cuff_palm_ratio = (
        round(float(cuff_width) / float(palm_width), 3)
        if palm_width not in (None, 0) and cuff_width is not None
        else None
    )

    common_metrics = [
        {
            "key": "palm_width_px",
            "label": "Palm Width",
            "value": int(palm_width) if palm_width is not None else None,
            "unit": "px",
        },
        {
            "key": "cuff_width_px",
            "label": "Cuff Width",
            "value": int(cuff_width) if cuff_width is not None else None,
            "unit": "px",
        },
        {
            "key": "cuff_palm_ratio",
            "label": "Cuff / Palm Ratio",
            "value": cuff_palm_ratio,
            "unit": "",
        },
        {
            "key": "solidity",
            "label": "Silhouette Solidity",
            "value": round(float(geometry["solidity"]), 3),
            "unit": "",
        },
    ]

    if mode == "smooth":
        display_metrics = common_metrics + [
            {
                "key": "fold_density",
                "label": "Loose-Fold Density",
                "value": round(100.0 * float(fold_density), 1),
                "unit": "%",
            },
        ]

        evidence_type = "loose_fit"
        evidence_label = "Loose Fold / Excess Material"
        explanation = (
            "Oversize evidence is based on loose-fold regions together with "
            "global glove-fit geometry. Broad folds indicate excess material, "
            "while palm/cuff measurements and silhouette features support the "
            "loose-fitting classification."
        )

        legend = [
            {"name": "Loose / excess material", "colour": "yellow"},
            {"name": "Palm width", "colour": "cyan"},
            {"name": "Cuff width", "colour": "orange"},
            {"name": "Oversize evidence region", "colour": "magenta"},
        ]

    else:
        display_metrics = common_metrics + [
            {
                "key": "silhouette_extent",
                "label": "Silhouette Extent",
                "value": round(float(geometry["extent"]), 3),
                "unit": "",
            },
            {
                "key": "width_height_ratio",
                "label": "Width / Height Ratio",
                "value": round(float(geometry["width_height_ratio"]), 3),
                "unit": "",
            },
        ]

        evidence_type = "cuff_geometry"
        evidence_label = "Lower / Cuff Geometry"
        explanation = (
            "Cotton knit texture naturally contains many edges and wrinkles, "
            "so oversize is assessed mainly from cuff fullness and global "
            "silhouette geometry instead of wrinkle texture."
        )

        legend = [
            {"name": "Lower / cuff geometry", "colour": "yellow"},
            {"name": "Palm width", "colour": "cyan"},
            {"name": "Cuff width", "colour": "orange"},
            {"name": "Oversize evidence region", "colour": "magenta"},
        ]

    return {
        "panel_title": "Oversize Evidence",
        "status": "Detected" if detected else "Not Detected",
        "confidence_percent": round(100.0 * float(score), 1),
        "material_mode": mode,
        "evidence_type": evidence_type,
        "evidence_label": evidence_label,
        "explanation": explanation,
        "display_metrics": display_metrics,
        "legend": legend,
    }


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_oversize(
    processed: dict,
    segmentation: dict,
) -> dict:
    """
    Detect an oversized / loose-fitting glove.

    Oversize is a global fit defect, but the returned mask is designed to
    explain the decision visually rather than simply highlighting the whole
    glove outline.

    Smooth latex/nitrile:
      - highlight loose-fold evidence
      - draw palm/cuff measurement cross-sections

    Cotton:
      - draw palm/cuff measurement cross-sections
      - highlight only the lower/cuff silhouette region
    """
    result = {
        "defect_name": "oversize",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": (
            "Explainable global fit geometry "
            "+ loose-fold evidence"
        ),
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }

    if processed is None or segmentation is None:
        return result

    glove_mask = segmentation.get("glove_mask")
    glove_area = int(
        segmentation.get("glove_area", 0)
    )

    if (
        glove_mask is None
        or glove_area <= 0
        or not np.any(glove_mask > 0)
    ):
        return result

    mask_bool = glove_mask > 0

    geometry = _extract_geometry(mask_bool)

    if geometry is None:
        return result

    mode, median_gradient = _infer_texture_mode(
        processed,
        mask_bool,
    )

    subscores = _geometry_subscores(geometry)

    measurement_mask, measurement_rows = (
        _measurement_evidence_mask(
            mask_bool,
            geometry["bounding_box"],
        )
    )

    fold_density = 0.0
    fold_score = 0.0
    fold_mask = np.zeros_like(glove_mask)

    if mode == "smooth":
        fold_density, fold_mask = (
            _smooth_glove_fold_evidence(
                processed,
                mask_bool,
            )
        )

        fold_score = _clip01(
            (
                fold_density
                - FOLD_DENSITY_LOW
            )
            / FOLD_DENSITY_RANGE
        )

        # Smooth latex/nitrile:
        # folds carry the most weight, but geometry is still required so
        # ordinary texture alone cannot dominate the decision.
        base_geometry_fold_score = (
            0.38 * fold_score
            + 0.22 * subscores["concavity_score"]
            + 0.18 * subscores["cuff_fullness_score"]
            + 0.14 * subscores["roughness_score"]
            + 0.08 * subscores["spread_score"]
        )

        # A second loose-fit path is useful for upright gloves where the
        # silhouette itself is not strongly concave/spread, but the glove
        # is visibly bulky and contains many broad folds. This avoids
        # under-scoring samples such as a straight, oversized latex glove.
        fold_bulk_score = (
            0.55 * fold_score
            + 0.45 * subscores["bulky_extent_score"]
        )

        detection_score = max(
            base_geometry_fold_score,
            fold_bulk_score,
        )

        # Explainable output:
        #   yellow fold pixels = loose/excess material evidence
        #   horizontal yellow lines = palm/cuff geometry measurements
        evidence_mask = cv2.bitwise_or(
            fold_mask,
            measurement_mask,
        )

    else:
        # Cotton knit texture makes wrinkle counting unreliable.
        # Use shape/fullness features instead.
        detection_score = (
            0.25 * subscores["concavity_score"]
            + 0.22 * subscores["cuff_fullness_score"]
            + 0.18 * subscores["roughness_score"]
            + 0.25 * subscores["bulky_extent_score"]
            + 0.10 * subscores["spread_score"]
        )

        cotton_lower_boundary = (
            _cotton_lower_boundary_evidence(
                mask_bool,
                geometry["bounding_box"],
                thickness=6,
            )
        )

        # Cotton already has strong knit texture, so do not highlight
        # wrinkle-like texture. Show the two geometry cross-sections and
        # the lower/cuff silhouette that contributes to cuff-fullness.
        evidence_mask = cv2.bitwise_or(
            measurement_mask,
            cotton_lower_boundary,
        )

    detection_score = float(
        np.clip(
            detection_score,
            0.0,
            1.0,
        )
    )

    result["detected"] = (
        detection_score
        >= LOCAL_DETECTION_THRESHOLD
    )

    result["detection_score"] = detection_score

    # The shared evaluator always draws a rectangular bounding box.
    # To avoid a meaningless giant box around the entire glove, return a
    # box around the EXPLAINABLE EVIDENCE region. The complete glove box
    # remains available in measurements["glove_bounding_box"].
    evidence_bbox = _bbox_from_mask(
        evidence_mask > 0
    )

    result["bounding_box"] = evidence_bbox
    result["mask"] = evidence_mask

    evidence_area = int(
        np.count_nonzero(
            evidence_mask > 0
        )
    )

    palm_width_px = (
        int(measurement_rows["palm_span"][2])
        if measurement_rows["palm_span"] is not None
        else None
    )

    cuff_width_px = (
        int(measurement_rows["cuff_span"][2])
        if measurement_rows["cuff_span"] is not None
        else None
    )

    prototype_details = _build_prototype_details(
        mode=mode,
        detected=result["detected"],
        score=detection_score,
        palm_width=palm_width_px,
        cuff_width=cuff_width_px,
        geometry=geometry,
        fold_density=fold_density,
    )

    result["measurements"] = {
        "area_pct": round(
            100.0
            * evidence_area
            / max(glove_area, 1),
            2,
        ),
        "global_defect": True,
        "mode": mode,
        "visualization_type": (
            "loose_folds_plus_measurement_lines"
            if mode == "smooth"
            else "cuff_geometry_plus_measurement_lines"
        ),
        "evidence_description": (
            "Loose folds/excess material plus palm and cuff width measurements"
            if mode == "smooth"
            else "Palm/cuff width measurements plus lower-cuff silhouette"
        ),
        "glove_bounding_box": tuple(
            int(v)
            for v in geometry["bounding_box"]
        ),
        "palm_measurement_line": _line_tuple(
            measurement_rows["palm_y"],
            measurement_rows["palm_span"],
        ),
        "cuff_measurement_line": _line_tuple(
            measurement_rows["cuff_y"],
            measurement_rows["cuff_span"],
        ),
        "palm_measurement_width_px": palm_width_px,
        "cuff_measurement_width_px": cuff_width_px,
        "cuff_palm_ratio": (
            round(
                float(cuff_width_px) / float(palm_width_px),
                4,
            )
            if palm_width_px not in (None, 0) and cuff_width_px is not None
            else None
        ),
        "prototype_details": prototype_details,
        "median_gradient": round(
            median_gradient,
            3,
        ),
        "solidity": round(
            geometry["solidity"],
            4,
        ),
        "contour_roughness": round(
            geometry["contour_roughness"],
            4,
        ),
        "extent": round(
            geometry["extent"],
            4,
        ),
        "width_height_ratio": round(
            geometry["width_height_ratio"],
            4,
        ),
        "cuff_ratio": round(
            geometry["cuff_ratio"],
            4,
        ),
        "fold_density": round(
            fold_density,
            4,
        ),
        "fold_score": round(
            fold_score,
            3,
        ),
        "concavity_score": round(
            subscores["concavity_score"],
            3,
        ),
        "cuff_fullness_score": round(
            subscores["cuff_fullness_score"],
            3,
        ),
        "roughness_score": round(
            subscores["roughness_score"],
            3,
        ),
        "spread_score": round(
            subscores["spread_score"],
            3,
        ),
        "bulky_extent_score": round(
            subscores["bulky_extent_score"],
            3,
        ),
        "base_geometry_fold_score": round(
            base_geometry_fold_score
            if mode == "smooth"
            else detection_score,
            3,
        ),
        "fold_bulk_score": round(
            fold_bulk_score
            if mode == "smooth"
            else 0.0,
            3,
        ),
    }

    return result


# ============================================================
# OPTIONAL QUICK TEST
# ============================================================

if __name__ == "__main__":
    import os

    from preprocessing import (
        load_image,
        preprocess_image,
    )
    from segmentation import segment_glove

    candidate_folders = [
        os.path.join(
            "datasets",
            "cotton",
            "oversize",
        ),
        os.path.join(
            "datasets",
            "latex",
            "oversize",
        ),
        os.path.join(
            "datasets",
            "nitrile",
            "oversize",
        ),
        os.path.join(
            "dataset",
            "cotton",
            "oversize",
        ),
        os.path.join(
            "dataset",
            "latex",
            "oversize",
        ),
        os.path.join(
            "dataset",
            "nitrile",
            "oversize",
        ),
    ]

    sample_path = None

    for folder in candidate_folders:
        if not os.path.isdir(folder):
            continue

        for filename in sorted(
            os.listdir(folder)
        ):
            if filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                )
            ):
                sample_path = os.path.join(
                    folder,
                    filename,
                )
                break

        if sample_path is not None:
            break

    if sample_path is None:
        print(
            "No oversize image found under "
            "datasets/ or dataset/."
        )

    else:
        image = load_image(sample_path)
        processed = preprocess_image(image)
        segmentation = segment_glove(
            processed
        )

        output = detect_oversize(
            processed,
            segmentation,
        )

        print(f"Image: {sample_path}")
        print(
            f"Detected: "
            f"{output['detected']}"
        )
        print(
            "Score:",
            round(
                output["detection_score"],
                3,
            ),
        )
        print(
            "Bounding box:",
            output["bounding_box"],
        )
        print(
            "Measurements:",
            output["measurements"],
        )
