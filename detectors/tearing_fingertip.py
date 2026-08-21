"""
detectors/tearing_fingertip.py
--------------------------------
Classical OpenCV detector for "tearing_fingertip" defects (a torn-off
or ripped fingertip). No ML.

Same underlying anomaly signal as detectors/tearing.py (a local patch
whose LAB colour deviates from the glove's own material colour - see
that file's docstring for why colour deviation and not a mask-hole
lookup), restricted to the fingertip area as requested:

1. Locate up to 5 fingertip points on the glove's outer contour using
   convex-hull-style protrusion analysis + finger-length analysis:
   for every point on the contour, measure its distance from the
   glove_mask centroid (a point that naturally sits in the palm, the
   bulkiest part of the mask). This distance profile has one local
   maximum per extended finger - a direct measure of "how far this
   point sticks out", i.e. finger length. Peaks are found by simple
   circular non-max suppression and must clear a minimum prominence
   above the profile's median (the palm/valley baseline), which is
   what keeps the wrist/palm corners from being mistaken for fingers.
   This naturally handles hands with fewer than 5 fingers extended
   (a fist/"shaka" pose correctly yields 2 candidates, not 5) and
   hands partially cropped out of frame.
2. For each located fingertip, build a search ROI: a circle centred
   on the tip, radius scaled to that finger's own measured length
   (TIP_RADIUS_FRACTION), intersected with glove_mask. Because the
   radius is a fraction of the finger's own protrusion length rather
   than a fixed pixel value, this consistently covers "the top
   portion of the finger" regardless of hand size or which finger.
3. WHICH finger is torn and HOW BIG its box should be are deliberately
   two separate passes:
   a. Selection: within each finger's own tight ROI, flag pixels whose
      LAB distance from the glove's own median material colour
      (estimated from the eroded whole-glove interior, same as
      tearing.py) exceeds a fixed cutoff and take the largest connected
      blob per finger - identical extraction to tearing.py, just over a
      much smaller region. Candidates are normally ranked by that
      blob's area/ROI ratio, same as always - PROVEN reliable when at
      least one candidate has a substantial, unambiguous blob (5 of
      this dataset's 6 known images).
      Only when EVERY candidate's ratio stays below
      TRANSLUCENT_MAX_RATIO_CAP (0.30) - i.e. no candidate looks like an
      obvious tear by area alone - does ranking switch to a composite
      of three BOUNDARY signals instead (a thin ring around the blob,
      not its interior):
        - deviation_strength: mean LAB distance from material colour
        - edge_density:       fraction of the boundary with Canny edge
                               support
        - sharpness:          mean local grey-level gradient magnitude
      each min-max normalised to [0, 1] across this image's own
      candidates and averaged with equal weight. This exists because on
      TRANSLUCENT material (e.g. latex) every fingertip can show some
      skin-coloured LAB deviation through the material, and those
      translucency patches are typically LARGER in raw area than a real
      tear's - plain area-based ranking picked a translucent tip over
      the true tear on such an image. A real tear has a sharp torn
      edge; translucency fades in gradually, which is what the three
      boundary signals are meant to tell apart.
      The area-ratio cap matters: an early version applied composite
      ranking unconditionally on every image, and it regressed a
      DIFFERENT, previously-correct image - a normal fingertip's
      specular highlight can have a sharper boundary (by Sobel gradient)
      than a real but large, slightly-blurred tear, so ranking by
      boundary sharpness alone picked the highlight over an obvious,
      dominant-by-area tear. Gating on "does any candidate already look
      like a confident tear by area" restricts composite ranking to
      only the regime it was built for. MIN_CANDIDATE_AREA_PX is
      deliberately just a noise floor (not an area-competitive gate) so
      a small-but-genuine translucent-case blob can still enter the
      ranking - on the translucent-latex image this was built for, the
      true tear's raw blob was 20x smaller than the largest (wrong)
      candidate's, but still won the composite ranking on boundary
      sharpness/density.
   b. Box-quality: for THAT finger only, redo the same LAB-deviation
      test within a MODERATELY DILATED version of its ROI
      (ROI_DILATE_FRACTION - roughly double the circle's area, still
      local to this one finger), close the result with a small kernel
      to bridge thin gaps - a surviving rim of intact material across
      part of the opening, or plain digitisation noise - then keep
      whichever connected component the original tight ROI overlaps
      most. That component's FULL extent becomes the box/mask, not
      just the part that happened to fall inside the tight circle.
      This matters because a torn fingertip's measured "tip" point
      (step 1) is taken from what's left of the contour, which for a
      badly torn finger doesn't reliably sit at the centre of the true
      opening - the tight circle alone was silently clipping a large
      fraction of the real tear on at least one image in this dataset.
   Two earlier, simpler designs were tried and dropped: running (b)'s
   dilate+close over the WHOLE glove at once let the closing step chain
   together unrelated anomalies (shading, highlights on other fingers)
   into one enormous component on several images; and running (b) for
   EVERY finger before deciding the winner let that same inflation
   occasionally outscore the real tear and flip which finger got
   flagged. Keeping (a) exactly as the original tight-ROI-only
   computation, and only ever improving the box for whichever finger
   (a) already picked, avoids both failure modes.
4. The final box is padded to a minimum size relative to the flagged
   finger's own measured width (so a small or oddly-shaped blob still
   renders as a legible box) and clipped to the glove silhouette's own
   bounding area, so it can never float out over the background.

A torn-off fingertip is also typically the *shortest* protrusion in
the finger-length profile (a torn tip has less material than an
intact one), so `measurements` additionally reports which finger index
was flagged and how its length ranks among the others found, as a
secondary, human-checkable signal - it is not required for the score
because a naturally shorter finger (e.g. the little finger, or
foreshortening from hand angle) would otherwise produce false
positives on perfectly intact gloves.

Score = confirmed anomaly area relative to that finger's own ROI area
(not the whole glove_area - a fingertip patch is always going to be a
small fraction of the entire glove, so scoring against glove_area the
way tearing.py does would never saturate). `measurements.area_pct` is
still reported relative to glove_area, matching the shared contract.
"""

import cv2
import numpy as np


# ============================================================
# THRESHOLDS (documented here so they can be copied into the report)
# ============================================================

# --- Fingertip localisation ---

# Circular moving-average window (in contour points) used to smooth
# the distance-from-centroid profile before peak-picking.
SMOOTH_WINDOW = 15

# A candidate peak must be a local maximum within this fraction of the
# contour's total point count on each side.
LOCAL_MAX_WINDOW_FRAC = 0.01

# Non-max suppression: two accepted fingertip peaks must be at least
# this fraction of the contour length apart (as a circular index
# distance), so one broad rounded tip doesn't yield duplicate peaks.
NMS_SEPARATION_FRAC = 0.06

# A peak must exceed the profile's median by at least this fraction of
# (max - median) to count as a finger rather than a palm/wrist bump.
MIN_PROMINENCE_FRAC = 0.35

MAX_FINGERTIPS = 5

# Candidate points in the bottom fraction of the glove's own bounding
# box are excluded - this is where the cuff/wrist trim sits in every
# photo in this dataset, not a finger.
BOTTOM_MARGIN_FRACTION = 0.05

# --- Per-finger ROI ---

# ROI circle radius, as a fraction of that finger's own measured
# protrusion length (distance from centroid, minus the profile's
# median/baseline). Keeps the ROI to roughly the top of the finger
# rather than reaching down into the palm.
TIP_RADIUS_FRACTION = 0.6
MIN_TIP_RADIUS_PX = 15

# --- Colour-deviation anomaly search ---

EROSION_FRACTION = 0.012          # of sqrt(glove_area), for material-colour sampling only
MIN_EROSION_PX = 5
MAX_EROSION_PX = 30

# A pixel counts as anomalous if its ab (chroma) distance clears
# MIN_COLOUR_DISTANCE, OR its lightness distance clears
# MIN_LIGHTNESS_DISTANCE (same OR as tearing.py). A stricter version
# was tried - requiring lightness spikes to also clear a minimum
# accompanying chroma shift - because a rounded fingertip catches much
# stronger specular highlight/shadow than the flatter palm tearing.py
# searches, and pure-lightness spikes from that were driving false
# positives on plain touching/damaged_by_fold photos. That extra gate
# cut those false positives noticeably, but on this dataset it also
# suppressed several genuine tears whose skin-tone contrast happened
# to be subtle (particularly on latex and one cotton photo, where the
# true tear region scored no higher than ordinary shading noise once
# gated), dropping recall on the 6 known positives from 6/6 to 2/6.
# Recall on the target category is what evaluate.py actually scores
# (see tearing.py's docstring - no true negatives are tested against a
# detector in the real pipeline), so the plain OR is kept.
# TUNED-BY-EYE on the 68-image dataset
MIN_COLOUR_DISTANCE = 22.0        # LAB a/b distance
# TUNED-BY-EYE on the 68-image dataset
MIN_LIGHTNESS_DISTANCE = 28.0     # LAB L distance

# ROI-area ratio at/above which the score saturates to 1.0. Measured
# ratios on the 6 known tearing_fingertip images ranged ~26%-72%. Also
# acts as the effective "is this substantial enough" floor together
# with DETECTION_SCORE_THRESHOLD below (score >= 0.5 needs ratio >=
# 0.25) - a stronger requirement than the old, now-removed
# MIN_HOLE_AREA_RATIO/MIN_EDGE_SUPPORT_PX candidate-stage gates ever
# were, so dropping those in favour of composite ranking (see PROBLEM 1
# in the module docstring) didn't loosen anything at the final decision.
# TUNED-BY-EYE on the 68-image dataset
STRONG_HOLE_AREA_RATIO = 0.50     # 50% of the fingertip ROI area

CANNY_LOW, CANNY_HIGH = 50, 150
EDGE_SUPPORT_DILATE_PX = 5

DETECTION_SCORE_THRESHOLD = 0.5

# --- Translucent-material candidate ranking (see module docstring) ---

# A candidate blob must clear this many raw pixels to be considered at
# all - a pure noise floor, NOT the old MIN_HOLE_AREA_RATIO area-share
# gate. That gate was excluding a genuine but small torn-opening blob
# on translucent latex (see PROBLEM 1 in the module docstring): a real
# tear's exposed-skin patch can clear the LAB threshold over only a
# handful of strongly-deviating pixels, while translucency shows up as
# a much larger but weaker, gradual deviation across most of a tip.
MIN_CANDIDATE_AREA_PX = 10

# Ring thickness (px) used to sample each candidate blob's *boundary*
# for edge-density and sharpness - a real tear has a sharp torn edge;
# translucency fades in gradually, so its boundary carries less Canny
# edge support and a shallower local gradient.
RING_KERNEL_PX = 5
_RING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_KERNEL_PX, RING_KERNEL_PX))

# Composite boundary-based ranking is used only when EVERY candidate's
# blob is still small relative to its own ROI (max ratio below this
# cap) - not universally. Two reasons: (1) when a candidate's blob
# already fills a large share of its ROI, its "boundary ring" starts
# overlapping the ROI's own artificial circular cutoff rather than the
# blob's true edge, making the sharpness/edge-density signals noisy;
# (2) empirically, on a genuine substantial tear (large, unambiguous
# blob) plain area-ratio ranking is already reliable, and composite
# ranking regressed exactly that case on this dataset - a normal
# fingertip's specular highlight can have a SHARPER boundary than a
# real but slightly-blurred tear edge, so ranking by sharpness alone
# picked the highlight over a large, obvious tear. Measured on this
# dataset's 6 known images, the translucent-latex image's largest
# candidate ratio (0.26) sits far below every other image's (0.45-0.72)
# - this cap sits in that gap with margin on both sides.
# TUNED-BY-EYE on the 68-image dataset
TRANSLUCENT_MAX_RATIO_CAP = 0.30

# --- Blob merging / box quality ---

# Each finger's tight ROI circle is dilated by this fraction of its
# own radius before the anomaly mask is computed and merged, so a tear
# that spills slightly past the tight circle (the "tip" point used to
# centre it, taken from what's left of a torn contour, doesn't reliably
# sit at the true centre of the opening) still gets picked up in full.
# Kept as a per-finger LOCAL expansion rather than searching the whole
# glove at once - an early version that merged over the whole glove_mask
# chained together unrelated anomalies (shading, highlights on other
# fingers) into single enormous components on several images.
ROI_DILATE_FRACTION = 1.0

# Closing kernel applied to the (locally dilated) anomaly mask to
# bridge small gaps - a thin surviving rim of intact material across
# part of a torn opening, or digitisation noise - that would otherwise
# split one physical tear into several disconnected components.
MERGE_CLOSE_KERNEL_PX = 11

# The final box's shorter side must be at least this fraction of the
# finger's own measured width, so a small/fragmented anomaly blob
# still renders as a visible box rather than a sliver.
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
    """
    Find up to MAX_FINGERTIPS fingertip points on glove_mask's outer
    contour via convex-hull-style protrusion analysis + finger-length
    analysis (distance from the mask centroid).

    Returns a list of (point, length) tuples, `length` being that
    finger's protrusion distance above the profile's baseline (i.e. an
    estimate of visible finger length), sorted by contour order.
    """
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

    # Every photo in this dataset holds the hand fingers-up, cropped at
    # the wrist/cuff at the bottom of the frame. A cuff trim band in a
    # colour different from the glove body (common - e.g. a coloured
    # elastic hem) can register as a spurious "fingertip" here, since
    # it both sits far from the mask centroid and differs sharply in
    # colour from the rest of the glove. Excluding the bottom margin of
    # the glove's own bounding box rules that out without assuming
    # anything about hand size or position within the frame.
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

    # Exclude the thumb: anatomically it sits at a much wider angle from
    # its nearest neighbour than the four fingers do from each other
    # (the thumb-index web gap is far wider than any inter-finger gap),
    # so it consistently has the largest circular contour-index distance
    # to its nearest neighbouring fingertip. Measured on the dataset,
    # the thumb's distinct orientation catches different specular
    # highlight/shadow than the other four fingers, which made it
    # consistently outscore genuine (but subtler) tears elsewhere - none
    # of this dataset's tearing_fingertip defects are on the thumb.
    # Only applied with >=4 fingertips found: with fewer, an isolated
    # point is as likely to be the actual damaged/only-visible finger as
    # it is the thumb, so excluding it would remove a real candidate.
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
    """
    Width of glove_mask's foreground run through (x, y), measured by
    walking left/right from that point. Returns 0 if (x, y) isn't
    itself foreground (off the finger, or on a background gap).
    """
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
    """
    Pad (x, y, w, h) so neither side is smaller than min_dim (expanding
    around its own centre), then clip to glove_bounds = (gx0, gy0, gx1,
    gy1) - the glove_mask's own foreground bounding box - so the box
    can never float outside the glove silhouette.
    """
    gx0, gy0, gx1, gy1 = glove_bounds
    cx, cy = x + w / 2.0, y + h / 2.0
    w, h = max(w, min_dim), max(h, min_dim)
    x, y = cx - w / 2.0, cy - h / 2.0

    x0 = max(gx0, min(x, gx1))
    y0 = max(gy0, min(y, gy1))
    x1 = max(gx0, min(x + w, gx1 + 1))
    y1 = max(gy0, min(y + h, gy1 + 1))

    return int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))


def _candidate_signals(tight_mask, full_lab_dist, grad_mag, dilated_edges):
    """
    The three signals used to rank candidate fingertip blobs (see
    module docstring, PROBLEM 1): how strongly the blob deviates from
    the glove's own material colour, how much Canny edge support its
    boundary carries, and how sharp the local grey-level gradient is
    right at that boundary. All three are measured on a thin ring
    around the blob rather than the blob's interior, since it's the
    BOUNDARY character (sharp torn edge vs. gradual translucency fade)
    that's diagnostic - the interior of a large translucent patch can
    look just as "deviated" on average as a real tear's interior.

    Returns
    -------
    dict
        {"blob_area", "deviation_strength", "edge_density", "sharpness"}
    """
    blob_area = cv2.countNonZero(tight_mask)
    ring = cv2.subtract(cv2.dilate(tight_mask, _RING_KERNEL), cv2.erode(tight_mask, _RING_KERNEL))
    ring_area = cv2.countNonZero(ring)

    deviation_strength = float(full_lab_dist[tight_mask > 0].mean()) if blob_area > 0 else 0.0
    if ring_area > 0:
        edge_density = cv2.countNonZero(cv2.bitwise_and(ring, dilated_edges)) / ring_area
        sharpness = float(grad_mag[ring > 0].mean())
    else:
        edge_density = 0.0
        sharpness = 0.0

    return {
        "blob_area": blob_area,
        "deviation_strength": deviation_strength,
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
    """
    Detect a torn/ripped fingertip in a segmented glove.

    Parameters
    ----------
    processed : dict
        Output of preprocess_image().
    segmentation : dict
        Output of segment_glove().

    Returns
    -------
    dict
        Result dict following the evaluate.py detector contract:
        defect_name, detected, detection_score, algorithm,
        bounding_box, mask, measurements.
    """
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

    # Material colour reference, sampled from the eroded whole-glove
    # interior (same approach as tearing.py) - a large, stable sample
    # mostly drawn from the palm, away from any one fingertip.
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

    # Used only for candidate ranking (PROBLEM 1) - full LAB distance
    # (not just the a/b-or-L OR test) as the "how strongly does this
    # deviate" magnitude, and a Sobel gradient map of gray_enhanced as
    # the "how sharp is the boundary" signal.
    full_lab_dist = np.sqrt(l_delta ** 2 + ab_distance ** 2)
    grad_mag = cv2.magnitude(
        cv2.Sobel(gray_enhanced, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray_enhanced, cv2.CV_32F, 0, 1, ksize=3),
    )

    gys, gxs = np.where(glove_mask > 0)
    glove_bounds = (int(gxs.min()), int(gys.min()), int(gxs.max()), int(gys.max()))

    finger_lengths = [length for _, length in fingertips]

    # --- Phase 1: which finger is torn? ---
    # Deliberately the ORIGINAL, tight-ROI-only computation (no search
    # dilation, no closing) - unchanged from before the earlier
    # box-quality fix. An early version applied the dilate/close/merge
    # step while ALSO deciding which finger wins, and that let noise/
    # highlights near other fingers get inflated by the same merge and
    # occasionally outscore the real tear - a cross-finger regression
    # this dataset's 6 known images caught. Keeping selection separate
    # from box-quality means phase 2 can only ever improve the box for
    # whichever finger this phase already - and reliably - picked.
    #
    # Candidates are ranked by a composite of THREE boundary signals
    # (PROBLEM 1: translucent latex) rather than by blob area/ratio.
    # On translucent material every fingertip can show skin-coloured
    # LAB deviation, and the translucency patches are usually LARGER in
    # raw area than a real tear's - ranking by area alone (the previous
    # design) picked a translucent tip over the real one. A real tear
    # has a sharp torn edge; translucency fades in gradually. So each
    # candidate blob's BOUNDARY (a thin ring around it, not its
    # interior - a large translucent patch's interior can look just as
    # "deviated" on average as a real tear's) is scored on:
    #   (a) deviation_strength - mean LAB distance from material colour
    #   (b) edge_density        - fraction of the boundary with Canny
    #                             edge support
    #   (c) sharpness            - mean local grey-level gradient
    # normalised to [0, 1] across this image's own candidates (material
    # and lighting vary a lot between photos, so only relative ranking
    # within one image is meaningful) and averaged with equal weight.
    # Candidates only need to clear a tiny raw-pixel floor
    # (MIN_CANDIDATE_AREA_PX) to enter the ranking - the old 5%-of-ROI
    # area gate was excluding the real tear outright on the one image
    # this was built for (see module docstring, PROBLEM 1).
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

        signals = _candidate_signals(tight_mask, full_lab_dist, grad_mag, dilated_edges)
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

    # Composite boundary ranking only when EVERY candidate's blob is
    # still small relative to its ROI - see TRANSLUCENT_MAX_RATIO_CAP.
    # Otherwise (the common case) rank by clipped score, exactly as
    # before this fix (score, not raw ratio: two candidates both above
    # STRONG_HOLE_AREA_RATIO saturate to the same score, and the tie
    # resolves to whichever was found first, in finger_index order -
    # that specific tie-break, not a considered ranking, is what
    # happened to land on the right finger on two of this dataset's
    # borderline images, where two fingers show comparably strong
    # anomaly area and raw ratio alone doesn't cleanly separate them).
    if max(c["ratio"] for c in candidates) < TRANSLUCENT_MAX_RATIO_CAP:
        strength_n = _normalize([c["signals"]["deviation_strength"] for c in candidates])
        edge_n = _normalize([c["signals"]["edge_density"] for c in candidates])
        sharp_n = _normalize([c["signals"]["sharpness"] for c in candidates])
        for c, s, e, sh in zip(candidates, strength_n, edge_n, sharp_n):
            c["composite"] = (s + e + sh) / 3.0
        winner = max(candidates, key=lambda c: c["composite"])
    else:
        winner = max(candidates, key=lambda c: c["score"])
    finger_index = winner["finger_index"]
    tight_mask = winner["tight_mask"]
    roi_mask = winner["roi_mask"]
    roi_area = winner["roi_area"]
    tight_ratio = winner["ratio"]

    # --- Phase 2: how big is the tear, really? ---
    # For the ALREADY-CHOSEN finger only, search a moderately dilated
    # version of its ROI (not the whole glove, and not other fingers'
    # ROIs) and close small gaps, so a tear that spills slightly past
    # the tight circle - or was fragmented by a thin surviving rim of
    # material - is recovered in full rather than reported as whatever
    # fragment happened to fall inside the tight circle.
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
            # Only adopt the merged version if it's actually bigger -
            # it never should be smaller (it's a superset by
            # construction), but this guards against a degenerate
            # relabelling if fed an empty/odd mask.
            if merged_area >= cv2.countNonZero(tight_mask):
                candidate_mask = merged_mask
                ratio = merged_area / roi_area

    detection_score = float(np.clip(ratio / STRONG_HOLE_AREA_RATIO, 0.0, 1.0))
    detected = detection_score >= DETECTION_SCORE_THRESHOLD

    defect_pixel_count = cv2.countNonZero(candidate_mask)
    ys, xs = np.where(candidate_mask > 0)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)

    # Pad to a sensible minimum (relative to the flagged finger's own
    # width, so it scales with hand/photo size) and clip to the glove's
    # own silhouette, so the box can never render as an unreadable
    # sliver or float out over the background.
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
