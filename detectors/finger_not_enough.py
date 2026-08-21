"""
detectors/finger_not_enough.py
--------------------------------
Classical OpenCV detector for "finger_not_enough" defects (a
knitting/moulding defect where two adjacent fingers are fused together,
a finger fails to form as its own lobe, or a finger is present as its
own lobe but noticeably shorter than its neighbours). No ML.

Approach
--------
This started as a convexity-defect valley detector (find the notches
between fingers, count them). That signal has a structural blind spot:
a fused or missing finger erases exactly the valley that would be
needed to bound its own slot, so any localisation built on top of
valleys never had a boundary to work with on the images it needed to
localise. Verified on this dataset - none of the 5 real
finger_not_enough images ever produced the 4 valleys needed.

This version localises via fingertip PEAKS instead, reusing the
distance-from-centroid peak finder proven in tearing_fingertip.py
(a peak is a local maximum in each contour point's distance from the
glove's own centroid - robust to a short finger, since even a stubby
finger still typically sticks out further than the surrounding palm
edge; a valley between two normal fingers does not need to exist for
this to find the fingers themselves):

1. Locate up to 5 fingertip peaks via that distance-from-centroid
   profile, at a lower prominence cutoff than tearing_fingertip.py
   uses (this detector explicitly wants short/stubby fingers to still
   register, not just fully-extended ones).

DESIGN PRINCIPLE: a glove always has exactly 5 fingers - thumb + 4
main fingers (index, middle, ring, pinky). That's a hard prior, not
something inferred per image: fewer than 4 confirmed MAIN fingers is
always a defect, and the only open question is where. Making that
prior actually usable requires confirming the thumb reliably, because
the main peak-finder above frequently doesn't find it at all (it sits
closer to the palm than the four extended fingers, so its centroid-
distance protrusion is often below the prominence bar) - so a second,
dedicated mechanism (_identify_thumb) is used purely to place the
thumb, using the fact that on every image in this dataset it sits at
the leftmost anatomical position:
   a. If the leftmost found peak is dramatically shorter than the
      median of the others (THUMB_LENGTH_RATIO), it's confirmed thumb
      and removed from the main-finger pool.
   b. Otherwise, search the glove mask's own column-top-height profile
      (not the smoothed centroid-distance profile the main peak-finder
      uses) between the glove's left edge and the leftmost peak for a
      genuine BULGE - a point that rises higher than its surroundings
      on both sides (_find_bulge). This catches a thumb that never
      cleared the main peak-finder's bar at all: real material is
      still there, geometrically distinct from a monotonic taper down
      to the palm baseline, which is what genuinely empty space looks
      like. A confirmed bulge marks the thumb's position (used only to
      anchor gap boundaries) without being added as a "peak".
An earlier version only ever attempted this when 5 peaks were already
found (an "extra" one that must be the thumb); gating it there instead
of always running it was itself the bug behind FAILURE 1 below - with
only 4 peaks, a thumb hiding among them (rather than a genuinely
complete 4-main-finger hand) was silently read as "complete".

2. main_peaks = peaks with the confirmed thumb (if any) removed. Fewer
   than 4 main_peaks means a main finger didn't register as its own
   protrusion - fused into a neighbour, or missing outright.
   Localising it (_localize_missing_main_finger) has to distinguish
   two different shapes of evidence:
     - INTERIOR: one gap between two found main fingers is a clear,
       isolated standout relative to the OTHER interior gaps
       (STRONG_INTERIOR_MARGIN) - unambiguous, trust it directly.
     - EDGE: a missing pinky or index doesn't widen any interior gap
       at all (FAILURE 2b) - it just shortens the whole finger band,
       so no single gap stands out. When interior isn't a strong
       standout, each edge (left, beyond a confirmed thumb if any;
       right, out to the glove's own silhouette) is checked instead:
       it only counts as "missing finger here" if a bulge search finds
       NO real material there (genuinely empty, not just a confirmed
       thumb the earlier step already accounted for - FAILURE 2a) and
       the empty space is wide enough to plausibly hold a whole finger
       (EDGE_MIN_RATIO_FOR_CANDIDACY). If no edge qualifies either,
       fall back to the (non-standout) interior gap anyway - the best
       signal still available.

3. bounding_box for the located gap is the middle 60% of its x-range
   (trimmed inward so it doesn't overlap the fingers on either side),
   from the median fingertip height of the fingers bounding THAT gap
   specifically down to the valley/webbing between them - the MEDIAN
   column-top height across the trimmed x-range, not the single
   shallowest point: a lone bump or partially-collapsed stub partway
   across a wide gap previously dominated that minimum and collapsed
   the box to a sliver even though most of the gap sat much lower.
   Height is additionally floored at MIN_GAP_BOX_HEIGHT_FRACTION of the
   found fingers' own median length, so a gap whose local surface still
   happens to sit unusually high renders a legible box regardless.

4. detection_score blends how abnormal the worst gap's width is with a
   floor based on how many main fingers are missing by count (mirroring
   the previous version's fixed 0.75/1.0 tiers, since a very short
   median spacing sample - only 2-3 main peaks found - can make the gap
   ratio alone noisy).

5. If all 4 main fingers ARE found, there's no missing slot to
   localise - but one of the four could still be present and simply
   too SHORT (e.g. a finger that formed but didn't get enough
   material). That's checked as a secondary signal using the original
   valley-baseline-to-peak slot method: it only ever runs here, when
   peak-counting has already confirmed a normal 4-main-finger topology,
   so it isn't fighting the same fused/missing blind spot that
   motivated dropping it as the primary signal.

Fallback
--------
Whole-glove bounding box is used only when fewer than 2 peaks are
found at all - there's no pair of points to compute any spacing from,
so no gap can be meaningfully localised.
"""

import cv2
import numpy as np


# ============================================================
# THRESHOLDS (documented here so they can be copied into the report)
# ============================================================

# --- Fingertip peak localisation (distance-from-centroid profile) ---

# Circular moving-average window (in contour points) used to smooth
# the distance-from-centroid profile before peak-picking.
PEAK_SMOOTH_WINDOW = 15

# A candidate peak must be a local maximum within this fraction of the
# contour's total point count on each side.
PEAK_LOCAL_MAX_WINDOW_FRAC = 0.01

# Non-max suppression: two accepted peaks must be at least this
# fraction of the contour length apart (circular index distance).
# tearing_fingertip.py uses 0.06 at a much higher prominence cutoff
# (0.35), where few enough candidates survive that this rarely
# matters. At this detector's much lower prominence, two genuinely
# separate adjacent fingertips commonly sit only ~0.055-0.06 of the
# contour apart in index terms (tall fingers mean many contour points
# per finger, independent of how close the tips are in x) - 0.06 was
# merging them into one. 0.03 still merges true double-bumps on a
# single tip without merging distinct fingers.
PEAK_NMS_SEPARATION_FRAC = 0.03

# A peak must exceed the profile's median by at least this fraction of
# (max - median). Deliberately much lower than tearing_fingertip.py's
# 0.35 - that detector only wants fully-extended fingers, this one
# specifically needs a short/stubby finger to still register as its
# own peak rather than being smoothed into the palm.
PEAK_MIN_PROMINENCE_FRAC = 0.12

MAX_FINGERTIPS = 5

# Candidate points in the bottom fraction of the glove's own bounding
# box are excluded - the cuff/wrist trim, not a finger.
PEAK_BOTTOM_MARGIN_FRACTION = 0.05

# --- Hard prior: 4 main fingers + 1 thumb (see module docstring) ---

EXPECTED_MAIN_FINGERS = 4

# The leftmost peak is confirmed thumb-by-length when its own length
# is below this fraction of the median of the OTHER peaks' lengths.
# 0.4 was chosen from this dataset's known cases: a confirmed thumb's
# ratio was 0.16-0.28 across every case it applied to, while a
# genuinely-present main finger that's merely shorter than its
# neighbours (not the thumb) never dropped below ~0.4 while also being
# the leftmost peak.
# TUNED-BY-EYE on the 68-image dataset
THUMB_LENGTH_RATIO = 0.4

# _find_bulge's prominence (px, in this dataset's resized-to-1000px-
# longest-side frame) must clear this to count as "real material" -
# confirming a thumb the main peak-finder missed entirely, or ruling
# out an edge as a missing-finger location. Calibrated so a confirmed
# thumb bulge (15-109px prominence across this dataset's cases) still
# passes, while background/mask noise (<5px) does not.
# TUNED-BY-EYE on the 68-image dataset
BULGE_MIN_PROMINENCE_PX = 20

# An edge gap only counts as a plausible missing-finger location if
# it's at least this fraction of the median interior gap wide -
# otherwise it's just the normal small margin between the outermost
# finger and the glove's own silhouette edge, not empty finger-sized
# space.
EDGE_MIN_RATIO_FOR_CANDIDACY = 0.7

# The widest interior gap is trusted directly (skipping edge checks
# entirely) once it's at least this many times the narrowest OTHER
# interior gap - a genuinely fused/missing interior finger showed a
# 2.3-3.3x margin on this dataset's known cases, well clear of the
# ~1.2-1.3x margin that's just normal finger-spacing variation.
# TUNED-BY-EYE on the 68-image dataset
STRONG_INTERIOR_MARGIN = 2.0

# Same idea, for the case where there's only one interior gap to judge
# (so no "margin over the others" can be computed) - an absolute ratio
# bar instead.
# TUNED-BY-EYE on the 68-image dataset
STRONG_INTERIOR_SINGLE_RATIO = 1.3

# --- Gap-based localisation of a missing/fused finger ---

# A gap (between two adjacent peaks, or from an outermost peak to the
# glove's own bounding-box edge) counts as "contains a missing finger"
# once it's this many times wider than the median gap.
GAP_RATIO_THRESHOLD = 1.6

# Gap ratio at/above which detection_score saturates to 1.0.
GAP_SEVERE_RATIO = 2.4

# The reported box is the middle fraction of the gap's x-range, i.e.
# GAP_TRIM_FRAC is trimmed off *each* side so the box doesn't overlap
# the genuine fingers flanking it.
GAP_TRIM_FRAC = 0.2

# The gap box's height is floored at this fraction of the found
# fingers' own median length, so a gap whose local surface happens to
# sit unusually high (see _gap_bounding_box) still renders a legible
# box instead of a sliver.
MIN_GAP_BOX_HEIGHT_FRACTION = 0.6

DETECTION_SCORE_ONE_MISSING = 0.75   # exactly 4 of 5 fingers found
DETECTION_SCORE_SEVERE = 1.0         # 3 or fewer fingers found

# --- Secondary signal: a present-but-short finger (5 peaks found) ---

MIN_VALLEY_DEPTH_RATIO = 0.10
MAX_VALLEY_ANGLE_DEG = 115.0
MAX_VALLEY_DEPTH_Y_FRACTION = 0.75
NMS_SEPARATION_FRAC = 0.03
SHORT_FINGER_RATIO_THRESHOLD = 0.45
MIN_VALLEYS_FOR_LOCALIZATION = 4

DETECTION_SCORE_THRESHOLD = 0.5

ALGORITHM = (
    "Fingertip localisation via contour distance-from-centroid peaks "
    "against a hard prior of 4 main fingers + 1 thumb; the thumb is "
    "identified separately (by length ratio, or a column-top-height "
    "bulge search) so it never masks a genuinely missing main finger; "
    "when fewer than 4 main fingers are found, the missing one is "
    "localised to an abnormally wide interior gap when one clearly "
    "stands out, else to whichever edge is both wide enough and free "
    "of real material (no bulge) - catching an edge-missing finger "
    "that widens no interior gap at all; when all 4 main fingers are "
    "found, falls back to convexity-defect valley/slot analysis to "
    "catch a finger that's present but short of its neighbours' length"
)


def _empty_result():
    return {
        "defect_name": "finger_not_enough",
        "detected": False,
        "detection_score": 0.0,
        "algorithm": ALGORITHM,
        "bounding_box": None,
        "mask": None,
        "measurements": {},
    }


def _largest_contour(glove_mask):
    """Returns the largest contour in glove_mask, or None if unusable."""
    contours, _ = cv2.findContours(glove_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4 or cv2.contourArea(contour) <= 0:
        return None

    return contour


# ============================================================
# FINGERTIP PEAK LOCALISATION
# ============================================================

def _circular_smooth(values, window):
    n = len(values)
    if window <= 1 or n == 0:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.concatenate([values[-window:], values, values[:window]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[window:window + n]


def locate_peaks(glove_mask):
    """
    Find up to MAX_FINGERTIPS fingertip peaks via convex-hull-style
    protrusion analysis (distance from the mask centroid) - same
    technique as tearing_fingertip.py's _locate_fingertips, at a lower
    prominence cutoff.

    Purely a peak-finder - thumb identification is a separate, dedicated
    step (_identify_thumb) that runs regardless of how many peaks this
    finds, using a different signal (the mask's own column-top-height
    profile, not this smoothed centroid-distance one) precisely because
    the thumb so often doesn't clear THIS function's prominence bar at
    all (see module docstring).

    Returns
    -------
    list of (x, y, length)
        Every found fingertip, `length` being that point's protrusion
        above the profile's median (how far it reaches beyond the palm
        baseline), sorted left-to-right by x. Empty if the mask has no
        usable contour.
    """
    contours, _ = cv2.findContours(glove_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    n = len(contour)
    if n < 20:
        return []

    # cv2.findContours starts the array at an arbitrary point. Left
    # alone, that seam can fall between two real fingertips - the
    # circular NMS below then sees them as index-adjacent (artificially
    # "close together") even though they're spatially far apart, and
    # drops one. Re-rooting the array at the bottom-most (deepest-y)
    # point instead guarantees the seam sits in the wrist/cuff region -
    # always excluded from candidates by min_valid_y below - so it can
    # never land between two fingers.
    contour = np.roll(contour, -int(np.argmax(contour[:, 1])), axis=0)

    moments = cv2.moments(glove_mask)
    if moments["m00"] == 0:
        return []
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    dists = np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)
    smoothed = _circular_smooth(dists, PEAK_SMOOTH_WINDOW)

    median_d = float(np.median(smoothed))
    max_d = float(smoothed.max())
    prominence_threshold = median_d + PEAK_MIN_PROMINENCE_FRAC * (max_d - median_d)

    _, bbox_y, _, bbox_h = cv2.boundingRect(contour)
    min_valid_y = bbox_y + (1.0 - PEAK_BOTTOM_MARGIN_FRACTION) * bbox_h

    half_win = max(3, int(n * PEAK_LOCAL_MAX_WINDOW_FRAC))
    candidates = []
    for i in range(n):
        if contour[i][1] > min_valid_y:
            continue
        idxs = np.arange(i - half_win, i + half_win + 1) % n
        if smoothed[i] >= smoothed[idxs].max() and smoothed[i] > prominence_threshold:
            candidates.append(i)
    candidates.sort(key=lambda i: smoothed[i], reverse=True)

    min_sep = int(n * PEAK_NMS_SEPARATION_FRAC)
    selected = []
    for i in candidates:
        if all(min(abs(i - j), n - abs(i - j)) > min_sep for j in selected):
            selected.append(i)
        if len(selected) >= MAX_FINGERTIPS:
            break

    def _to_point(i):
        return (int(contour[i][0]), int(contour[i][1]), float(smoothed[i] - median_d))

    selected.sort(key=lambda i: contour[i][0])
    return [_to_point(i) for i in selected]


# ============================================================
# THUMB IDENTIFICATION (see module docstring, DESIGN PRINCIPLE)
# ============================================================

def _find_bulge(glove_mask, x_left, x_right, step=4):
    """
    Scan glove_mask's column-top height (topmost foreground y at each
    x) across [x_left, x_right] for the most prominent LOCAL MINIMUM in
    y - a point that rises higher than the profile on both sides of it,
    i.e. real material bulging up from an otherwise lower/tapering
    silhouette. Distinguishes a thumb (or any other digit) too weak to
    clear the main peak-finder's bar from genuinely empty space, which
    instead shows a monotonic taper with no interior local minimum.

    Returns
    -------
    tuple(int, int, float) or None
        (x, y, prominence) of the most prominent bulge, or None if the
        region is too narrow to judge or has no interior local minimum.
    """
    h, w = glove_mask.shape[:2]
    xs, ys = [], []
    for x in range(max(0, int(x_left)), min(w, int(x_right) + 1), step):
        col = np.nonzero(glove_mask[:, x])[0]
        if col.size:
            xs.append(x)
            ys.append(int(col[0]))
    if len(ys) < 5:
        return None

    best = None
    for i in range(1, len(ys) - 1):
        prominence = min(max(ys[:i]), max(ys[i:])) - ys[i]
        if prominence > 0 and (best is None or prominence > best[2]):
            best = (xs[i], ys[i], float(prominence))
    return best


def _identify_thumb(peaks, glove_mask, bx):
    """
    Identify the thumb using the hard prior that it's always at the
    leftmost anatomical position (see module docstring) - confirmed
    two independent ways, tried in order:

    1. The leftmost found peak is dramatically shorter than the median
       of the other peaks (THUMB_LENGTH_RATIO) - it cleared the main
       peak-finder's bar, but the thumb sits closer to the palm than
       the four extended fingers so it's still much shorter than them.
    2. Otherwise, search the region between the glove's own left edge
       and the leftmost peak for a genuine bulge (_find_bulge) - real
       material the main peak-finder's smoothed profile missed
       entirely.

    Returns
    -------
    tuple(float or None, list)
        (thumb_x, main_peaks). thumb_x is the thumb's x-coordinate if
        confirmed by either method, else None. main_peaks is `peaks`
        with a length-confirmed thumb removed (a bulge-only thumb was
        never one of the registered peaks, so nothing is removed for
        that case - thumb_x alone is enough to anchor gap boundaries).
    """
    if not peaks:
        return None, []

    ordered = sorted(peaks, key=lambda p: p[0])
    leftmost = ordered[0]
    others = ordered[1:]

    if others:
        median_other_len = float(np.median([p[2] for p in others]))
        if median_other_len > 1e-6 and leftmost[2] / median_other_len < THUMB_LENGTH_RATIO:
            return float(leftmost[0]), others

    bulge = _find_bulge(glove_mask, bx, leftmost[0])
    if bulge is not None and bulge[2] >= BULGE_MIN_PROMINENCE_PX:
        return float(bulge[0]), list(ordered)

    return None, list(ordered)


# ============================================================
# GAP-BASED LOCALISATION (missing/fused finger)
# ============================================================

def _localize_missing_main_finger(main_peaks, thumb_x, glove_mask, bx, bw):
    """
    Locate the one missing/fused main finger, given fewer than
    EXPECTED_MAIN_FINGERS main_peaks were found (see module docstring,
    point 2). Tries, in order:

    1. INTERIOR standout: the widest gap between two adjacent main
       peaks is a clear, isolated outlier relative to the other
       interior gaps (STRONG_INTERIOR_MARGIN), or - when there's only
       one interior gap to compare it against - clears an absolute bar
       (STRONG_INTERIOR_SINGLE_RATIO). This is FAILURE-2a-safe: the
       thumb-index gap is never a candidate at all, since the thumb was
       already excluded from main_peaks upstream (_identify_thumb).
    2. EDGE: neither side stood out on its own (FAILURE 2b - a missing
       edge finger shrinks the whole band instead of widening one
       gap), so each edge is checked for whether it's plausibly the
       missing finger's slot: wide enough to hold one
       (EDGE_MIN_RATIO_FOR_CANDIDACY) AND _find_bulge finds no real
       material there (a confirmed edge digit - e.g. the thumb beyond
       `thumb_x` - must NOT be re-flagged as "missing").
    3. Fallback to the (non-standout) interior gap regardless - the
       best signal still available - and only if there's no interior
       gap at all (a single main peak found) fall back to whichever
       edge is wider.

    Parameters
    ----------
    main_peaks : list of (x, y, length)
        Non-thumb peaks, sorted left-to-right (see _identify_thumb).
    thumb_x : float or None
        The confirmed thumb's x, if any - bounds the left edge search
        so the thumb's own space is never itself flagged as missing.
    glove_mask, bx, bw
        The glove mask and its bounding-rect x/width, for _find_bulge
        and the right-edge bound.

    Returns
    -------
    dict
        {"type", "x_left", "x_right", "width", "ratio", "bound_ys"} -
        same shape _gap_bounding_box already expects.
    """
    xs = [p[0] for p in main_peaks]
    ys = [p[1] for p in main_peaks]

    interior = [
        {"type": "interior", "x_left": xs[i], "x_right": xs[i + 1], "width": xs[i + 1] - xs[i],
         "bound_ys": [ys[i], ys[i + 1]]}
        for i in range(len(xs) - 1)
    ]
    widths = [g["width"] for g in interior if g["width"] > 0]
    median_gap = float(np.median(widths)) if widths else max(1.0, bw / float(EXPECTED_MAIN_FINGERS))
    for g in interior:
        g["ratio"] = g["width"] / median_gap if median_gap > 1e-6 else 0.0

    best_interior = max(interior, key=lambda g: g["ratio"]) if interior else None
    strong_interior = False
    if best_interior is not None:
        if len(interior) >= 2:
            worst_ratio = min(g["ratio"] for g in interior)
            strong_interior = (best_interior["ratio"] / max(worst_ratio, 0.05)) >= STRONG_INTERIOR_MARGIN
        else:
            strong_interior = best_interior["ratio"] >= STRONG_INTERIOR_SINGLE_RATIO

    if strong_interior:
        return best_interior

    left_edge_x0 = thumb_x if thumb_x is not None else bx
    right_edge_x1 = bx + bw
    edge_candidates = []

    left_bulge = _find_bulge(glove_mask, left_edge_x0, xs[0])
    if left_bulge is None or left_bulge[2] < BULGE_MIN_PROMINENCE_PX:
        width = xs[0] - left_edge_x0
        ratio = width / median_gap if median_gap > 1e-6 else 0.0
        if ratio >= EDGE_MIN_RATIO_FOR_CANDIDACY:
            edge_candidates.append({"type": "edge_left", "x_left": left_edge_x0, "x_right": xs[0],
                                     "width": width, "ratio": ratio, "bound_ys": [ys[0]]})

    right_bulge = _find_bulge(glove_mask, xs[-1], right_edge_x1)
    if right_bulge is None or right_bulge[2] < BULGE_MIN_PROMINENCE_PX:
        width = right_edge_x1 - xs[-1]
        ratio = width / median_gap if median_gap > 1e-6 else 0.0
        if ratio >= EDGE_MIN_RATIO_FOR_CANDIDACY:
            edge_candidates.append({"type": "edge_right", "x_left": xs[-1], "x_right": right_edge_x1,
                                     "width": width, "ratio": ratio, "bound_ys": [ys[-1]]})

    if edge_candidates:
        return max(edge_candidates, key=lambda g: g["ratio"])

    if best_interior is not None:
        return best_interior

    # Last resort: only 1 main peak, so no interior gap exists at all -
    # fall back to whichever edge is wider.
    left_width = xs[0] - left_edge_x0
    right_width = right_edge_x1 - xs[-1]
    if right_width >= left_width:
        return {"type": "edge_right", "x_left": xs[-1], "x_right": right_edge_x1, "width": right_width,
                "ratio": right_width / median_gap if median_gap > 1e-6 else 0.0, "bound_ys": [ys[-1]]}
    return {"type": "edge_left", "x_left": left_edge_x0, "x_right": xs[0], "width": left_width,
            "ratio": left_width / median_gap if median_gap > 1e-6 else 0.0, "bound_ys": [ys[0]]}


def _column_top_ys(glove_mask, x_left, x_right):
    """Topmost glove_mask y at each integer column in [x_left, x_right]."""
    h, w = glove_mask.shape[:2]
    tops = []
    for x in range(max(0, int(x_left)), min(w, int(x_right) + 1)):
        col = np.nonzero(glove_mask[:, x])[0]
        if col.size:
            tops.append(int(col[0]))
    return tops


def _gap_bounding_box(gap, glove_mask, min_height, glove_bottom):
    """
    The box for a flagged gap: the middle GAP_TRIM_FRAC-trimmed 60% of
    its x-range (so it doesn't overlap the genuine fingers on either
    side), from the median fingertip height of the two fingers bounding
    THIS gap down to the valley/webbing between them.

    The valley/webbing height is the MEDIAN column-top y across the
    trimmed x-range, not the single shallowest (minimum-y) column - a
    lone bump or fold partway across a wide gap (e.g. a partially-
    collapsed stub between two intact fingers) previously dominated
    that minimum and made the box collapse to a sliver, even though
    most of the gap's width sat much lower, at the true webbing level.
    The median is robust to exactly that kind of single narrow spike.

    box_h is then floored at min_height (~60% of the found fingers'
    own median length, passed in by the caller) so a gap whose local
    surface happens to sit unusually high still renders a legible box.
    """
    width = gap["x_right"] - gap["x_left"]
    trim = GAP_TRIM_FRAC * width
    shrunk_left = gap["x_left"] + trim
    shrunk_right = gap["x_right"] - trim
    if shrunk_right <= shrunk_left:
        shrunk_left, shrunk_right = gap["x_left"], gap["x_right"]

    target_peak_y = int(np.median(gap["bound_ys"]))

    col_tops = _column_top_ys(glove_mask, shrunk_left, shrunk_right)
    baseline_y = int(np.median(col_tops)) if col_tops else target_peak_y + min_height

    box_y = min(target_peak_y, baseline_y)
    box_h = max(baseline_y - box_y, min_height)
    box_h = max(5, min(box_h, glove_bottom - box_y))
    box_x = int(shrunk_left)
    box_w = max(5, int(shrunk_right - shrunk_left))
    return box_x, box_y, box_w, box_h


# ============================================================
# SECONDARY SIGNAL: present-but-short finger (5 peaks found)
# ============================================================

def _valley_angle_degrees(far, start, end):
    """Interior angle at `far` in the triangle far-start-end, degrees."""
    far, start, end = np.array(far, dtype=np.float64), np.array(start, dtype=np.float64), np.array(end, dtype=np.float64)
    a = np.linalg.norm(far - start)
    b = np.linalg.norm(far - end)
    c = np.linalg.norm(start - end)
    if a < 1e-6 or b < 1e-6:
        return 180.0
    cos_angle = np.clip((a ** 2 + b ** 2 - c ** 2) / (2 * a * b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def find_valleys(contour, min_depth_ratio=MIN_VALLEY_DEPTH_RATIO):
    """
    Find confirmed inter-finger valley points on `contour` via
    convexity defects. Only used for the secondary short-finger slot
    check (see module docstring) - a fused/missing finger's own
    valleys are exactly the ones this can't rely on, which is why
    finger counting no longer goes through this path.

    Returns
    -------
    list of (far_idx, far_pt, depth_px)
        Confirmed valleys, ordered left-to-right by x-coordinate.
    """
    glove_area = cv2.contourArea(contour)
    if glove_area <= 0:
        return []
    size_scale = float(np.sqrt(glove_area))
    min_depth_px = min_depth_ratio * size_scale

    _, bbox_y, _, bbox_h = cv2.boundingRect(contour)
    max_valid_y = bbox_y + MAX_VALLEY_DEPTH_Y_FRACTION * bbox_h

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    hull_idx = np.unique(hull_idx.flatten())
    hull_idx = np.sort(hull_idx).reshape(-1, 1)
    if len(hull_idx) < 4:
        return []

    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        return []

    candidates = []
    for start_idx, end_idx, far_idx, depth in defects[:, 0]:
        depth_px = depth / 256.0
        if depth_px < min_depth_px:
            continue

        far_pt = tuple(int(v) for v in contour[far_idx][0])
        if far_pt[1] > max_valid_y:
            continue

        start_pt = tuple(int(v) for v in contour[start_idx][0])
        end_pt = tuple(int(v) for v in contour[end_idx][0])
        angle = _valley_angle_degrees(far_pt, start_pt, end_pt)
        if angle > MAX_VALLEY_ANGLE_DEG:
            continue

        candidates.append((far_idx, far_pt, depth_px))

    # Non-max suppression: keep the deepest defect among any cluster of
    # candidates that are close together on the contour.
    n = len(contour)
    min_sep = int(n * NMS_SEPARATION_FRAC)
    candidates.sort(key=lambda c: c[2], reverse=True)
    confirmed = []
    for far_idx, far_pt, depth_px in candidates:
        if all(min(abs(far_idx - j), n - abs(far_idx - j)) > min_sep for j, _, _ in confirmed):
            confirmed.append((far_idx, far_pt, depth_px))

    confirmed.sort(key=lambda c: c[1][0])  # left -> right by x
    return confirmed


def locate_missing_finger(contour, valleys):
    """
    Measure every finger slot's length (valley baseline to fingertip
    peak) and return the worst (most below-median) one, if it clears
    SHORT_FINGER_RATIO_THRESHOLD. Only called when peak-counting has
    already found all 5 fingers - see module docstring.

    Returns
    -------
    dict or None
        {"slot_index", "length_ratio", "bounding_box", "finger_count"}
    """
    if len(valleys) < MIN_VALLEYS_FOR_LOCALIZATION:
        return None

    valleys = sorted(valleys, key=lambda v: v[2], reverse=True)[:4]
    valleys = sorted(valleys, key=lambda v: v[1][0])
    valley_xs = [v[1][0] for v in valleys]
    valley_ys = [v[1][1] for v in valleys]

    bx, by, bw, bh = cv2.boundingRect(contour)
    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]

    boundaries = [bx] + valley_xs + [bx + bw]

    slots = []
    for i in range(len(boundaries) - 1):
        x_left, x_right = boundaries[i], boundaries[i + 1]
        if x_right <= x_left:
            continue
        in_slot = (xs >= x_left) & (xs <= x_right)
        if not np.any(in_slot):
            continue
        peak_y = int(ys[in_slot].min())

        neighbour_ys = []
        if i > 0:
            neighbour_ys.append(valley_ys[i - 1])
        if i < len(valley_ys):
            neighbour_ys.append(valley_ys[i])
        baseline_y = float(np.mean(neighbour_ys)) if neighbour_ys else float(by + bh)

        slots.append({
            "index": i,
            "x_left": int(x_left),
            "x_right": int(x_right),
            "peak_y": peak_y,
            "baseline_y": baseline_y,
            "length": max(0.0, baseline_y - peak_y),
        })

    if len(slots) < 3:
        return None

    lengths = [s["length"] for s in slots]
    worst = None
    for i, s in enumerate(slots):
        others = lengths[:i] + lengths[i + 1:]
        median_other = float(np.median(others)) if others else 0.0
        ratio = (s["length"] / median_other) if median_other > 1e-6 else 1.0
        s["ratio"] = ratio

        # Outer slots (thumb-side/pinky-side) only have one neighbouring
        # valley to anchor a baseline on, and the thumb points
        # diagonally rather than straight up, so this vertical-length
        # metric systematically misreads it as "short" - restrict
        # flagging to the inner, two-neighbour slots.
        if i == 0 or i == len(slots) - 1:
            continue
        if worst is None or ratio < worst["ratio"]:
            worst = s

    if worst is None or worst["ratio"] >= SHORT_FINGER_RATIO_THRESHOLD:
        return None

    normal_peaks = [s["peak_y"] for s in slots if s is not worst]
    target_peak_y = int(np.median(normal_peaks)) if normal_peaks else worst["peak_y"]

    box_y = min(target_peak_y, int(worst["baseline_y"]))
    box_h = max(1, int(worst["baseline_y"]) - box_y)
    box_x = worst["x_left"]
    box_w = max(1, worst["x_right"] - worst["x_left"])

    return {
        "slot_index": worst["index"],
        "length_ratio": float(worst["ratio"]),
        "bounding_box": (int(box_x), int(box_y), int(box_w), int(box_h)),
        "finger_count": len(slots),
    }


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_finger_not_enough(processed, segmentation):
    """
    Detect a missing/fused/short finger in a segmented glove.

    Parameters
    ----------
    processed : dict
        Output of preprocess_image(). Unused directly - the signal is
        purely geometric - but accepted for a consistent detector
        signature.
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

    if glove_mask is None or not glove_area or glove_area <= 0:
        return _empty_result()

    contour = _largest_contour(glove_mask)
    if contour is None:
        return _empty_result()

    peaks = locate_peaks(glove_mask)
    bx, by, bw, bh = cv2.boundingRect(contour)
    thumb_x, main_peaks = _identify_thumb(peaks, glove_mask, bx)

    def _whole_glove_fallback():
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros_like(glove_mask)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        return {
            "defect_name": "finger_not_enough",
            "detected": True,
            "detection_score": DETECTION_SCORE_SEVERE,
            "algorithm": ALGORITHM,
            "bounding_box": (int(x), int(y), int(w), int(h)),
            "mask": mask,
            "measurements": {
                "area_pct": None,
                "fingers_found": len(peaks),
                "localization": "unavailable",
            },
        }

    # --- Fewer than 2 peaks total: no pair to compute any spacing from ---
    if len(peaks) < 2:
        return _whole_glove_fallback()

    # --- All 4 main fingers found: check the present-but-short signal
    # as a secondary check (only meaningful once count-wise nothing
    # looks missing). The thumb being confirmed or not doesn't matter
    # here - see _identify_thumb's docstring for why it's excluded
    # from the count in the first place. ---
    if len(main_peaks) >= EXPECTED_MAIN_FINGERS:
        valleys = find_valleys(contour)
        located = locate_missing_finger(contour, valleys)

        if located is None:
            return {
                "defect_name": "finger_not_enough",
                "detected": False,
                "detection_score": 0.0,
                "algorithm": ALGORITHM,
                "bounding_box": None,
                "mask": None,
                "measurements": {"area_pct": None, "fingers_found": len(peaks)},
            }

        x, y, w, h = located["bounding_box"]
        mask = np.zeros_like(glove_mask)
        mask[y:y + h, x:x + w] = 255
        score = float(np.clip(1.0 - located["length_ratio"], 0.0, 1.0))

        return {
            "defect_name": "finger_not_enough",
            "detected": True,
            "detection_score": score,
            "algorithm": ALGORITHM,
            "bounding_box": (int(x), int(y), int(w), int(h)),
            "mask": mask,
            "measurements": {
                "area_pct": None,
                "fingers_found": len(peaks),
                "missing_slot_index": located["slot_index"],
                "length_ratio": round(located["length_ratio"], 3),
                "localization": "slot",
            },
        }

    # --- Fewer than 4 main fingers: a genuine finger is missing or
    # fused - localise it via _localize_missing_main_finger. ---
    if not main_peaks:
        return _whole_glove_fallback()

    worst_gap = _localize_missing_main_finger(main_peaks, thumb_x, glove_mask, bx, bw)
    median_finger_length = float(np.median([p[2] for p in main_peaks]))
    min_gap_height = max(5, int(MIN_GAP_BOX_HEIGHT_FRACTION * median_finger_length))
    x, y, w, h = _gap_bounding_box(worst_gap, glove_mask, min_gap_height, by + bh)

    gap_score = float(np.clip(
        (worst_gap["ratio"] - GAP_RATIO_THRESHOLD) / (GAP_SEVERE_RATIO - GAP_RATIO_THRESHOLD),
        0.0, 1.0,
    ))
    count_floor = DETECTION_SCORE_ONE_MISSING if len(main_peaks) == EXPECTED_MAIN_FINGERS - 1 else DETECTION_SCORE_SEVERE
    score = max(gap_score, count_floor)

    mask = np.zeros_like(glove_mask)
    mask[y:y + h, x:x + w] = 255

    return {
        "defect_name": "finger_not_enough",
        "detected": True,
        "detection_score": float(score),
        "algorithm": ALGORITHM,
        "bounding_box": (int(x), int(y), int(w), int(h)),
        "mask": mask,
        "measurements": {
            "area_pct": None,
            "fingers_found": len(peaks),
            "main_fingers_found": len(main_peaks),
            "thumb_confirmed": thumb_x is not None,
            "gap_type": worst_gap["type"],
            "gap_ratio": round(worst_gap["ratio"], 3),
            "localization": "gap",
        },
    }
