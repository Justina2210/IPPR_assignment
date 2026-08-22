# Known Limitations

Consolidated list of every known limitation in the classical-CV glove
defect detection pipeline, for the report's Critical Analysis section.
Numbers below are from the latest `evaluate.py` run and FP sweep against
the full 68-image dataset (`datasets/{cotton,latex,nitrile}`, 3
detectors implemented: `tearing`, `tearing_fingertip`,
`finger_not_enough`).

## 1. Thresholds tuned by eye on the 68-image dataset

12 threshold constants across the three detectors were calibrated by
observing their effect on this specific 68-image dataset rather than
derived from a closed-form/geometric argument. Each is marked in the
source with a `# TUNED-BY-EYE on the 68-image dataset` comment, so they
can be identified at a glance and are not presented in the report as
principled derivations:

- `detectors/tearing.py`: `MIN_COLOUR_DISTANCE` (22.0), `MIN_LIGHTNESS_DISTANCE` (28.0), `STRONG_HOLE_AREA_RATIO` (0.09)
- `detectors/tearing_fingertip.py`: `MIN_COLOUR_DISTANCE` (22.0), `MIN_LIGHTNESS_DISTANCE` (28.0), `STRONG_HOLE_AREA_RATIO` (0.50), `TRANSLUCENT_MAX_RATIO_CAP` (0.30), `MIN_CHROMA_FOR_AREA_RANKING` (4.5)
- `detectors/finger_not_enough.py`: `THUMB_LENGTH_RATIO` (0.4), `BULGE_MIN_PROMINENCE_PX` (20), `STRONG_INTERIOR_MARGIN` (2.0), `STRONG_INTERIOR_SINGLE_RATIO` (1.3)

With only 5-6 positive examples per defect category, these values are
fitted to a small sample and are not guaranteed to generalise to
photos taken under different lighting, backgrounds, or camera angles.

## 2. Per-detector false-positive rates (single-defect thresholds over-fire on unrelated defects)

Each detector's `DETECTION_SCORE_THRESHOLD` (0.5) was tuned only
against its own defect's images, because `evaluate.py`'s real
evaluation loop only ever runs a detector against the one folder
matching its name (see `evaluate.py`'s closing docstring: "each
labelled image is only ever tested against its own matching detector
- there are no true negatives in this setup"). None of the three
detectors were ever tuned against negative (non-matching-defect)
examples, so nothing in their calibration discourages firing on a
colour/texture anomaly that happens to come from a *different* defect.

FP sweep result (each detector run against every image **not** in its
own folder, same `detected = score >= 0.5` rule `evaluate.py` uses):

| Detector | FP rate | Accepted baseline |
|---|---|---|
| `tearing` | **24/63** | (no baseline set) |
| `tearing_fingertip` | **36/62** | must stay ≤ 36/62 |
| `finger_not_enough` | **14/63** | must stay ≤ 34/63 |

Why they over-fire, by detector:
- `tearing` and `tearing_fingertip` both key off local LAB colour
  deviation from the glove's own material colour. That signal is, by
  construction, indistinguishable from `dirty`, `stain`, `spotting`,
  `discoloration`, and `plastic_contamination` on colour evidence
  alone - all five are also local colour anomalies. `tearing`'s FP
  list includes exactly these categories (e.g.
  `cotton/dirty/cotton_dirty_1.jpeg` score 1.00,
  `nitrile/stain/nitrile_stain_3.jpeg` score 1.00). This is called out
  explicitly in `detectors/tearing.py`'s own module docstring.
- `finger_not_enough`'s `detection_score` is a CATEGORY VERDICT from
  finger count alone (`FINGER_COUNT_SCORE_ONE_MISSING` = 0.75,
  `FINGER_COUNT_SCORE_SEVERE` = 1.0 - see
  `detectors/finger_not_enough.py`'s module docstring, point 4) - it no
  longer factors in how wide the localised gap itself is. A weighted
  blend with the gap's own width (`worst_gap["ratio"]`) was built and
  then measured directly against all 19 images that reach the
  gap-localisation branch (the 5 real positives + these 14 false
  positives): `gap_score` was 0.0000 on 4 of the 5 real positives *and*
  13 of the 14 false positives, and the raw gap ratios overlap almost
  completely between the two classes (positives 1.03-2.26, false
  positives 0.80-1.57; the best possible single split point over all 19
  values only reaches 15/19 correct). The blend could not move the
  false-positive count at any weight that didn't also risk real
  detections, so it was removed rather than kept as measured dead
  weight - see `FINGER_COUNT_SCORE_ONE_MISSING`'s comment in the source
  for the full measurement. `worst_gap["ratio"]` is still used to place
  the box and is still reported in `measurements["gap_ratio"]`, just not
  used to decide the score.

  Visually auditing all 14 false positives (folder contents, not just
  the geometry) splits them roughly in half: 6 (`cotton_touching_2`,
  `latex_touching_1`, `latex_touching_2`, `nitrile_touching_1`,
  `nitrile_touching_2`, `latex_damaged_by_fold_2`) show a genuinely
  fused or folded finger on inspection - the detector's count is a
  defensible read of the glove's real shape, it's just that "fingers
  touching/folded" and "finger not enough" are different folder labels
  for geometrically similar hand shapes. The other 7 (colour/
  contamination/beading/tearing categories) are plainly intact
  five-finger gloves where the peak-finder simply missed a finger - a
  real detection gap, not a labelling overlap. One more
  (`nitrile_tearing_fingertip_2`) is a curled/fisted hand pose that
  breaks the centroid-distance peak method's core assumption
  independent of finger count.
- Because each detector is only ever invoked in the real pipeline
  against its own matching folder (the registry maps 1 defect → 1
  detector), none of this FP behaviour is visible in the headline
  detection-rate numbers - it only shows up when a detector is run
  against unrelated images, as done here and in the GUI's "Accuracy
  evaluation" cross-test.

## 3. Translucent-latex ambiguity for `tearing_fingertip`

`latex_tearing_fingertip_1.jpg` scores **0.04** and is reported as
**not detected**, despite the box being correctly localised on the
actually-torn finger. On translucent latex, a torn fingertip's
exposed-skin patch is small and only weakly colour-deviating (mean LAB
chroma distance ~7.8 over a 34px blob), while an intact fingertip's
translucency can independently produce a *larger* but non-torn colour
deviation elsewhere on the same hand. The detector's boundary-based
composite ranking (see `detectors/tearing_fingertip.py`'s module
docstring, "PROBLEM 1") already picks the correct finger for this
image, but the confidence score is still derived from raw anomaly area,
which stays small for a genuinely subtle tear.

Two fixes were tried and rejected because they wrecked the FP rate
when tested against the full 62-image sweep:
- Flooring the score at the winning candidate's composite confidence:
  raised `tearing_fingertip` FP from 36/62 to **59/62** (composite
  ranking always produces a relatively-best winner on any image, which
  is not evidence a real tear exists).
- A stricter absolute chroma/edge/sharpness triple-floor: still fired
  on **28/62** sweep images, since ordinary `dirty`/`discoloration`/
  `plastic_contamination` fingertips routinely show a *stronger*
  absolute colour+edge signal than this one subtle tear.

Conclusion: this image is fundamentally ambiguous for colour-based
detection at a confidence level that survives the FP sweep. It is
reported honestly as a miss (`tearing_fingertip` detection rate is
5/6, 83.3%) rather than forced past the threshold.

## 4. Fallback vs. true localization

`detectors/finger_not_enough.py` has a documented fallback: "Whole-glove
bounding box is used only when fewer than 2 peaks are found at all -
there's no pair of points to compute any spacing from, so no gap can be
meaningfully localised" (module docstring, "Fallback" section).

Verified directly against all 16 successfully-detected images across
the three detectors (bounding box compared to the glove's own full
silhouette bounding box): **the fallback did not trigger on any image**
in the current 68-image dataset - every reported box is a true,
sub-glove localisation, not the whole-glove fallback. The fallback path
exists and is exercised only if a future/uploaded photo produces fewer
than 2 detected fingertip peaks (e.g. a badly cropped or heavily
obscured hand).

## 5. Folder labels as ground truth, with unlabelled secondary defects

Both `evaluate.py`'s detection-rate metric and the GUI's "Accuracy
evaluation (full dataset)" cross-test use the dataset's folder name as
the sole ground-truth label for an image. A photo in, say,
`latex/tearing/` is assumed to contain *only* tearing and nothing else,
when in practice a real photo can carry secondary, unlabelled defects
(e.g. a tearing photo that is also visibly dirty). Any such secondary
defect that a detector correctly flags is scored as a false positive
under this scheme, because there is no ground truth for it. This caveat
is shown directly above the accuracy table in `app.py`, in the caption
text inside the "Accuracy evaluation (full dataset)" expander
(immediately above the "Run full evaluation" button): "Ground truth
comes from dataset folder labels. Images may contain unlabelled
secondary defects, which appear as false positives..." - and is why
`evaluate.py`'s own docstring says
precision/recall/confusion-matrix metrics are not its primary output
for the single-label detection-rate run - though the GUI's separate
full cross-test does compute them anyway, with this caveat attached.

## 6. Segmentation weaknesses: similar-hue backgrounds (e.g. teal cloth)

`segmentation.py`'s foreground/background split is a LAB colour/
brightness distance from an estimated (border-sampled) background
colour. This historically produced near-total segmentation failures
when the glove and background shared a similar hue - explicitly
documented in `_otsu_threshold_mask()`'s own docstring in
`segmentation.py`: "this was the root cause of near-total segmentation
failures, e.g. a blue glove on a similarly-hued teal background." The
fix in place (Otsu-based adaptive thresholding inside
`_otsu_threshold_mask()`, replacing a fixed "median + k·std" formula)
removed this failure mode for every photo in
the current 68-image dataset - the full `evaluate.py` run and the FP
sweep both show **0 segmentation failures out of 68 images**. This is a
mitigation, not a guarantee: the underlying risk (foreground and
background sitting close together in LAB colour space) is a property
of the photo, not something the algorithm can fully rule out, so a new
photo with an even closer glove/background colour match than any in
this dataset could still under-segment.
