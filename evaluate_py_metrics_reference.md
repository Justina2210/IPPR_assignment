# evaluate.py — Metrics Reference (for app.py development)

This lists every field `evaluate.py` produces, so `app.py` can be built against
a known, fixed set of outputs instead of guessing field names.

## Per-image record (written to `results.csv`)

| Field | Type | Meaning |
|---|---|---|
| `material` | str | nitrile / latex / cotton |
| `defect_name` | str | the defect being tested (= ground truth, since each image only runs its matching detector) |
| `image_path` | str | path to the source image |
| `status` | str | `success` / `segmentation_failure` / `detector_failure` / `not_implemented` |
| `detected` | bool | final detected/not-detected call (score vs. `DETECTION_THRESHOLD = 0.5`) |
| `detection_score` | float 0.0–1.0 | the detector's confidence score |
| `algorithm` | str | short description of the technique used, e.g. "LAB colour deviation + connected components" |
| `bounding_box` | (x, y, w, h) or None | detected region, auto-derived from mask if the detector didn't supply one |
| `glove_area` | int | pixel count of the segmented glove (denominator for area %) |
| `defect_area_pct` | float or None | defect size as % of glove area, from the detector's `measurements` |
| `processing_time_ms` | float | how long that image took |
| `error` | str or None | failure reason, if any |

## Per-defect / per-material / overall summary (written to `summary.json`)

- `total_images`
- `detected` (count)
- `detection_rate_pct`
- `segmentation_failures`, `segmentation_failure_rate_pct`
- `detector_failures`, `detector_failure_rate_pct`
- `not_implemented`
- `mean_detection_score`, `stdev_detection_score`
- `mean_score_when_detected`
- `mean_processing_time_ms`
- `mean_defect_area_pct`
- `algorithm`

## Not included, on purpose

- **Accuracy / precision / recall** — no true negatives in this design (each image
  is only tested against its matching detector), so these aren't computable.
- **Multi-class prediction** — the system never picks among all 12 defects for one
  image, only yes/no on the single selected defect.

## The GUI's separate cross-test (app.py's run_full_evaluation)

The "not computable" statement above is about `evaluate.py`'s own run
specifically, not a claim that these metrics can never be produced from
this codebase. `app.py`'s `run_full_evaluation()` (behind the "Accuracy
evaluation (full dataset)" expander) is a genuinely different evaluation
with a different ground-truth setup, not an extension of `evaluate.py`'s
run:

- `evaluate.py` runs each image through only the ONE detector matching
  its own folder label — there is no case where a detector is asked
  about an image it doesn't apply to, so there's no way to define a
  true negative.
- `run_full_evaluation()` instead cross-tests every implemented
  detector against every image in the dataset (via
  `_run_cached_detector()`), using "does this image's folder name equal
  this detector's defect name" as ground truth for that pair. That
  produces actual TP/FP/FN/TN counts per detector, so Accuracy /
  Precision / Recall / F1 are computable there — but they're being
  computed against a much less trustworthy ground truth than
  `evaluate.py`'s: a `dirty` image labelled "not tearing" for the
  `tearing` cross-test may still genuinely be torn as an unlabelled
  secondary defect, which would score as a false positive despite the
  detector being right (see the caveat text `app.py` shows directly
  above that table).

In short: `evaluate.py`'s numbers and `app.py`'s cross-test numbers are
two different measurements answering two different questions, not one
superseding the other. Don't read a low Accuracy/Precision from the GUI
table as contradicting `evaluate.py`'s detection-rate figures, or vice
versa — check which ground-truth setup a given number came from before
comparing it to anything else.

## For app.py (single-image GUI view)

Per the architecture doc's Section 9 GUI output spec, these are the fields to
show for one image after running a detector:

- **Detection Result** — `detected`
- **Detection Score** — `detection_score` (display as %)
- **Defect Area** — `defect_area_pct` (display as %, "when applicable")

These come from the same detector return value that `evaluate.py` consumes —
see the detector contract (function signature, required return keys, registry
line) in the main guideline doc for how to call a detector directly from
`app.py`.
