# Ka Sin (M2) — dirty, stain, spotting

Detectors for the three surface-contamination defects, written against the
contract in `Detector_Development_Guideline.docx`.

## Files

| File | Purpose |
|---|---|
| `detectors/_anomaly.py` | Shared colour-anomaly front-end. **Not a detector** — the leading underscore keeps it out of `DETECTOR_REGISTRY`. |
| `detectors/dirty.py` | `detect_dirty(processed, segmentation)` |
| `detectors/stain.py` | `detect_stain(processed, segmentation)` |
| `detectors/spotting.py` | `detect_spotting(processed, segmentation)` |
| `detectors/__init__.py` | Makes `detectors` a package so `importlib` can find it. |
| `evaluate_kasin.py` | Member-level evaluation — mutual discrimination and specificity. Does not replace the shared `evaluate.py`. |

All three are already registered in `evaluate.py`, and nothing outside
`detectors/` was modified.

## Design

All three defects are foreign material sitting on an otherwise uniform glove
surface, and all three are **darker** than the glove. They therefore share one
question — *which pixels do not look like glove material?* — and differ only in
*what shape that material has*. The expensive part is written once in
`_anomaly.py`; the three detectors stay small.

Pipeline: interior margin → illumination background → anomaly maps →
candidate mask → blob measurement → per-defect scoring.

Three decisions carry most of the weight:

**The illumination model is a robust median, not a blur or a closing.** A
morphological closing takes a local maximum, so one specular highlight
propagates across its neighbourhood and makes the whole glove read as "too
dark" — this drove some images to 99% anomaly area. A plain blur has the
opposite failure: large dirty patches drag the estimate down and hide
themselves. A downsampled median with a second re-estimation pass survives
both.

**`edge_contact` rejects regions not surrounded by glove.** Real contamination
has clean glove on every side. The two biggest false-positive sources do not:
forearm skin runs off the bottom of the analysed area, and the shading band on
a curved finger runs along its silhouette. One ratio removes both without
modelling skin colour or touching segmentation.

**`isolation` separates dirty from stain.** A stain leaves the glove around it
clean, so its darkening is many times that of its surroundings (5.5–17.7 on the
true stains). Soiling fades outward into a halo, so the ratio stays low
(3.3–4.6 on the dirty images).

Thresholds are fixed constants, not Otsu. Otsu always returns a split, so on a
clean glove it manufactures a defect out of ordinary shading. A system that has
to be able to answer "nothing wrong here" needs an absolute physical threshold.

## Results

Detection rate on own folders (threshold 0.5, as applied by `evaluate.py`):

| Detector | Detected | Highest-scoring detector is correct |
|---|---|---|
| dirty | 6/6 | 3/6 |
| stain | 5/5 | 5/5 |
| spotting | 6/6 | 6/6 |

Response on images that are not these defects:

| Detector | Geometry defects (39) | Colour defects (12) |
|---|---|---|
| dirty | 23 (59.0%) | 4 (33.3%) |
| stain | 14 (35.9%) | 12 (100%) |
| spotting | 0 (0.0%) | 0 (0.0%) |

Negatives are split deliberately. Geometry defects (tearing, touching, folds,
beading, oversize, finger-not-enough) have no colour anomaly, so a response
there is a genuine false positive. Colour defects (discoloration, plastic
contamination) genuinely do contain colour anomalies, so a low-level response
is expected — that ambiguity belongs to score comparison, not to the detector.

## Limitations — state these in the report, don't hide them

1. **Thresholds were tuned by eye on the same 68 images used for testing.** The
   detection rates above are therefore optimistic and are not held-out
   estimates. No separate validation set exists.

2. **Dirty is out-scored by stain on 3 of its 6 images.** A heavy smear and a
   broad stain are physically similar, and pushing further on 17 images would
   be fitting noise. If the GUI names a defect by top score, dirty/stain
   margins should be treated as ambiguous rather than decisive.

3. **Dirty fires on 59% of geometry defects.** These are fold and pose shadows.
   The chromatic gate helps but cannot fully separate them: the overlap band
   around 0.09–0.18 chromatic fraction contains both the weakest true positive
   (0.089) and the strongest shadow case (0.099). It is applied as a soft
   multiplier rather than a veto because a hard veto removes more true
   positives than false ones — grey dust on white cotton is genuinely close to
   achromatic.

4. **Only 17 images across two materials per defect.** Dirty has no nitrile
   example, stain has no cotton example, spotting has no cotton example.

## Two issues for the group

**`evaluate.py` line 78: `DATASET_ROOT = "dataset"`, but the folder is
`datasets`.** No evaluation run finds any images until this is fixed. I used a
local symlink rather than edit shared code.

**The cuff cut only fires on nitrile.** On pale latex and cotton, Stage 2's
colour confirmation rejects the forearm because skin and cream latex read as
"the same material", so wrist and forearm stay inside `glove_mask` on 9 of my
17 images. Per the guideline this is a segmentation failure, not a detector
one, so I have not patched it — `edge_contact` limits the damage on my side,
but it will affect anyone doing colour-based work. `cotton_plastic_2.jpeg` is a
separate segmentation failure: a chunk of teal background sits inside the mask.
