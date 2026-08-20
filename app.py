"""
app.py
------
Streamlit GUI for the Glove Defect Detection System.

Wraps evaluate.py's single-image pipeline for interactive use - one
image at a time, not the whole-dataset batch run. The heavy lifting
(preprocessing -> segmentation -> the one detector matching the
selected defect, loaded via DETECTOR_REGISTRY / load_detector()) comes
from evaluate.py. The on-screen overlay is drawn by draw_rich_overlay()
below, not evaluate.py's own draw_overlay() - that function is shared
with the batch evaluation pipeline (it's what produces the on-disk
overlay images in outputs/overlays/) and is kept untouched; this UI
wants a richer, "industrial inspection" style render (label card,
measurements, semi-transparent mask fill) that the shared function
doesn't need to grow just for the GUI.

run_pipeline() below is the one addition: it mirrors evaluate_image()'s
own logic instead of calling it directly, because this UI also needs
the intermediate `processed` / `segmentation` dicts (for the grayscale
and glove-mask panels) and evaluate_image() doesn't return them. Calling
evaluate_image() AND re-running preprocess_image()/segment_glove() to
get those dicts would run the pipeline twice per image, so instead this
wrapper runs preprocess_image() and segment_glove() once and feeds them
to the detector exactly the way evaluate_image() does.

The visual design (colours, cards, step badges, segmented pickers) is
adapted from a static HTML mockup. Its own JS defect-detection logic
was not used anywhere here - every result on screen comes from the
real Python pipeline above.
"""

import io
import os
import time
import traceback
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import evaluate
from evaluate import DATASET_ROOT, DETECTOR_REGISTRY, MIN_GLOVE_AREA, load_detector
from preprocessing import load_image, preprocess_image
from segmentation import segment_glove

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SEGMENTATION_ALGORITHMS = [
    "Resize (aspect-preserving) + mild bilateral denoising",
    "Grayscale / HSV / LAB colour-space conversion + CLAHE contrast enhancement",
    "LAB background estimation (border sampling) + Otsu-thresholded colour/lightness foreground mask",
    "Morphological open/close cleaning, largest-connected-component selection, hole filling",
    "Geometric plateau detection + colour-confirmed cuff/forearm removal",
]

st.set_page_config(page_title="Glove Defect Detection", layout="centered")


# ============================================================
# STYLE
# ============================================================

def inject_style(step2_dim, step3_dim):
    st.markdown(
        f"""
        <style>
        :root {{
            --ink:#111820; --grey:#6B7A85; --page:#F5F6F6; --card:#FFFFFF;
            --line:#E2E7EA; --blue:#1F4FD8; --red:#D8382B; --green:#127A52;
        }}

        .stApp {{ background-color: var(--page); }}
        .block-container {{ padding-top: 2.2rem; max-width: 700px; }}
        html, body, [class*="css"] {{ color: var(--ink); }}

        /* step / result cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--card) !important;
            border-radius: 16px !important;
            border: 1px solid var(--line) !important;
            box-shadow: none !important;
        }}

        .step-head {{ display:flex; align-items:center; gap:12px; margin-bottom:14px; }}
        .step-num {{
            width:26px; height:26px; border-radius:50%; background:var(--ink); color:#fff;
            font-size:13px; font-weight:600; display:flex; align-items:center; justify-content:center;
            flex:0 0 auto;
        }}
        .step-num.done {{ background: var(--green); }}
        .step-title {{ font-size:15px; font-weight:600; color:var(--ink); }}
        .step-hint {{ font-size:13px; color:var(--grey); }}

        /* segmented / list pickers */
        .st-key-materialpills button, .st-key-defectpills button {{
            border-radius:10px !important; border:1px solid var(--line) !important;
            background:var(--card) !important; color:var(--ink) !important;
        }}
        .st-key-materialpills button[kind="primary"], .st-key-defectpills button[kind="primary"] {{
            background:var(--ink) !important; border-color:var(--ink) !important;
            color:#fff !important; font-weight:600 !important;
        }}
        .st-key-defectpills button {{ justify-content:flex-start !important; }}
        .st-key-defectpills button p {{ text-align:left !important; font-size:14px; }}
        .st-key-materialpills button:hover, .st-key-defectpills button:hover {{ border-color:var(--blue) !important; }}

        /* primary CTA (Run detection) stays blue */
        button[kind="primary"] {{ background:var(--blue); border-color:var(--blue); font-weight:600; }}
        button[kind="primary"]:hover {{ background:#1740B0; border-color:#1740B0; }}

        /* result header bar */
        .result-head {{ padding:16px 20px; border-radius:16px 16px 0 0; color:#fff; margin: -1px -1px 0 -1px; }}
        .result-head .main {{ font-size:18px; font-weight:700; }}
        .result-head .sub {{ font-size:13px; font-weight:400; opacity:.92; margin-top:3px; }}
        .head-bad {{ background: var(--red); }}
        .head-good {{ background: var(--green); }}
        .head-warn {{ background: var(--grey); }}

        .panel-caption {{ font-size:12px; color:var(--grey); text-align:center; margin-bottom:4px; }}
        .st-key-resultcard [data-testid="stImage"] img {{ border-radius:8px; }}

        [data-testid="stMetricLabel"] {{ font-size:12px; color:var(--grey); text-transform:uppercase; letter-spacing:.03em; }}
        [data-testid="stMetricValue"] {{ font-size:19px; font-weight:600; color:var(--ink); }}

        {".st-key-step2card { opacity: 0.45; pointer-events: none; }" if step2_dim else ""}
        {".st-key-step3card { opacity: 0.45; pointer-events: none; }" if step3_dim else ""}
        </style>
        """,
        unsafe_allow_html=True,
    )


def step_head_html(number, title, done):
    done_cls = "done" if done else ""
    return (
        f'<div class="step-head">'
        f'<div class="step-num {done_cls}">{number}</div>'
        f'<div class="step-title">{title}</div>'
        f"</div>"
    )


# ============================================================
# DATA DISCOVERY
# ============================================================

@st.cache_data
def list_materials():
    if not os.path.isdir(DATASET_ROOT):
        return []
    return sorted(
        name for name in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, name))
    )


@st.cache_data
def list_defects(material):
    material_dir = os.path.join(DATASET_ROOT, material)
    if not os.path.isdir(material_dir):
        return []
    return sorted(
        name for name in os.listdir(material_dir)
        if os.path.isdir(os.path.join(material_dir, name))
    )


@st.cache_data
def list_images(material, defect):
    defect_dir = os.path.join(DATASET_ROOT, material, defect)
    if not os.path.isdir(defect_dir):
        return []
    return sorted(
        name for name in os.listdir(defect_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )


@st.cache_data
def implemented_detectors():
    """Which DETECTOR_REGISTRY entries actually import successfully right now."""
    working = sorted(name for name in DETECTOR_REGISTRY if load_detector(name) is not None)
    return working, len(DETECTOR_REGISTRY)


def save_uploaded_file(uploaded_file):
    """
    The pipeline takes a file path (load_image() is cv2.imread under the
    hood), so an uploaded in-memory file has to be written to disk first.
    """
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# ============================================================
# PIPELINE WRAPPER (see module docstring)
# ============================================================

def run_pipeline(image_path, defect_name):
    """
    Same status/record logic as evaluate.evaluate_image(), but also
    returns `processed` and `segmentation` so the UI can render the
    intermediate stages without a second pipeline pass.
    """
    record = {
        "image_path": image_path,
        "defect_name": defect_name,
        "status": None,
        "detected": False,
        "detection_score": 0.0,
        "algorithm": None,
        "bounding_box": None,
        "measurements": {},
        "glove_area": None,
        "processing_time_ms": None,
        "error": None,
    }

    start = time.time()
    processed = None
    segmentation = None

    try:
        image = load_image(image_path)
        processed = preprocess_image(image)
        segmentation = segment_glove(processed)
    except Exception as exc:
        record["status"] = "segmentation_failure"
        record["error"] = f"preprocessing/segmentation crashed: {exc}"
        record["processing_time_ms"] = (time.time() - start) * 1000
        return record, None, processed, segmentation

    record["glove_area"] = segmentation["glove_area"]

    if segmentation["glove_area"] < MIN_GLOVE_AREA:
        record["status"] = "segmentation_failure"
        record["error"] = "glove mask area below MIN_GLOVE_AREA threshold"
        record["processing_time_ms"] = (time.time() - start) * 1000
        return record, processed["original"], processed, segmentation

    detector_func = load_detector(defect_name)
    if detector_func is None:
        record["status"] = "not_implemented"
        record["processing_time_ms"] = (time.time() - start) * 1000
        return record, processed["original"], processed, segmentation

    try:
        result = detector_func(processed, segmentation)
        evaluate.validate_result(result)
    except Exception as exc:
        record["status"] = "detector_failure"
        record["error"] = f"{exc}\n{traceback.format_exc(limit=2)}"
        record["processing_time_ms"] = (time.time() - start) * 1000
        return record, processed["original"], processed, segmentation

    bounding_box = result.get("bounding_box") or evaluate._bbox_from_mask(result.get("mask"))

    record["status"] = "success"
    record["detected"] = bool(result["detected"]) and result["detection_score"] >= evaluate.DETECTION_THRESHOLD
    record["detection_score"] = float(result["detection_score"])
    record["algorithm"] = result["algorithm"]
    record["bounding_box"] = bounding_box
    record["measurements"] = result.get("measurements", {})
    record["processing_time_ms"] = (time.time() - start) * 1000

    return record, processed["original"], processed, segmentation


# ============================================================
# RICH OVERLAY (industrial-inspection style, GUI only)
# ============================================================

_OVERLAY_RED = (0, 0, 230)     # BGR - defect found
_OVERLAY_GREY = (120, 120, 120)  # BGR - boxed but not over the detection threshold


def draw_rich_overlay(original_bgr, record):
    """
    Industrial-inspection-style render for the result card: a tight box
    around the defect, a label card beside it (defect name + score%,
    plus a measurement line - area% or length ratio - when available),
    and a semi-transparent fill over the detected mask region.

    Deliberately separate from evaluate.py's draw_overlay(), which stays
    untouched (it's shared with the batch pipeline and produces the
    on-disk overlay images in outputs/overlays/).
    """
    overlay = original_bgr.copy()
    bbox = record.get("bounding_box")
    if bbox is None:
        return overlay

    x, y, w, h = bbox
    mask = record.get("mask")
    color = _OVERLAY_RED if record.get("detected") else _OVERLAY_GREY

    if mask is not None and cv2.countNonZero(mask) > 0:
        fill = overlay.copy()
        fill[mask > 0] = color
        overlay = cv2.addWeighted(fill, 0.35, overlay, 0.65, 0)

    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

    defect_label = record["defect_name"].replace("_", " ")
    lines = [f"{defect_label}: {record.get('detection_score', 0.0):.0%}"]

    measurements = record.get("measurements") or {}
    area_pct = measurements.get("area_pct")
    length_ratio = measurements.get("length_ratio")
    if area_pct is not None:
        lines.append(f"area {area_pct:.2f}%")
    elif length_ratio is not None:
        lines.append(f"length ratio {length_ratio:.2f}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, pad, line_h = 0.55, 1, 6, 20

    text_w = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    card_w = text_w + pad * 2
    card_h = line_h * len(lines) + pad * 2

    # Label card sits just above the box; if that would run off the top
    # of the frame, put it below the box instead.
    card_x = max(0, min(x, overlay.shape[1] - card_w))
    card_y = y - card_h - 4
    if card_y < 0:
        card_y = min(y + h + 4, overlay.shape[0] - card_h)
    card_y = max(0, card_y)

    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), color, thickness=cv2.FILLED)
    for i, line in enumerate(lines):
        ty = card_y + pad + line_h * i + 14
        cv2.putText(overlay, line, (card_x + pad, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


# ============================================================
# COMPOSITE FIGURE (for "Save figure")
# ============================================================

def build_composite_png(material, defect, record, original_bgr, processed, segmentation, overlay_bgr):
    def to_rgb(arr):
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    panels = [
        ("Original", to_rgb(original_bgr)),
        ("Grayscale", to_rgb(processed["gray_enhanced"])),
        ("Glove mask", to_rgb(segmentation["glove_mask"])),
        ("Overlay", to_rgb(overlay_bgr)),
    ]

    thumb_w = 260
    thumbs = []
    for label, arr in panels:
        img = Image.fromarray(arr)
        ratio = thumb_w / img.width
        img = img.resize((thumb_w, max(1, int(img.height * ratio))))
        thumbs.append((label, img))

    big_w = 760
    big = Image.fromarray(to_rgb(overlay_bgr))
    ratio = big_w / big.width
    big = big.resize((big_w, max(1, int(big.height * ratio))))

    strip_w = thumb_w * 4 + 30
    canvas_w = max(big_w, strip_w) + 40
    strip_h = max(t.height for _, t in thumbs) + 26
    canvas_h = 70 + big.height + 20 + strip_h + 30

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = f"{material} / {defect} - {record['status']} (score={record['detection_score']:.1%})"
    draw.text((20, 15), title, fill="black", font=font)

    x = (canvas_w - big.width) // 2
    canvas.paste(big, (x, 55))

    y = 55 + big.height + 25
    x = (canvas_w - strip_w) // 2
    for label, thumb in thumbs:
        canvas.paste(thumb, (x, y))
        draw.text((x + thumb.width // 2 - len(label) * 3, y + thumb.height + 6), label, fill="black", font=font)
        x += thumb_w + 10

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("material", None)
st.session_state.setdefault("defect", None)
st.session_state.setdefault("full_eval_counts", None)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<h2 style="font-size:22px; font-weight:600; letter-spacing:-.01em; margin-bottom:4px;">'
    'Glove check <span style="color:#6B7A85; font-weight:400;">&middot; defect detection</span></h2>',
    unsafe_allow_html=True,
)
st.caption("Run the real detection pipeline on a sample or uploaded glove photo.")

step2_dim = st.session_state.material is None
step3_dim = st.session_state.defect is None
inject_style(step2_dim, step3_dim)


# ============================================================
# STEP 1 — MATERIAL
# ============================================================

materials = list_materials()
if not materials:
    st.error(f"No material folders found under `{DATASET_ROOT}/`.")
    st.stop()

with st.container(border=True, key="step1card"):
    st.markdown(step_head_html(1, "Choose the glove material", st.session_state.material is not None), unsafe_allow_html=True)
    with st.container(key="materialpills"):
        cols = st.columns(len(materials))
        for col, m in zip(cols, materials):
            with col:
                if st.button(
                    m.capitalize(), key=f"material-{m}", use_container_width=True,
                    type="primary" if st.session_state.material == m else "secondary",
                ):
                    if st.session_state.material != m:
                        st.session_state.material = m
                        st.session_state.defect = None
                    st.rerun()


# ============================================================
# STEP 2 — DEFECT
# ============================================================

defects = list_defects(st.session_state.material) if st.session_state.material else []

with st.container(border=True, key="step2card"):
    st.markdown(step_head_html(2, "Choose what to look for", st.session_state.defect is not None), unsafe_allow_html=True)
    if step2_dim:
        st.markdown('<div class="step-hint">Choose a material first.</div>', unsafe_allow_html=True)
    elif not defects:
        st.warning(f"No defect folders found under `{DATASET_ROOT}/{st.session_state.material}/`.")
    else:
        with st.container(key="defectpills"):
            for d in defects:
                label = d.replace("_", " ").title()
                if st.button(
                    label, key=f"defect-{d}", use_container_width=True,
                    type="primary" if st.session_state.defect == d else "secondary",
                ):
                    st.session_state.defect = d
                    st.rerun()


# ============================================================
# STEP 3 — IMAGE + RUN
# ============================================================

material = st.session_state.material
defect = st.session_state.defect
image_path, uploaded_file, display_name = None, None, None
run_clicked = False

with st.container(border=True, key="step3card"):
    st.markdown(step_head_html(3, "Load a glove photo", st.session_state.get("last_record") is not None), unsafe_allow_html=True)
    if step3_dim:
        st.markdown('<div class="step-hint">Choose a defect first.</div>', unsafe_allow_html=True)
    else:
        image_source = st.radio(
            "Image source", ["Sample from dataset", "Upload my own"],
            horizontal=True, label_visibility="collapsed",
        )

        if image_source == "Sample from dataset":
            images = list_images(material, defect)
            if not images:
                st.warning(f"No images found in `{DATASET_ROOT}/{material}/{defect}/`.")
            else:
                chosen = st.selectbox("Image file", images, label_visibility="collapsed")
                image_path = os.path.join(DATASET_ROOT, material, defect, chosen)
                display_name = chosen
                st.image(image_path, caption=chosen, width=240)
        else:
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded_file is not None:
                display_name = uploaded_file.name
                st.image(uploaded_file, caption=uploaded_file.name, width=240)

        run_clicked = st.button(
            "Run detection", type="primary", use_container_width=True,
            disabled=(image_path is None and uploaded_file is None),
        )


# ============================================================
# RUN
# ============================================================

if run_clicked:
    run_path = save_uploaded_file(uploaded_file) if uploaded_file is not None else image_path

    with st.spinner("Running preprocessing -> segmentation -> detector..."):
        record, original_bgr, processed, segmentation = run_pipeline(run_path, defect)

    if uploaded_file is not None:
        try:
            os.remove(run_path)
        except OSError:
            pass

    st.session_state["last_record"] = record
    st.session_state["last_original"] = original_bgr
    st.session_state["last_processed"] = processed
    st.session_state["last_segmentation"] = segmentation
    st.session_state["last_display_name"] = display_name
    st.session_state["last_material"] = material
    st.session_state["last_defect"] = defect


# ============================================================
# RESULT CARD
# ============================================================

record = st.session_state.get("last_record")
original_bgr = st.session_state.get("last_original")
processed = st.session_state.get("last_processed")
segmentation = st.session_state.get("last_segmentation")
res_material = st.session_state.get("last_material")
res_defect = st.session_state.get("last_defect")

if record is not None:
    status = record["status"]
    defect_label = record["defect_name"].replace("_", " ")

    if status == "success":
        if record["detected"]:
            head_cls, main_txt = "head-bad", "Defect found"
            sub_txt = f"{defect_label} detected at {record['detection_score']:.0%} confidence"
        else:
            head_cls, main_txt = "head-good", "Pass"
            sub_txt = f"No {defect_label} detected ({record['detection_score']:.0%} confidence)"
    elif status == "segmentation_failure":
        area = record.get("glove_area")
        area_note = f" (glove_area = {area}px, needs >= {MIN_GLOVE_AREA}px)" if area is not None else ""
        head_cls, main_txt = "head-warn", "Segmentation failed"
        sub_txt = f"The glove silhouette could not be reliably found{area_note}"
    elif status == "not_implemented":
        implemented, total = implemented_detectors()
        head_cls, main_txt = "head-warn", "Detector not implemented"
        sub_txt = (
            f"No detector registered yet for '{record['defect_name']}'. "
            f"{len(implemented)} of {total} detectors are currently implemented"
            + (f": {', '.join(implemented)}." if implemented else ".")
        )
    elif status == "detector_failure":
        head_cls, main_txt = "head-warn", "Detector failed"
        sub_txt = "The detector raised an error while running on this image."
    else:
        head_cls, main_txt = "head-warn", f"Unexpected status: {status}"
        sub_txt = ""

    with st.container(border=True, key="resultcard"):
        st.markdown(
            f'<div class="result-head {head_cls}"><div class="main">{main_txt}</div>'
            f'<div class="sub">{sub_txt}</div></div>',
            unsafe_allow_html=True,
        )

        if status == "detector_failure":
            with st.expander("Error details"):
                st.code(record.get("error") or "unknown error")

        if original_bgr is not None:
            overlay = draw_rich_overlay(original_bgr, record)

            st.image(overlay, channels="BGR", use_container_width=True)

            if processed is not None and segmentation is not None:
                p1, p2, p3, p4 = st.columns(4)
                for col, (label, img, channels) in zip(
                    (p1, p2, p3, p4),
                    [
                        ("Original", original_bgr, "BGR"),
                        ("Grayscale", processed["gray_enhanced"], "GRAY"),
                        ("Glove mask", segmentation["glove_mask"], "GRAY"),
                        ("Overlay", overlay, "BGR"),
                    ],
                ):
                    with col:
                        st.markdown(f'<div class="panel-caption">{label}</div>', unsafe_allow_html=True)
                        st.image(img, channels=channels, use_container_width=True)

            if status == "success":
                area_pct = record.get("measurements", {}).get("area_pct")

                m1, m2 = st.columns(2)
                m1.metric("Detection Score", f"{record['detection_score']:.1%}")
                m2.metric("Defect Area", f"{area_pct:.2f}%" if area_pct is not None else "n/a")

            with st.expander("Algorithms used"):
                st.markdown(f"**Detector:** {record['algorithm'] or '— (no detector ran)'}")
                st.markdown("**Segmentation / preprocessing:**")
                for algo in SEGMENTATION_ALGORITHMS:
                    st.markdown(f"- {algo}")

            if processed is not None and segmentation is not None:
                png_bytes = build_composite_png(
                    res_material, res_defect, record, original_bgr, processed, segmentation, overlay,
                )
                st.download_button(
                    "Save figure", png_bytes,
                    file_name=f"{res_material}_{res_defect}_{record['status']}.png",
                    mime="image/png", use_container_width=True,
                )

else:
    st.info("Choose a material, defect and image above, then press **Run detection**.")


# ============================================================
# FULL-DATASET ACCURACY EVALUATION
# ============================================================

def _run_cached_detector(defect_name, processed, segmentation):
    """
    Run one detector against already preprocessed+segmented data,
    mirroring evaluate.evaluate_image()'s detected/threshold logic
    without paying for preprocessing/segmentation again - that part is
    shared across all detectors for a given image (see
    run_full_evaluation), since it doesn't depend on which defect is
    being tested for.
    """
    if segmentation is None or segmentation.get("glove_area", 0) < MIN_GLOVE_AREA:
        return False
    detector_func = load_detector(defect_name)
    if detector_func is None:
        return False
    try:
        result = detector_func(processed, segmentation)
        evaluate.validate_result(result)
    except Exception:
        return False
    return bool(result["detected"]) and float(result["detection_score"]) >= evaluate.DETECTION_THRESHOLD


def run_full_evaluation(progress_bar):
    """
    Cross-test every implemented detector against every image in
    datasets/ (not just the images in that detector's own folder, the
    way evaluate.py's run does). Ground truth for a given
    (detector, image) pair is just "does the image's own folder name
    equal the detector's defect name" - see the caveat shown above the
    resulting table for what that does and doesn't capture.

    Returns (counts, total_images) where counts is
    {defect_name: {"TP", "FP", "FN", "TN"}} for every implemented
    detector.
    """
    images = list(evaluate.discover_images())
    total_images = len(images)

    # Preprocessing + segmentation is identical no matter which
    # detector runs next, so it's computed once per image here and
    # reused for all len(implemented) detector calls on that image,
    # instead of redoing it once per detector.
    cache = {}
    for i, (material, defect_folder, image_path) in enumerate(images):
        try:
            image = load_image(image_path)
            processed = preprocess_image(image)
            segmentation = segment_glove(processed)
        except Exception:
            processed, segmentation = None, None
        cache[image_path] = (processed, segmentation)
        progress_bar.progress(
            (i + 1) / total_images * 0.3,
            text=f"Preprocessing {i + 1}/{total_images}: {os.path.basename(image_path)}",
        )

    implemented, _ = implemented_detectors()
    counts = {d: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for d in implemented}

    total_steps = max(1, len(implemented) * total_images)
    step = 0
    for d in implemented:
        for material, defect_folder, image_path in images:
            processed, segmentation = cache[image_path]
            predicted = _run_cached_detector(d, processed, segmentation)
            actual = defect_folder == d

            if predicted and actual:
                counts[d]["TP"] += 1
            elif predicted and not actual:
                counts[d]["FP"] += 1
            elif not predicted and actual:
                counts[d]["FN"] += 1
            else:
                counts[d]["TN"] += 1

            step += 1
            progress_bar.progress(
                0.3 + (step / total_steps) * 0.7,
                text=f"Testing {d}: {step}/{total_steps} image-detector pairs",
            )

    progress_bar.progress(1.0, text=f"Done - {len(implemented)} detectors x {total_images} images")
    return counts, total_images


def _accuracy_row(name, c):
    n = c["TP"] + c["FP"] + c["FN"] + c["TN"]
    accuracy = (c["TP"] + c["TN"]) / n if n else None
    precision = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) > 0 else None
    recall = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "Detector": name,
        "TP": c["TP"],
        "FP": c["FP"],
        "FN": c["FN"],
        "TN": c["TN"],
        "Accuracy": f"{accuracy:.0%}" if accuracy is not None else "n/a",
        "Precision": f"{precision:.0%}" if precision is not None else "n/a",
        "Recall": f"{recall:.0%}" if recall is not None else "n/a",
        "F1": f"{f1:.0%}" if f1 is not None else "n/a",
    }


with st.expander("Accuracy evaluation (full dataset)", expanded=False):
    st.caption(
        "Ground truth comes from dataset folder labels. Images may contain "
        "unlabelled secondary defects, which appear as false positives "
        "(e.g. a tearing image might also genuinely be dirty)."
    )

    if st.button("Run full evaluation", use_container_width=True):
        progress_bar = st.progress(0.0, text="Starting...")
        counts, total_images = run_full_evaluation(progress_bar)
        st.session_state["full_eval_counts"] = counts
        st.session_state["full_eval_total_images"] = total_images

    counts = st.session_state.get("full_eval_counts")
    if counts:
        rows = [_accuracy_row(name.replace("_", " "), c) for name, c in sorted(counts.items())]

        overall = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for c in counts.values():
            for k in overall:
                overall[k] += c[k]
        rows.append(_accuracy_row("Overall", overall))

        df = pd.DataFrame(rows)
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={"Detector": st.column_config.TextColumn("Detector", width="medium")},
        )

        st.download_button(
            "Download CSV", df.to_csv(index=False).encode("utf-8"),
            file_name="accuracy_evaluation.csv", mime="text/csv", use_container_width=True,
        )
