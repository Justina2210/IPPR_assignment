"""
app.py
------
Streamlit GUI for the Glove Defect Detection System.

Wraps evaluate.py's single-image pipeline for interactive use - one
image at a time, not the whole-dataset batch run. On Run, this calls
evaluate_image() directly, which internally does exactly
preprocess_image() -> segment_glove() -> the one detector matching the
selected defect (loaded via DETECTOR_REGISTRY / load_detector()) - so
the pipeline wiring and the overlay drawing both come from evaluate.py
rather than being duplicated here.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from evaluate import (
    DATASET_ROOT,
    DETECTOR_REGISTRY,
    MIN_GLOVE_AREA,
    draw_overlay,
    evaluate_image,
    load_detector,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

st.set_page_config(page_title="Glove Defect Detection", layout="wide")


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
    evaluate_image() takes a file path (it calls preprocessing.load_image,
    which is cv2.imread under the hood), so an uploaded in-memory file has
    to be written to disk first rather than passed straight through.
    """
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# ============================================================
# RESULT RENDERING
# ============================================================

def render_status_message(record):
    status = record["status"]

    if status == "segmentation_failure":
        area = record.get("glove_area")
        area_note = (
            f" (glove_area = {area}px, needs >= {MIN_GLOVE_AREA}px)"
            if area is not None else ""
        )
        st.warning(
            f"**Segmentation failed** — the glove silhouette could not be "
            f"reliably found in this image{area_note}.\n\n{record.get('error') or ''}"
        )

    elif status == "not_implemented":
        implemented, total = implemented_detectors()
        st.info(
            f"**Detector not implemented yet** for `{record['defect_name']}`. "
            f"{len(implemented)} of {total} detectors are currently implemented"
            + (f": {', '.join(implemented)}." if implemented else ".")
        )

    elif status == "detector_failure":
        st.error("**Detector failed** while running on this image.")
        with st.expander("Error details"):
            st.code(record.get("error") or "unknown error")

    elif status != "success":
        st.error(f"Unexpected status: `{status}`")


def render_result_summary(record):
    detected = record["detected"]
    score = record["detection_score"]
    area_pct = record.get("measurements", {}).get("area_pct")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Detection Result**")
        if detected:
            st.error("❌ DEFECT DETECTED")
        else:
            st.success("✅ PASS")
    with col2:
        st.metric("Detection Score", f"{score:.1%}")
    with col3:
        st.metric("Defect Area", f"{area_pct:.2f}%" if area_pct is not None else "n/a")

    st.caption(f"Algorithm: {record['algorithm']}")


def render_images(original_bgr, record):
    overlay = draw_overlay(
        original_bgr, record["defect_name"], record["detected"],
        record["detection_score"], record["bounding_box"],
    )

    overlay_only = st.toggle("Show overlay only")

    if overlay_only:
        st.image(overlay, channels="BGR", caption="Overlay", use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_bgr, channels="BGR", caption="Original", use_container_width=True)
        with col2:
            st.image(overlay, channels="BGR", caption="Overlay", use_container_width=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Glove Defect Detection")

materials = list_materials()
if not materials:
    st.sidebar.error(f"No material folders found under `{DATASET_ROOT}/`.")
    st.stop()

material = st.sidebar.selectbox("1. Material", materials)

defects = list_defects(material)
if not defects:
    st.sidebar.error(f"No defect folders found under `{DATASET_ROOT}/{material}/`.")
    st.stop()

defect = st.sidebar.selectbox("2. Defect", defects)

st.sidebar.markdown("**3. Image**")
image_source = st.sidebar.radio(
    "Image source", ["Sample from dataset", "Upload my own"], label_visibility="collapsed",
)

image_path = None
uploaded_file = None
display_name = None

if image_source == "Sample from dataset":
    images = list_images(material, defect)
    if not images:
        st.sidebar.warning(f"No images found in `{DATASET_ROOT}/{material}/{defect}/`.")
    else:
        chosen = st.sidebar.selectbox("Image file", images)
        image_path = os.path.join(DATASET_ROOT, material, defect, chosen)
        display_name = chosen
        st.sidebar.image(image_path, caption=chosen, use_container_width=True)
else:
    uploaded_file = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        display_name = uploaded_file.name
        st.sidebar.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

run_clicked = st.sidebar.button(
    "4. Run", type="primary", use_container_width=True,
    disabled=(image_path is None and uploaded_file is None),
)


# ============================================================
# MAIN
# ============================================================

st.title("Glove Defect Detection")
st.caption(f"Material: **{material}**  |  Defect: **{defect}**")

if run_clicked:
    run_path = save_uploaded_file(uploaded_file) if uploaded_file is not None else image_path

    with st.spinner("Running preprocessing -> segmentation -> detector..."):
        record, original_bgr = evaluate_image(run_path, defect)

    if uploaded_file is not None:
        try:
            os.remove(run_path)
        except OSError:
            pass

    st.session_state["last_record"] = record
    st.session_state["last_original"] = original_bgr
    st.session_state["last_display_name"] = display_name

record = st.session_state.get("last_record")
original_bgr = st.session_state.get("last_original")

if record is None:
    st.info("Choose a material, defect and image on the left, then press **Run**.")
else:
    st.subheader(f"Result: {st.session_state.get('last_display_name', record['image_path'])}")
    render_status_message(record)

    if record["status"] == "success":
        render_result_summary(record)

    if original_bgr is not None:
        render_images(original_bgr, record)
