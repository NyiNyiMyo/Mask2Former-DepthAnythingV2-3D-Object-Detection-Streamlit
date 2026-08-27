import time
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="YOLOv26 | SIS Instance Segmentation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown(
    """
    <style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    /* Main title */
    .main-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #28C1C9;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.0rem;
        font-weight: 500;
        color: #8b949e;
        margin-bottom: 1.0rem;
    }
    /* Section headings */
    .section-title {
        font-size: 1.0rem;
        font-weight: 500;
        margin-top: 0.7rem;
        margin-bottom: 0.7rem;
    }
    /* Info cards */
    .info-card {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.05);
        text-align: center;
    }
    .info-value {
        font-size: 1.00rem;
        font-weight: 500;
    }
    .info-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 0.15rem;
    }
    /* Example buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
    }
    /* Sidebar */
    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sidebar-item {
        margin-bottom: 0.65rem;
    }
    .sidebar-label {
        color: #8b949e;
        font-size: 0.8rem;
    }
    .sidebar-value {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Model
# ============================================================
@st.cache_resource
def load_model():
    return YOLO("yolo26s-seg.pt")

model = load_model()

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">ℹ️ Model Information</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### ℹ️ Model Information")
    st.markdown(
        """
        <div class="sidebar-item">
            <div class="sidebar-label">Model</div>
            <div class="sidebar-value">YOLOv26</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Task</div>
            <div class="sidebar-value">Instance Segmentation</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Dataset</div>
            <div class="sidebar-value">Surgical Instruments</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Framework</div>
            <div class="sidebar-value">Ultralytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### 🎛️ Inference Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )
    st.divider()
    st.caption(
        "YOLOv26 Instance Segmentation Inference"
    )

# ============================================================
# Header
# ============================================================
st.markdown(
    '<div class="main-title">🧊 YOLOv26 Instance Segmentation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "🏞️Surgical Instruments Segmentation🛰️"
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# Image Input
# ============================================================
st.markdown(
    '<div class="section-title">🗂️ Input Image</div>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Upload a surgical image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ============================================================
# Example Images
# ============================================================
example_files = [
    "sample1.jpg",
    "sample2.jpg",
    "sample3.jpg",
    "sample4.jpg",
    "sample5.jpg",
    "sample6.jpg",
    "sample7.jpg",
    "sample8.jpg",
    "sample9.jpg",
    "sample10.jpg",
]
available_examples = [
    file for file in example_files
    if __import__("os").path.exists(file)
]
selected_example = None
if available_examples:
    st.markdown(
        '<div class="section-title">🖼️ Try an Example</div>',
        unsafe_allow_html=True,
    )
    example_cols = st.columns(10) # st.columns(len(available_examples))
    for i, example in enumerate(available_examples):
        with example_cols[i]:
            st.image(
                example,
                use_container_width=True,
            )
            if st.button(
                "Select", # f"Use Example {i + 1}",
                key=f"example_{i}",
            ):
                selected_example = example

# ============================================================
# Determine Input Image
# ============================================================
input_image = None
if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
elif selected_example is not None:
    input_image = Image.open(selected_example).convert("RGB")

# ============================================================
# Run Inference
# ============================================================
if input_image is not None:
    st.divider()
    st.markdown(
        '<div class="section-title">🎯 Segmentation Results</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Running YOLOv26 inference..."):
        start_time = time.perf_counter()
        results = model.predict(
            source=input_image,
            conf=confidence,
            save=False,
            verbose=False,
        )
        inference_time = time.perf_counter() - start_time
    result = results[0]
    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    annotated_image_bgr = result.plot()
    annotated_image_rgb = annotated_image_bgr[..., ::-1]

    image_col1, image_col2 = st.columns(2)
    with image_col1:
        st.markdown("**Original Image**")
        st.image(
            input_image,
            use_container_width=True,
        )
    with image_col2:
        st.markdown("**YOLOv26 Segmentation**")
        st.image(
            annotated_image_rgb,
            use_container_width=True,
        )
    # ========================================================
    # Statistics
    # ========================================================
    st.write("")
    st.markdown(
        '<div class="section-title">📊 Inference Statistics</div>',
        unsafe_allow_html=True,
    )
    if result.boxes is not None:
        num_instances = len(result.boxes)
        if num_instances > 0:
            class_ids = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )
            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )
            detected_classes = [
                result.names[class_id]
                for class_id in class_ids
            ]
            unique_classes = list(
                dict.fromkeys(detected_classes)
            )
            average_confidence = float(
                np.mean(confidences)
            )
        else:
            average_confidence = 0.0
            unique_classes = []
    else:
        num_instances = 0
        average_confidence = 0.0
        unique_classes = []

    st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.metric(
            "Instances",
            num_instances,
        )
    with stat2:
        st.metric(
            "Classes",
            len(unique_classes),
        )
    with stat3:
        st.metric(
            "Avg. Confidence",
            f"{average_confidence:.2%}",
        )
    with stat4:
        st.metric(
            "Inference Time",
            f"{inference_time * 1000:.1f} ms",
        )

    # ========================================================
    # Detected Classes
    # ========================================================
    if unique_classes:
        st.write("")
        st.markdown("**Detected Instrument Classes**")
        class_text = "  •  ".join(
            unique_classes
        )
        st.info(class_text)
else:
    st.info(
        "Upload an image or select one of the example images."
        # "above to run YOLOv26 instance segmentation."
    )

# ============================================================
# Model Summary Cards
# ============================================================
card1, card2, card3, card4 = st.columns(4)

with card1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-value">YOLOv26</div>
            <div class="info-label">Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with card2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-value">SIS</div>
            <div class="info-label">Dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with card3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-value">6</div>
            <div class="info-label">Classes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with card4:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-value">Instance</div>
            <div class="info-label">Segmentation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.write("")

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "YOLOv26 • Instance Segmentation • Surgical Instruments"
)
