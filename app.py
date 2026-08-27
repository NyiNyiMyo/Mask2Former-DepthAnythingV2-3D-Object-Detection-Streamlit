import time
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import onnxruntime
import cv2

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="3D Object Detection",
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
def load_hf_model_file():
    with st.spinner(f"Downloading Models from Hugging Face... Please wait."):
        file_path = hf_hub_download(repo_id="NyiNyiMyo/mask2former_coco", filename="mask2former_coco.onnx")
        data_path = hf_hub_download(repo_id="NyiNyiMyo/mask2former_coco", filename="mask2former_coco.onnx.data")
    with st.spinner(f"Downloading Models from Hugging Face... Please wait."):
        file_path2 = hf_hub_download(repo_id="NyiNyiMyo/depth_anything_v2_vitb", filename="depth_anything_v2_vitb.onnx")

    return file_path, file_path2

file_path, file_path2 = load_hf_model_file()

session = onnxruntime.InferenceSession(file_path, providers=['CUDAExecutionProvider',
                                                           'CPUExecutionProvider'])

session2 = onnxruntime.InferenceSession(file_path2, providers=['CUDAExecutionProvider',
                                                           'CPUExecutionProvider'])

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">📟 Makers - Deploy 🚀</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### ℹ️ Model Information")
    st.markdown(
        """
        <div class="sidebar-item">
            <div class="sidebar-label">Models</div>
            <div class="sidebar-value">Mask2Former & DepthAnythingV2</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Task</div>
            <div class="sidebar-value">Monocular 3D Object Detection</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Dataset</div>
            <div class="sidebar-value">COCO</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Framework</div>
            <div class="sidebar-value">ONNX</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### 🎚️ Inference Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )
    st.divider()
    st.caption(
        "3D Object Detection with Panoptic Segmentation Fusion Inference"
    )

# ============================================================
# Header
# ============================================================
st.markdown(
    '<div class="main-title">🧊 3D Object Detection using DepthAnythingV2</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "🏞️Panoptic Segmentation with Depth Estimation🛰️"
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
    "Upload an image",
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
    # input_image = Image.open(uploaded_file).convert("RGB")
    input_image = cv2.imread(uploaded_file)
elif selected_example is not None:
    # input_image = Image.open(selected_example).convert("RGB")
    input_image = cv2.imread(selected_example)

# ============================================================
# Run Inference
# ============================================================
if input_image is not None:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    input_name2 = session2.get_inputs()[0].name

    st.divider()
    st.markdown(
        '<div class="section-title">🎯 Results</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Running Inference..."):
        start_time = time.perf_counter()

        img_resized = cv2.resize(input_image, (384, 384))
        img_input = (img_resized[:, :, ::-1].astype(np.float32) ) / 255
        img_input = (img_input - mean) / std
        img_input = img_input.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, H, W)
    
        outputs = session.run(None, {input_name: img_input})
    
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (518, 518))
        img_input = (img_resized[:, :, ::-1].astype(np.float32) ) / 255
        img_input = (img_input - mean) / std
        img_input = img_input.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, H, W)
    
        outputs2 = session2.run(None, {input_name2: img_input})

        H, W = input_image.shape[:2]
    
        # -------------------------
        # OUTPUTS
        # -------------------------
        cls_logits = outputs[0][0]     # (Q, C+1) -> (100, 134)
        mask_logits_raw = outputs[1][0] # (Q, Hm, Wm) -> (100, 96, 96)
    
        probs = softmax(cls_logits)
    
        # Use classes 0 to 132 (ignoring the last index background class)
        scores = np.max(probs[:, :-1], axis=1)
        labels = np.argmax(probs[:, :-1], axis=1)
    
        keep = scores > confidence
    
        scores = scores[keep]
        labels = labels[keep]
        mask_logits_filtered = mask_logits_raw[keep]

        inference_time = time.perf_counter() - start_time
    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    annotated_image_bgr = input_image
    annotated_image_rgb = annotated_image_bgr[..., ::-1]

    image_col1, image_col2 = st.columns(2)
    with image_col1:
        st.markdown("**Original Image**")
        st.image(
            input_image,
            use_container_width=True,
        )
    with image_col2:
        st.markdown("**3D Object Detection**")
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
    # if result.boxes is not None:
    #     num_instances = len(result.boxes)
    #     if num_instances > 0:
    #         class_ids = (
    #             result.boxes.cls
    #             .cpu()
    #             .numpy()
    #             .astype(int)
    #         )
    #         confidences = (
    #             result.boxes.conf
    #             .cpu()
    #             .numpy()
    #         )
    #         detected_classes = [
    #             result.names[class_id]
    #             for class_id in class_ids
    #         ]
    #         unique_classes = list(
    #             dict.fromkeys(detected_classes)
    #         )
    #         average_confidence = float(
    #             np.mean(confidences)
    #         )
    #     else:
    #         average_confidence = 0.0
    #         unique_classes = []
    # else:
    #     num_instances = 0
    #     average_confidence = 0.0
    #     unique_classes = []

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
    # stat1, stat2, stat3, stat4 = st.columns(4)

    # with stat1:
    #     st.metric(
    #         "Instances",
    #         num_instances,
    #     )
    # with stat2:
    #     st.metric(
    #         "Classes",
    #         len(unique_classes),
    #     )
    # with stat3:
    #     st.metric(
    #         "Avg. Confidence",
    #         f"{average_confidence:.2%}",
    #     )
    # with stat4:
    #     st.metric(
    #         "Inference Time",
    #         f"{inference_time * 1000:.1f} ms",
    #     )

    # ========================================================
    # Detected Classes
    # ========================================================
    # if unique_classes:
    #     st.write("")
    #     st.markdown("**Detected Classes**")
    #     class_text = "  •  ".join(
    #         unique_classes
    #     )
    #     st.info(class_text)
else:
    st.info(
        "Upload an image or select one of the example images."
        # "above to run inference."
    )

# ============================================================
# Model Summary Cards
# ============================================================
card1, card2, card3, card4 = st.columns(4)

with card1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-value">Mask2Former</div>
            <div class="info-label">Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with card2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-value">COCO</div>
            <div class="info-label">Dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with card3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-value">133</div>
            <div class="info-label">Classes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with card4:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-value">Monocular 3D</div>
            <div class="info-label">Object Detection</div>
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
    "DepthAnythingV2 • 3D Object Detection • COCO"
)
