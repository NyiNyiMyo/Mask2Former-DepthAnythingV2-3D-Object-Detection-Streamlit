import time
import numpy as np
import streamlit as st
from PIL import Image
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
        file_path = hf_hub_download(repo_id="NyiNyiMyo/mask2former_coco_tiny", filename="mask2former_coco_tiny.onnx")
        data_path = hf_hub_download(repo_id="NyiNyiMyo/mask2former_coco_tiny", filename="mask2former_coco_tiny.onnx.data")

    session = onnxruntime.InferenceSession(file_path, providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
    
    session2 = onnxruntime.InferenceSession("depth_anything_v2_vits.onnx", providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
    return session, session2

session, session2 = load_hf_model_file()

#--- COCO categories list 133 ---
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 0, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 4, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 5, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 6, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 7, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 8, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 9, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 10, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 11, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 12, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 13, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 14, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 15, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 16, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 17, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 18, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 19, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 20, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 21, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 22, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 23, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 24, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 25, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 26, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 27, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 28, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 29, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 30, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 32, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 33, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 34, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 35, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 36, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 37, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 38, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 39, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 40, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 41, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 42, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 43, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 44, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 45, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 46, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 47, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 48, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 49, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 50, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 51, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 52, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 53, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 54, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 55, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 56, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 57, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 58, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 59, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 60, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 61, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 62, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 63, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 64, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 65, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 66, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 67, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 68, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 69, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 70, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 71, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 72, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 73, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 74, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 75, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 76, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 77, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 78, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 79, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 80, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 81, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 82, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 83, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 84, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 85, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 86, "name": "door"},
    {"color": [218, 88, 184], "isthing": 0, "id": 87, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 88, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 89, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 90, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 91, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 92, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 93, "name": "mirror"},
    {"color": [193, 0, 92], "isthing": 0, "id": 94, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 95, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 96, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 97, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 98, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 99, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 100, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 101, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 102, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 103, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 104, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 105, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 106, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 107, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 108, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 109, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 110, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 111, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 112, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 113, "name": "water"},
    {"color": [183, 121, 142], "isthing": 0, "id": 114, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 115, "name": "window"},
    {"color": [107, 142, 35], "isthing": 0, "id": 116, "name": "tree"},
    {"color": [190, 153, 153], "isthing": 0, "id": 117, "name": "fence"},
    {"color": [146, 139, 141], "isthing": 0, "id": 118, "name": "ceiling"},
    {"color": [70, 130, 180], "isthing": 0, "id": 119, "name": "sky"},
    {"color": [134, 199, 156], "isthing": 0, "id": 120, "name": "cabinet"},
    {"color": [209, 226, 140], "isthing": 0, "id": 121, "name": "table"},
    {"color": [96, 36, 108], "isthing": 0, "id": 122, "name": "floor"},
    {"color": [96, 96, 96], "isthing": 0, "id": 123, "name": "pavement"},
    {"color": [64, 170, 64], "isthing": 0, "id": 124, "name": "mountain"},
    {"color": [152, 251, 152], "isthing": 0, "id": 125, "name": "grass"},
    {"color": [208, 229, 228], "isthing": 0, "id": 126, "name": "dirt"},
    {"color": [206, 186, 171], "isthing": 0, "id": 127, "name": "paper"},
    {"color": [152, 161, 64], "isthing": 0, "id": 128, "name": "food"},
    {"color": [116, 112, 0], "isthing": 0, "id": 129, "name": "building"},
    {"color": [0, 114, 143], "isthing": 0, "id": 130, "name": "rock"},
    {"color": [102, 102, 156], "isthing": 0, "id": 131, "name": "wall"},
    {"color": [250, 141, 255], "isthing": 0, "id": 132, "name": "rug"},
]

COCO_BY_ID = {c["id"]: c for c in COCO_CATEGORIES}

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def project_points(points_3d, fx, fy, cx, cy):
    """Project 3D points into 2D image plane."""
    points_2d = []
    for X, Y, Z in points_3d:
        if Z <= 0:
            points_2d.append((None, None))
        else:
            x = int((X * fx) / Z + cx)
            y = int((Y * fy) / Z + cy)
            points_2d.append((x, y))
    return points_2d

def draw_cube(img, box_3d_2d, color=(0,255,0), thickness=2):
    """Draw cube given its projected 2D corner points."""
    # Top face
    for i in range(4):
        cv2.line(img, box_3d_2d[i], box_3d_2d[(i+1)%4], color, thickness, lineType=cv2.LINE_AA)
    # Bottom face
    for i in range(4, 8):
        cv2.line(img, box_3d_2d[i], box_3d_2d[4 + (i+1)%4], color, thickness, lineType=cv2.LINE_AA)
    # Vertical edges
    for i in range(4):
        cv2.line(img, box_3d_2d[i], box_3d_2d[i+4], color, thickness, lineType=cv2.LINE_AA)
    return img

def draw_slash_hatch_slanted(img, quad_pts, color=(0,255,255), spacing=10, thickness=1, direction=1, alpha=0.5):
    """
    Draw slash hatch lines inside a quad with optional transparency.
    """
    overlay = img.copy()  # temporary layer for transparency
    quad_pts = np.array(quad_pts, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(quad_pts)

    # Determine diagonal vector for slash direction
    if direction == 1:
        dir_vec = np.array([1, -1], dtype=np.float32)  # /
    else:
        dir_vec = np.array([1, 1], dtype=np.float32)   # \

    dir_vec /= np.linalg.norm(dir_vec)  # normalize
    perp_vec = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32)

    num_lines = int((w + h) / spacing) + 2
    center = np.array([x + w/2, y + h/2])

    for i in range(-num_lines, num_lines):
        offset = perp_vec * (i * spacing)
        p1 = center + offset - dir_vec * max(w, h)
        p2 = center + offset + dir_vec * max(w, h)

        line_pts = np.linspace(p1, p2, 100)
        inside = [tuple(pt.astype(int)) for pt in line_pts if cv2.pointPolygonTest(quad_pts, pt, False) >= 0]
        if len(inside) >= 2:
            for a, b in zip(inside[:-1], inside[1:]):
                cv2.line(overlay, a, b, color, thickness, lineType=cv2.LINE_AA)

    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

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
    pil_image = Image.open(uploaded_file)
    
    # Convert to OpenCV BGR format
    input_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
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

        # -------------------------
        # BUILD PANOPTIC MAP & MASKS
        # -------------------------
        panoptic_map = np.zeros((H, W), dtype=np.int32)
        score_order = np.argsort(-scores)
        things_instances = []
    
        # Loop through detected instances in order of confidence
        for idx in score_order:
            cls = labels[idx]
            score = scores[idx]
    
            # FIXED: Extract individual 2D mask (96x96) and resize it safely to (W, H)
            single_logits = mask_logits_filtered[idx]
            resized_logits = cv2.resize(
                single_logits,
                (W, H),
                interpolation=cv2.INTER_LINEAR
            )
    
            mask = sigmoid(resized_logits) > 0.5
    
            if mask.sum() == 0:
                continue
    
            # Set unique mapping ID (1-indexed to keep 0 as empty background)
            panoptic_map[mask] = cls + 1
            if cls < len(COCO_CATEGORIES) and COCO_CATEGORIES[cls].get("isthing", 0) == 1:
                things_instances.append((cls, mask, score))
    
        # -------------------------
        # COLORIZE (PANOPTIC)
        # -------------------------
        overlay = input_image.copy()
        alpha = 0.7  # Adjusted slightly for cleaner blending visibility
    
        for val in np.unique(panoptic_map):
            if val == 0:  # Skip background
                continue
    
            mask = panoptic_map == val
            cls_idx = val - 1
    
            if cls_idx < len(COCO_CATEGORIES):
                color = COCO_CATEGORIES[cls_idx]["color"]
            else:
                color = [0, 255, 0] # Default fallback color
    
            # OpenCV reads colors as BGR. Reverse RGB definition to match BGR
            bgr_color = np.array(color[::-1], dtype=np.uint8)
    
            overlay[mask] = (
                alpha * bgr_color + (1 - alpha) * overlay[mask]
            ).astype(np.uint8)
    
        orig_h, orig_w = input_image.shape[:2]
    
        # Camera intrinsics (simple guess)
        fx = fy = orig_w
        cx = orig_w / 2.0
        cy = orig_h / 2.0
    
        depth = outputs2[0].squeeze().astype(np.uint8)
    
        depth = cv2.resize(depth, (orig_w, orig_h))
    
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_norm = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    
        # -------------------------
        # DRAW BOXES (Things only)
        # -------------------------
        for cls, mask, score in things_instances:
            ys, xs = np.where(mask)
    
            if len(xs) == 0:
                continue
    
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()
    
            if cls < len(COCO_CATEGORIES):
                color = COCO_CATEGORIES[cls]["color"]
                label_name = COCO_CATEGORIES[cls]["name"]
            else:
                color = [0, 255, 0]
                label_name = f"Class {cls}"
    
            bgr_color = tuple(int(c) for c in color[::-1])
    
            # Render bounding boxes and text layers onto target overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr_color, 2)
            cv2.putText(
                overlay,
                f"{label_name} {score:.2f}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                bgr_color,
                2,
                cv2.LINE_AA
            )
    
            depth_patch = depth[y1:y2, x1:x2]
    
            if depth_patch.size == 0:
                continue
    
            obj_depth = np.median(depth_patch) + 1e-6
    
            # Width & height from bbox pixels -> meters
            box_width = (x2 - x1) / fx * obj_depth
            box_height = (y2 - y1) / fy * obj_depth
    
            # Depth from min/max inside bbox (depth variation)
            margin = 0.1
            depth_mask = (depth_patch > obj_depth - margin) & (depth_patch < obj_depth + margin)
            if np.sum(depth_mask) > 0:
                filtered_depths = depth_patch[depth_mask]
                min_d = np.percentile(filtered_depths, 5)
                max_d = np.percentile(filtered_depths, 95)
                box_depth = max_d - min_d
                if box_depth < 1e-6:
                    box_depth = box_width * 0.5
            else:
                box_depth = box_width * 0.5
    
            scale_xy = 1
            scale_z = 1
    
            box_width *= scale_xy
            box_height *= scale_xy
            box_depth *= scale_z
    
            # 3D cube corners (centered at bbox center)
            center_x = ((x1 + x2) / 2 - cx) / fx * obj_depth
            center_y = ((y1 + y2) / 2 - cy) / fy * obj_depth
    
            # --- front pixel corners (TL, TR, BR, BL) ---
            u_f = np.array([x1, x2, x2, x1], dtype=np.float32)
            v_f = np.array([y1, y1, y2, y2], dtype=np.float32)
            Zf = float(obj_depth)
    
            # Convert front pixel corners -> camera coords
            Xf = ((u_f - cx) / fx) * Zf
            Yf = ((v_f - cy) / fy) * Zf
            front_corners = [(float(Xf[i]), float(Yf[i]), Zf) for i in range(4)]
    
            # --- back face: shift in 3D space ---
            back_shift_x = box_depth * 0.5
            back_shift_z = box_depth
    
            Xb = Xf + back_shift_x
            Yb = Yf.copy()
            Zb = Zf + back_shift_z
            back_corners = [(float(Xb[i]), float(Yb[i]), float(Zb)) for i in range(4)]
    
            corners_3d = front_corners + back_corners
            ordered = [corners_3d[i] for i in [0,1,2,3,4,5,6,7]]
    
            # Project 3D points back to 2D image coordinates
            pts_2d = project_points(ordered, fx, fy, cx, cy)
            if any(p[0] is None for p in pts_2d):
                print(f"Skipping box due to invalid projection points for box {(x1, y1, x2, y2)}")
                continue
    
            # Draw the Wireframe Cube
            overlay = draw_cube(overlay, pts_2d, color=bgr_color, thickness=2)
    
            # Draw Corner Node Points
            for (x, y) in pts_2d:
                if x is not None and y is not None:
                    cv2.circle(overlay, (x, y), 4, (0, 165, 255), -1)
    
            # Face textures mapping sets
            top_face_pts = [pts_2d[i] for i in [0,4,5,1]]
            bottom_face_pts = [pts_2d[i] for i in [2,6,7,3]]
            side_face_pts = [pts_2d[i] for i in [5,1,2,6]]
    
            # Render textured perspective hatch layers
            overlay = draw_slash_hatch_slanted(overlay, bottom_face_pts, color=(0,0,255), spacing=8, thickness=2, direction=-1, alpha=0.3)
            overlay = draw_slash_hatch_slanted(overlay, side_face_pts, color=(255,255,0), spacing=8, thickness=2, direction=-1, alpha=0.3)

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    image_col1, image_col2 = st.columns(2)
    with image_col1:
        st.markdown("**Original Image**")
        st.image(
            cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )
    with image_col2:
        st.markdown("**3D Object Detection**")
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
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
