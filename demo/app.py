from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import streamlit as st


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v4"
    / "epoch_5.pt"
)

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        color: #8b8b93;
        font-size: 0.98rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }

    .image-label {
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .muted {
        color: #8b8b93;
        font-size: 0.82rem;
    }

    .sidebar-card {
        padding: 0.75rem;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.08);
        margin-bottom: 0.7rem;
    }

    .legend {
        display: flex;
        justify-content: center;
        gap: 1.8rem;
        margin: 0.7rem 0 1rem 0;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.85rem;
    }

    .legend-dot {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        display: inline-block;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">'
    '3D Brain Tumor MRI Segmentation'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">'
    'Multimodal 3D U-Net inference for volumetric brain tumor segmentation'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# VALIDATION
# ============================================================

if not RAW_DIR.exists():

    st.error(
        f"BraTS dataset not found:\n\n{RAW_DIR}"
    )

    st.stop()


if not MODEL_PATH.exists():

    st.error(
        f"V4 checkpoint not found:\n\n{MODEL_PATH}"
    )

    st.stop()


# ============================================================
# PATIENT DISCOVERY
# ============================================================

@st.cache_data
def get_patients():

    return sorted(
        path.name
        for path in RAW_DIR.iterdir()
        if path.is_dir()
    )


patients = get_patients()

if not patients:

    st.error(
        "No BraTS patient directories found."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Inference")

    patient_id = st.selectbox(
        "Patient",
        patients,
    )

    modality = st.selectbox(
        "MRI Modality",
        [
            "t1n",
            "t1c",
            "t2w",
            "t2f",
        ],
        format_func=lambda x: {
            "t1n": "T1 Native",
            "t1c": "T1 Contrast",
            "t2w": "T2 Weighted",
            "t2f": "T2 FLAIR",
        }[x],
    )

    st.divider()

    run_prediction = st.button(
        "Run 3D Segmentation",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-card">
            <b>Model</b><br>
            <span class="muted">3D U-Net V4</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <b>Input</b><br>
            <span class="muted">T1n + T1c + T2w + T2f</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <b>Patch Size</b><br>
            <span class="muted">64 × 128 × 128</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <b>Threshold</b><br>
            <span class="muted">0.50</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PATIENT PATHS
# ============================================================

patient_dir = RAW_DIR / patient_id

modality_path = (
    patient_dir
    / f"{patient_id}-{modality}.nii.gz"
)

segmentation_path = (
    patient_dir
    / f"{patient_id}-seg.nii.gz"
)


# ============================================================
# LOAD NIFTI
# ============================================================

@st.cache_data
def load_nifti(path):

    image = nib.load(path)

    return image.get_fdata(
        dtype=np.float32
    )


try:

    mri_volume = load_nifti(
        modality_path
    )

except Exception as e:

    st.error(
        f"Unable to load MRI volume:\n\n{e}"
    )

    st.stop()


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

ground_truth = None

if segmentation_path.exists():

    try:

        ground_truth = load_nifti(
            segmentation_path
        )

        ground_truth = (
            ground_truth > 0
        ).astype(np.uint8)

    except Exception:

        ground_truth = None


# ============================================================
# VOLUME INFORMATION
# ============================================================

height, width, depth = mri_volume.shape


# ============================================================
# SESSION STATE
# ============================================================

if "selected_patient" not in st.session_state:

    st.session_state["selected_patient"] = patient_id


if "slice_index" not in st.session_state:

    st.session_state["slice_index"] = (
        depth // 2
    )


if "prediction" not in st.session_state:

    st.session_state["prediction"] = None


if "probability" not in st.session_state:

    st.session_state["probability"] = None


if "tumor_slices" not in st.session_state:

    st.session_state["tumor_slices"] = np.array(
        [],
        dtype=np.int64,
    )


if "patches" not in st.session_state:

    st.session_state["patches"] = None


if "prediction_patient" not in st.session_state:

    st.session_state["prediction_patient"] = None


# ============================================================
# RESET WHEN PATIENT CHANGES
# ============================================================

if (
    st.session_state["selected_patient"]
    != patient_id
):

    st.session_state["selected_patient"] = (
        patient_id
    )

    st.session_state["prediction"] = None

    st.session_state["probability"] = None

    st.session_state["tumor_slices"] = np.array(
        [],
        dtype=np.int64,
    )

    st.session_state["patches"] = None

    st.session_state["prediction_patient"] = None

    st.session_state["slice_index"] = (
        depth // 2
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_for_display(image):

    image = image.astype(
        np.float32
    )

    min_value = image.min()
    max_value = image.max()

    if max_value - min_value < 1e-6:

        return np.zeros_like(image)

    return (
        image - min_value
    ) / (
        max_value - min_value
    )


def calculate_dice(
    prediction,
    target,
):

    prediction = prediction.astype(
        np.float32
    )

    target = target.astype(
        np.float32
    )

    intersection = (
        prediction * target
    ).sum()

    return (
        2.0 * intersection + 1e-6
    ) / (
        prediction.sum()
        + target.sum()
        + 1e-6
    )


def create_overlay(
    mri_slice,
    prediction_slice,
    ground_truth_slice=None,
):

    # Base MRI as RGB
    base = np.stack(
        [mri_slice] * 3,
        axis=-1,
    )

    prediction_mask = (
        prediction_slice > 0
    )

    if ground_truth_slice is not None:

        ground_truth_mask = (
            ground_truth_slice > 0
        )

    else:

        ground_truth_mask = (
            np.zeros_like(
                prediction_mask,
                dtype=bool,
            )
        )

    # --------------------------------------------------------
    # RED = Prediction only
    # GREEN = Ground truth only
    # YELLOW = Overlap
    # --------------------------------------------------------

    prediction_only = (
        prediction_mask
        & ~ground_truth_mask
    )

    ground_truth_only = (
        ground_truth_mask
        & ~prediction_mask
    )

    overlap = (
        prediction_mask
        & ground_truth_mask
    )

    # Alpha blending
    alpha = 0.65

    # Prediction → red
    base[prediction_only] = (
        (1 - alpha)
        * base[prediction_only]
        + alpha
        * np.array([1.0, 0.0, 0.0])
    )

    # Ground truth → green
    base[ground_truth_only] = (
        (1 - alpha)
        * base[ground_truth_only]
        + alpha
        * np.array([0.0, 1.0, 0.0])
    )

    # Overlap → yellow
    base[overlap] = (
        (1 - alpha)
        * base[overlap]
        + alpha
        * np.array([1.0, 1.0, 0.0])
    )

    return np.clip(
        base,
        0,
        1,
    )


# ============================================================
# CACHED MODEL
# ============================================================

@st.cache_resource
def get_model():

    from predict import load_model

    return load_model(
        MODEL_PATH
    )


# ============================================================
# RUN 3D INFERENCE
# ============================================================

if run_prediction:

    with st.spinner(
        "Running 3D V4 segmentation..."
    ):

        try:

            from predict import (
                load_mri,
                predict_volume,
            )

            model = get_model()

            mri = load_mri(
                patient_dir
            )

            (
                prediction,
                probability,
                patches,
            ) = predict_volume(
                model,
                mri,
            )

            # -----------------------------------------------
            # Store prediction
            # -----------------------------------------------

            st.session_state["prediction"] = (
                prediction
            )

            st.session_state["probability"] = (
                probability
            )

            st.session_state["patches"] = (
                patches
            )

            st.session_state[
                "prediction_patient"
            ] = patient_id

            # -----------------------------------------------
            # Prediction = (Z, H, W)
            # -----------------------------------------------

            tumor_slices = np.where(
                prediction.sum(
                    axis=(1, 2)
                ) > 0
            )[0]

            st.session_state[
                "tumor_slices"
            ] = tumor_slices

            # -----------------------------------------------
            # Jump to useful tumor slice
            # -----------------------------------------------

            if len(tumor_slices) > 0:

                middle = (
                    len(tumor_slices) // 2
                )

                st.session_state[
                    "slice_index"
                ] = int(
                    tumor_slices[middle]
                )

            else:

                st.session_state[
                    "slice_index"
                ] = depth // 2

            st.rerun()

        except Exception as e:

            st.error(
                f"Prediction failed:\n\n{e}"
            )


# ============================================================
# SLICE NAVIGATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'Slice Navigation'
    '</div>',
    unsafe_allow_html=True,
)


new_slice = st.slider(
    "Axial Slice",
    min_value=0,
    max_value=depth - 1,
    value=int(
        st.session_state["slice_index"]
    ),
)

st.session_state["slice_index"] = (
    new_slice
)

slice_index = new_slice


# ============================================================
# TUMOR SLICE NAVIGATION
# ============================================================

tumor_slices = st.session_state[
    "tumor_slices"
]


if len(tumor_slices) > 0:

    st.caption(
        f"Model detected tumor on "
        f"{len(tumor_slices)} of {depth} slices."
    )

    nav1, nav2, nav3 = st.columns(
        [1, 1, 1]
    )

    with nav1:

        previous = tumor_slices[
            tumor_slices < slice_index
        ]

        if len(previous) > 0:

            if st.button(
                "← Previous Tumor",
                use_container_width=True,
            ):

                st.session_state[
                    "slice_index"
                ] = int(previous[-1])

                st.rerun()

        else:

            st.button(
                "← Previous Tumor",
                disabled=True,
                use_container_width=True,
            )

    with nav2:

        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding-top:0.45rem;
            ">
                <b>Slice</b><br>
                {slice_index + 1} / {depth}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav3:

        following = tumor_slices[
            tumor_slices > slice_index
        ]

        if len(following) > 0:

            if st.button(
                "Next Tumor →",
                use_container_width=True,
            ):

                st.session_state[
                    "slice_index"
                ] = int(following[0])

                st.rerun()

        else:

            st.button(
                "Next Tumor →",
                disabled=True,
                use_container_width=True,
            )


# ============================================================
# CURRENT MRI SLICE
# ============================================================

mri_slice = mri_volume[
    :,
    :,
    slice_index,
]

mri_slice = normalize_for_display(
    mri_slice
)


# ============================================================
# CURRENT GROUND TRUTH SLICE
# ============================================================

if ground_truth is not None:

    gt_slice = ground_truth[
        :,
        :,
        slice_index,
    ]

else:

    gt_slice = None


# ============================================================
# PATIENT OVERVIEW
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'Patient Overview'
    '</div>',
    unsafe_allow_html=True,
)


info1, info2, info3, info4 = st.columns(4)


with info1:

    st.metric(
        "Patient",
        patient_id,
    )


with info2:

    st.metric(
        "Volume",
        f"{height} × {width} × {depth}",
    )


with info3:

    st.metric(
        "Slice",
        f"{slice_index + 1} / {depth}",
    )


with info4:

    if ground_truth is not None:

        st.metric(
            "GT Tumor Voxels",
            f"{int(ground_truth.sum()):,}",
        )

    else:

        st.metric(
            "Ground Truth",
            "Unavailable",
        )


# ============================================================
# MRI + GROUND TRUTH
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'MRI & Ground Truth'
    '</div>',
    unsafe_allow_html=True,
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        f'<div class="image-label">'
        f'{modality.upper()} MRI'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.image(
        mri_slice,
        clamp=True,
        width=340,
    )


with col2:

    st.markdown(
        '<div class="image-label">'
        'Ground Truth'
        '</div>',
        unsafe_allow_html=True,
    )

    if gt_slice is not None:

        gt_display = (
            gt_slice.astype(
                np.uint8
            ) * 255
        )

        st.image(
            gt_display,
            clamp=True,
            width=340,
        )

        st.caption(
            f"{int(gt_slice.sum()):,} tumor pixels "
            "on this slice"
        )

    else:

        st.info(
            "Ground-truth segmentation "
            "is not available."
        )


# ============================================================
# V4 SEGMENTATION
# ============================================================

prediction = st.session_state[
    "prediction"
]

probability = st.session_state[
    "probability"
]


if prediction is not None:

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'V4 Segmentation'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Convert model orientation
    #
    # Prediction:
    # (Z, H, W)
    #
    # NIfTI:
    # (H, W, Z)
    # --------------------------------------------------------

    prediction_display = np.transpose(
        prediction,
        (1, 2, 0),
    )

    probability_display = np.transpose(
        probability,
        (1, 2, 0),
    )

    pred_slice = prediction_display[
        :,
        :,
        slice_index,
    ]

    probability_slice = probability_display[
        :,
        :,
        slice_index,
    ]

    # --------------------------------------------------------
    # Binary prediction display
    # --------------------------------------------------------

    pred_display = (
        pred_slice.astype(
            np.uint8
        ) * 255
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        mri_slice,
        pred_slice,
        gt_slice,
    )

    # --------------------------------------------------------
    # Prediction + Overlay
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            '<div class="image-label">'
            'Predicted Tumor Mask'
            '</div>',
            unsafe_allow_html=True,
        )

        st.image(
            pred_display,
            clamp=True,
            width=340,
        )

        st.caption(
            f"{int(pred_slice.sum()):,} predicted pixels "
            "on this slice"
        )


    with col2:

        st.markdown(
            '<div class="image-label">'
            'Prediction Overlay'
            '</div>',
            unsafe_allow_html=True,
        )

        st.image(
            overlay,
            clamp=True,
            width=340,
        )


    # --------------------------------------------------------
    # Probability Map
    # --------------------------------------------------------

    st.markdown(
        '<div class="image-label">'
        'Model Probability Map'
        '</div>',
        unsafe_allow_html=True,
    )

    probability_col1, probability_col2 = st.columns(
        [1, 2]
    )

    with probability_col1:

        st.image(
            probability_slice,
            clamp=True,
            width=340,
        )

    with probability_col2:

        st.markdown(
            """
            <div style="
                padding: 1rem 0;
                color: #8b8b93;
                font-size: 0.9rem;
                line-height: 1.7;
            ">

            <b>How to read the overlay</b>

            <br><br>

            <span style="color:#ff3333;">
            ●
            </span>
            <b> Red</b> — model prediction

            <br>

            <span style="color:#33cc33;">
            ●
            </span>
            <b> Green</b> — ground truth

            <br>

            <span style="color:#ffff22;">
            ●
            </span>
            <b> Yellow</b> — prediction and ground truth overlap

            <br><br>

            The probability map shows the model's
            confidence before thresholding.

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # CURRENT SLICE METRICS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Current Slice'
        '</div>',
        unsafe_allow_html=True,
    )

    current1, current2, current3 = st.columns(3)


    current_prediction_pixels = int(
        pred_slice.sum()
    )

    current_gt_pixels = (
        int(gt_slice.sum())
        if gt_slice is not None
        else 0
    )


    with current1:

        st.metric(
            "Predicted Pixels",
            f"{current_prediction_pixels:,}",
        )


    with current2:

        st.metric(
            "Ground Truth Pixels",
            f"{current_gt_pixels:,}",
        )


    with current3:

        st.metric(
            "Max Probability",
            f"{probability_slice.max():.3f}",
        )


    # ========================================================
    # VOLUME METRICS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Volume Metrics'
        '</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3, metric4 = st.columns(4)


    predicted_voxels = int(
        prediction.sum()
    )


    if ground_truth is not None:

        prediction_nifti = np.transpose(
            prediction,
            (1, 2, 0),
        )

        dice = calculate_dice(
            prediction_nifti,
            ground_truth,
        )

    else:

        dice = None


    with metric1:

        st.metric(
            "Predicted Tumor Voxels",
            f"{predicted_voxels:,}",
        )


    with metric2:

        st.metric(
            "Tumor Slices",
            len(tumor_slices),
        )


    with metric3:

        if dice is not None:

            st.metric(
                "Volume Dice",
                f"{dice:.4f}",
            )

        else:

            st.metric(
                "Volume Dice",
                "N/A",
            )


    with metric4:

        st.metric(
            "Inference Patches",
            st.session_state.get(
                "patches",
                "N/A",
            ),
        )


else:

    st.divider()

    st.info(
        "Select a patient and click "
        "**Run 3D Segmentation** to generate "
        "a V4 prediction."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Experimental research/portfolio system — "
    "not intended for clinical diagnosis."
)