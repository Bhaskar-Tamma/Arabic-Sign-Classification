import streamlit as st
import numpy as np
from PIL import Image
import io
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Arabic Sign Language Detector",
    page_icon="🤟",
    layout="centered"
)


# ============================================================
# MODEL CONFIGURATION
# ONLY H5 MODEL IS USED
# ============================================================

MODEL_PATH = "arabic_sign_language_model.h5"


# ============================================================
# ARABIC LETTER MAP
# ============================================================

ARABIC_GLYPHS = {
    "ain": "ع",
    "al": "ال",
    "aleff": "أ",
    "bb": "ب",
    "dal": "د",
    "dha": "ظ",
    "dhad": "ض",
    "fa": "ف",
    "gaaf": "ق",
    "ghain": "غ",
    "ha": "ه",
    "haa": "ح",
    "jeem": "ج",
    "kaaf": "ك",
    "khaa": "خ",
    "la": "لا",
    "laam": "ل",
    "meem": "م",
    "nun": "ن",
    "ra": "ر",
    "saad": "ص",
    "seen": "س",
    "sheen": "ش",
    "ta": "ط",
    "taa": "ت",
    "thaa": "ث",
    "thal": "ذ",
    "toot": "ة",
    "waw": "و",
    "ya": "ي",
    "yaa": "ي",
    "zay": "ز"
}


# ============================================================
# CLASS LABELS
# ============================================================

CLASS_LABELS = [
    "ain",
    "al",
    "aleff",
    "bb",
    "dal",
    "dha",
    "dhad",
    "fa",
    "gaaf",
    "ghain",
    "ha",
    "haa",
    "jeem",
    "kaaf",
    "khaa",
    "la",
    "laam",
    "meem",
    "nun",
    "ra",
    "saad",
    "seen",
    "sheen",
    "ta",
    "taa",
    "thaa",
    "thal",
    "toot",
    "waw",
    "ya",
    "yaa",
    "zay"
]


# ============================================================
# SIMPLE UI STYLING
# No HTML is used for the visible output.
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0d0d14;
    }

    [data-testid="stFileUploader"] {
        background-color: #13121c;
        border-radius: 14px;
        padding: 10px;
    }

    .stButton > button {
        background-color: #e8c97e;
        color: #0d0d14;
        border: none;
        border-radius: 30px;
        font-weight: 600;
        width: 100%;
        padding: 0.6rem 1rem;
    }

    .stButton > button:hover {
        background-color: #f0d890;
        color: #0d0d14;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title("🤟 Arabic Sign Language Detector")

st.caption(
    "MobileNet Image Classifier • 32 Arabic Sign Classes"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Settings")

    st.divider()

    st.write("**Model:**")
    st.write("MobileNet")

    st.write("**Model format:**")
    st.write("H5")

    st.write("**Model file:**")
    st.code("arabic_sign_language_model.h5")

    st.divider()

    st.subheader("Classes (32)")

    for i, label in enumerate(CLASS_LABELS, start=1):

        glyph = ARABIC_GLYPHS.get(label, "")

        st.write(
            f"{i}. {glyph}  {label}"
        )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload a hand-sign image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WEBP"
)


# ============================================================
# BEFORE IMAGE UPLOAD
# ============================================================

if uploaded is None:

    st.info(
        "Upload a hand-sign image above to classify the Arabic sign."
    )

    st.write(
        "After uploading the image, click **Classify Sign** "
        "to see the predicted Arabic letter and confidence."
    )


# ============================================================
# AFTER IMAGE UPLOAD
# ============================================================

if uploaded is not None:

    # Read image
    pil_img = Image.open(
        io.BytesIO(uploaded.read())
    )

    st.subheader("Uploaded Image")

    st.image(
        pil_img,
        width="stretch"
    )

    st.caption(
        f"File: {uploaded.name}"
    )


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not os.path.exists(MODEL_PATH):

        st.error(
            "Model file not found."
        )

        st.warning(
            "Please make sure "
            "`arabic_sign_language_model.h5` "
            "is in the same folder as `app.py`."
        )

    else:

        st.divider()


        # ====================================================
        # CLASSIFY BUTTON
        # ====================================================

        classify = st.button(
            "🔍  Classify Sign"
        )


        if classify:

            with st.spinner(
                "Classifying sign..."
            ):

                try:

                    # ========================================
                    # LOAD MODEL
                    # ========================================

                    import tensorflow as tf

                    model = tf.keras.models.load_model(
                        MODEL_PATH,
                        compile=False
                    )


                    # ========================================
                    # PREPROCESS IMAGE
                    # ========================================

                    img = pil_img.convert(
                        "RGB"
                    )

                    img = img.resize(
                        (224, 224)
                    )

                    image_array = np.array(
                        img,
                        dtype=np.float32
                    )

                    image_array = (
                        image_array / 255.0
                    )

                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )


                    # ========================================
                    # PREDICTION
                    # ========================================

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]


                    # ========================================
                    # TOP PREDICTION
                    # ========================================

                    top_index = int(
                        np.argmax(predictions)
                    )

                    top_label = CLASS_LABELS[
                        top_index
                    ]

                    top_confidence = float(
                        predictions[top_index]
                    )

                    top_glyph = ARABIC_GLYPHS.get(
                        top_label,
                        ""
                    )


                    # ========================================
                    # TOP 5 PREDICTIONS
                    # ========================================

                    top5_indices = np.argsort(
                        predictions
                    )[::-1][:5]


                    top5 = []

                    for index in top5_indices:

                        label = CLASS_LABELS[
                            index
                        ]

                        confidence = float(
                            predictions[index]
                        )

                        glyph = ARABIC_GLYPHS.get(
                            label,
                            ""
                        )

                        top5.append(
                            (
                                glyph,
                                label,
                                confidence
                            )
                        )


                    # ========================================
                    # CLASSIFICATION RESULT
                    # ========================================

                    st.divider()

                    st.subheader(
                        "Classification Result"
                    )


                    # Main Arabic character
                    st.markdown(
                        f"# {top_glyph}"
                    )


                    # Class name
                    st.markdown(
                        f"### {top_label.upper()}"
                    )


                    # Confidence
                    confidence_percentage = (
                        top_confidence * 100
                    )

                    st.write(
                        f"**Confidence: "
                        f"{confidence_percentage:.1f}%**"
                    )


                    # Confidence progress bar
                    st.progress(
                        min(
                            max(top_confidence, 0.0),
                            1.0
                        )
                    )


                    # Confidence explanation
                    if top_confidence >= 0.70:

                        st.success(
                            "High confidence prediction"
                        )

                    elif top_confidence >= 0.40:

                        st.warning(
                            "Moderate confidence prediction"
                        )

                    else:

                        st.warning(
                            "Low confidence prediction. "
                            "Consider uploading a clearer image."
                        )


                    # ========================================
                    # TOP PREDICTIONS
                    # ========================================

                    st.subheader(
                        "Top Predictions"
                    )

                    for glyph, label, confidence in top5:

                        percentage = (
                            confidence * 100
                        )

                        col1, col2 = st.columns(
                            [3, 1]
                        )

                        with col1:

                            st.write(
                                f"{glyph}  "
                                f"**{label.upper()}**"
                            )

                        with col2:

                            st.write(
                                f"**{percentage:.1f}%**"
                            )


                        # Small confidence bar
                        st.progress(
                            min(
                                max(confidence, 0.0),
                                1.0
                            )
                        )


                except Exception as e:

                    st.error(
                        "Unable to classify the image."
                    )

                    st.exception(e)