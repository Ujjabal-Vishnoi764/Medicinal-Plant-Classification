import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
from datetime import datetime


st.set_page_config(
    page_title="Medicinal Plant Classification",
    page_icon="🌿",
    layout="centered"
)


# Load trained model
model = joblib.load("model/plant_classifier.pkl")
encoder = joblib.load("model/label_encoder.pkl")

IMG_SIZE = (128, 128)


# Session History
if "history" not in st.session_state:
    st.session_state.history = []


# Plant information
plant_info = {

    "Aloe Vera": {
        "uses": "Traditionally used for skin care and minor skin irritation.",
        "benefits": "Commonly used in moisturizing and soothing skin products.",
        "precautions": "Do not use internally without proper medical advice."
    },

    "Hibiscus": {
        "uses": "Traditionally used in hair care and herbal preparations.",
        "benefits": "Commonly used in traditional herbal practices.",
        "precautions": "Avoid medicinal use if you have allergies or are unsure about interactions."
    },

    "Neem": {
        "uses": "Traditionally used in skin care and oral hygiene practices.",
        "benefits": "Known for compounds studied for antimicrobial properties.",
        "precautions": "Avoid consuming neem preparations without professional medical guidance."
    },

    "Tulsi": {
        "uses": "Traditionally used in herbal preparations and Ayurveda.",
        "benefits": "Commonly used as a traditional herbal plant.",
        "precautions": "Consult a healthcare professional before medicinal use, especially with regular medicines."
    },

    "Moringa": {
        "uses": "Leaves are commonly used as a nutritious food and in traditional practices.",
        "benefits": "Moringa leaves contain various vitamins and minerals.",
        "precautions": "Use medicinal preparations with appropriate professional guidance."
    }
}


# ---------------- MAIN PAGE ----------------

st.title("🌿 Medicinal Plant Classification")

st.write(
    "Upload a plant image to identify the medicinal plant using Machine Learning."
)

st.info(
    "Model: HOG + SVM | Classes: 5 | Image Size: 128 × 128"
)


# ---------------- ABOUT PROJECT ----------------

st.subheader("ℹ️ About Project")

st.write(
    "This project uses Artificial Intelligence and "
    "Machine Learning to identify medicinal plants "
    "from uploaded images."
)

st.write(
    "The system uses HOG feature extraction and "
    "an SVM classifier to classify five medicinal plants."
)

st.caption(
    "Model accuracy on the test split: 75.86%. "
    "The confidence shown is a model score converted "
    "to a percentage-like value and is not a calibrated probability."
)


# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Choose a plant image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Plant Image",
            width=300
        )

        image = image.resize(IMG_SIZE)

        image_array = np.array(image)

        gray_image = np.mean(
            image_array,
            axis=2
        )

        features = hog(
            gray_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )


        # Prediction
        prediction_number = model.predict(
            [features]
        )[0]


        # Decision score
        decision_scores = model.decision_function(
            [features]
        )

        if len(decision_scores.shape) > 1:
            score = np.max(decision_scores[0])
        else:
            score = np.max(decision_scores)


        # Confidence
        confidence = min(
            max((score + 1) * 50, 0),
            100
        )


        # Convert prediction number to plant name
        prediction = encoder.inverse_transform(
            [prediction_number]
        )[0]

        plant_name = prediction.strip()

        plant_name = plant_name.replace(" bg aug", "")
        plant_name = plant_name.replace("bg aug", "")
        plant_name = plant_name.strip()

        # Fix plant name matching
        if plant_name.lower() == "aloe vera":
            plant_name = "Aloe Vera"

        elif plant_name.lower() == "neem":
            plant_name = "Neem"

        elif plant_name.lower() == "tulsi":
            plant_name = "Tulsi"

        elif plant_name.lower() == "hibiscus":
            plant_name = "Hibiscus"

        elif plant_name.lower() == "moringa":
            plant_name = "Moringa"

       

        # Prediction History
        st.session_state.history.append({
            "Plant": plant_name,
            "Confidence": f"{confidence:.2f}%",
            "Time": datetime.now().strftime(
                "%d-%m-%Y %I:%M:%S %p"
            )
        })


        # Prediction Result
        st.success(
            f"🌿 Predicted Plant: {plant_name}"
        )


        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # Plant Information
        info = plant_info.get(
            plant_name,
            {}
        )


        if info:

            st.subheader("💊 Uses")
            st.write(info["uses"])

            st.subheader("🌱 Benefits")
            st.write(info["benefits"])

            st.subheader("⚠️ Precautions")
            st.write(info["precautions"])

        else:

            st.info(
                "Plant information is not available."
            )


        st.warning(
            "This information is for educational purposes only "
            "and is not a substitute for professional medical advice."
        )


    except Exception:

        st.error(
            "Unable to process this image. "
            "Please upload a valid JPG, JPEG or PNG image."
        )


# ---------------- HISTORY ----------------

st.subheader("📋 Prediction History")


if len(st.session_state.history) > 0:

    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )


    if st.button("🗑️ Clear History"):

        st.session_state.history = []

        st.success(
            "Prediction history cleared."
        )

        st.rerun()

else:

    st.info(
        "No prediction history yet."
    )