import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Load trained model, scaler, and label encoder

rfc = pickle.load(open('rfc_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Feature extraction function

def extract_features(image):
    img_array = np.array(image)

    # If image still has an alpha channel, drop it
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    red_array = img_array[:, :, 0]
    green_array = img_array[:, :, 1]
    blue_array = img_array[:, :, 2]

    red_hist, _ = np.histogram(red_array.ravel(), bins=16, range=(0, 256))
    green_hist, _ = np.histogram(green_array.ravel(), bins=16, range=(0, 256))
    blue_hist, _ = np.histogram(blue_array.ravel(), bins=16, range=(0, 256))

    features = np.concatenate((red_hist, green_hist, blue_hist))
    return features

# Streamlit app UI

st.title("Pokemon Type Predictor")
st.write("Upload a Pokemon image (clean sprite/artwork, ideally with plain or transparent background) and the model will predict its Type 1 based on color features.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features
    features = extract_features(image)

    # Reshape to (1, 48) since scaler/model expect 2D input
    features = features.reshape(1, -1)

    # Scale using the already-fitted scaler (transform only, no fitting)
    scaled_features = scaler.transform(features)

    # Predict
    prediction = rfc.predict(scaled_features)
    predicted_type = le.inverse_transform(prediction)[0]

    st.subheader(f"Predicted Type: {predicted_type}")

    # Show top probabilities for a bit more insight
    if hasattr(rfc, "predict_proba"):
        probs = rfc.predict_proba(scaled_features)[0]
        top_indices = np.argsort(probs)[::-1][:5]
        st.write("Top 5 predictions:")
        for idx in top_indices:
            type_name = le.inverse_transform([idx])[0]
            st.write(f"- {type_name}: {probs[idx]*100:.1f}%")