import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# Google Drive direct download links
MODEL_URL = "https://drive.google.com/uc?export=download&id=1G9Q04Nt2d__lm3Ow3QQ7IoU5Wd1tAziG"
ENCODER_URL = "https://drive.google.com/uc?export=download&id=1CiGHQzhZI_Iw80s7kG9NWUHLaJvHQqgF"

@st.cache_resource
def load_model_and_encoder():
    # Load model
    model_response = requests.get(MODEL_URL)
    model = joblib.load(io.BytesIO(model_response.content))

    # Load encoder
    encoder_response = requests.get(ENCODER_URL)
    encoder = joblib.load(io.BytesIO(encoder_response.content))

    return model, encoder

model, encoder = load_model_and_encoder()

# Required columns for prediction
valid_features = [
    'AG GROUP',
    'LO National ROS',
    'LO AG X product_id ROS',
    'LO Region X PID ROS',
    'LO AG X Region ROS',
    'LO State X PID ROS',
    'LO AG X State ROS',
    'LO City X PID ROS',
    'LO AG X City ROS',
]

categorical_cols = ['AG GROUP']

# Streamlit UI
st.title("📈 Predict Store X PID ROS")
uploaded_file = st.file_uploader("📁 Upload your input CSV", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Check for required columns
        missing = [col for col in valid_features if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            X = df[valid_features].copy()

            # Encode categorical columns (OrdinalEncoder handles unknowns with -1)
            X[categorical_cols] = encoder.transform(X[categorical_cols])

            # Predict
            df['Predicted Store X PID ROS'] = model.predict(X)

            st.success("✅ Prediction complete")
            st.dataframe(df.head(10))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv, "predicted_output.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Upload a CSV file to make predictions.")
