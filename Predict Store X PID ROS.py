import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import gdown
import os

# Google Drive file IDs
MODEL_FILE_ID = "1G9Q04Nt2d__lm3Ow3QQ7IoU5Wd1tAziG"
ENCODER_FILE_ID = "1CiGHQzhZI_Iw80s7kG9NWUHLaJvHQqgF"

@st.cache_resource
def load_model_and_encoder():
    model_path = "model.pkl"
    encoder_path = "encoder.pkl"

    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", model_path, quiet=False)
    if not os.path.exists(encoder_path):
        gdown.download(f"https://drive.google.com/uc?id={ENCODER_FILE_ID}", encoder_path, quiet=False)

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

model, encoder = load_model_and_encoder()

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

def main():
    st.title("📈 Predict & Analyze Store X PID ROS")

    uploaded_file = st.file_uploader("📁 Upload your input CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Check required feature columns
            missing = [col for col in valid_features if col not in df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                return

            if 'Store X PID ROS' not in df.columns or 'facility_code' not in df.columns or 'City' not in df.columns:
                st.error("Your file must include 'Actual', 'facility_code', and 'City' columns for analysis.")
                return

            # Prepare input and predict
            X = df[valid_features].copy()
            X[categorical_cols] = encoder.transform(X[categorical_cols])
            df['Predicted Store X PID ROS'] = model.predict(X)

            # Show predictions
            st.success("✅ Predictions complete.")
            st.dataframe(df.head(10))

            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predicted Results", csv, "predicted_output.csv", "text/csv")


        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
    else:
        st.info("⬆️ Upload your input file to get started.")

if __name__ == "__main__":
    main()
