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

def analyze_prediction_distribution(df, store_code):
    store_df = df[df['facility_code'] == store_code].copy()
    store_df['Ratio'] = store_df['RandomForest_Prediction'] / store_df['Actual'].replace(0, np.nan)
    store_df['Ratio'] = store_df['Ratio'].fillna(0)
    store_df['Ratio'] = round(store_df['Ratio'] / 0.1).astype(int) * 0.1
    store_df['Actual'] = round(store_df['Actual'] / 0.1).astype(int) * 0.1
    store_df['RandomForest_Prediction'] = round(store_df['RandomForest_Prediction'] / 0.01).astype(int) * 0.01

    count_matrix = store_df.pivot_table(
        index='Ratio',
        columns='RandomForest_Prediction',
        aggfunc='size',
        fill_value=0
    )
    row_totals = count_matrix.sum(axis=1)
    grand_total = row_totals.sum()
    count_matrix['%'] = (row_totals / grand_total * 100).round(1)
    count_matrix = count_matrix.reindex(sorted(count_matrix.columns[:-1]), axis=1).assign(**{'%': count_matrix['%']})

    return count_matrix

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

            if 'Actual' not in df.columns or 'facility_code' not in df.columns or 'City' not in df.columns:
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

            # City → Store selection
            df_unique = df[['facility_code', 'City']].drop_duplicates()
            cities = sorted(df_unique['City'].dropna().unique())
            selected_city = st.selectbox("🏙️ Select City", cities)

            stores = df_unique[df_unique['City'] == selected_city]['facility_code'].unique()
            selected_store = st.selectbox(f"🏬 Select Store in {selected_city}", sorted(stores))

            if selected_store:
                count_matrix = analyze_prediction_distribution(df, selected_store)
                count_matrix.index = [f"{val:.1f}" for val in count_matrix.index]

                st.subheader(f"🧾 Distribution Table — Store {selected_store}")
                st.dataframe(count_matrix)

                st.subheader("🔥 Heatmap of Prediction vs Ratio")
                plt.figure(figsize=(16, 10))
                sns.heatmap(count_matrix.iloc[:, :-1], annot=True, fmt='d', cmap='Blues')
                plt.yticks(rotation=0)
                plt.ylabel('Ratio (Prediction / Actual, rounded to 0.1)')
                plt.xlabel('Predicted ROS (Rounded to 0.01)')
                plt.title(f'Prediction vs Actual Ratio — Store {selected_store}')
                st.pyplot(plt)
                plt.clf()

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
    else:
        st.info("⬆️ Upload your input file to get started.")

if __name__ == "__main__":
    main()
