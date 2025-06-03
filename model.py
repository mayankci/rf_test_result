import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Core logic for processing
def analyze_prediction_distribution(df, store_code):
    store_df = df[df['facility_code'] == store_code].copy()

    # Compute prediction-to-actual ratio and round
    store_df['Ratio'] = store_df['RandomForest_Prediction'] / store_df['Actual'].replace(0, np.nan)
    store_df['Ratio'] = store_df['Ratio'].fillna(0)
    store_df['Ratio'] = round(store_df['Ratio'] / 0.1).astype(int) * 0.1

    # Round Actual and Prediction values
    store_df['Actual'] = round(store_df['Actual'] / 0.1).astype(int) * 0.1
    store_df['RandomForest_Prediction'] = round(store_df['RandomForest_Prediction'] / 0.01).astype(int) * 0.01

    # Pivot to get count matrix
    count_matrix = store_df.pivot_table(
        index='Ratio',
        columns='RandomForest_Prediction',
        aggfunc='size',
        fill_value=0
    )

    # Add % distribution column
    row_totals = count_matrix.sum(axis=1)
    grand_total = row_totals.sum()
    count_matrix['%'] = (row_totals / grand_total * 100).round(1)

    # Sort prediction columns, keep % at end
    count_matrix = count_matrix.reindex(sorted(count_matrix.columns[:-1]), axis=1).assign(**{'%': count_matrix['%']})

    return count_matrix

# Streamlit app UI
def main():
    st.title("📊 Store Prediction vs Actual Distribution")

    # File uploader
    uploaded_file = st.file_uploader("📁 Upload CSV file (with 'facility_code', 'City', 'RandomForest_Prediction', 'Actual')", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load dataset
            df = pd.read_csv(uploaded_file)

            # Check required columns
            required_columns = {'facility_code', 'City', 'RandomForest_Prediction', 'Actual'}
            if not required_columns.issubset(df.columns):
                st.error(f"The file must contain columns: {', '.join(required_columns)}")
                return

            # Filter out invalid store codes
            df = df[~df['facility_code'].astype(str).str.startswith("LKST01")]

            # Drop duplicates to get unique store-city pairs
            df_unique = df[['facility_code', 'City']].drop_duplicates()

            # First Dropdown: City
            cities = sorted(df_unique['City'].dropna().unique())
            selected_city = st.selectbox("🏙️ Select City", cities)

            # Second Dropdown: Stores in that City
            stores_in_city = df_unique[df_unique['City'] == selected_city]
            store_options = sorted(stores_in_city['facility_code'].unique())
            selected_store = st.selectbox("🏬 Select Store in " + selected_city, store_options)

            if selected_store:
                count_matrix = analyze_prediction_distribution(df, selected_store)

                # Format index
                count_matrix.index = [f"{val:.1f}" for val in count_matrix.index]

                # Display table
                st.subheader(f"🧾 Distribution Table — Store {selected_store}")
                st.dataframe(count_matrix)

                # Heatmap
                st.subheader("🔥 Heatmap of Prediction vs Ratio")
                plt.figure(figsize=(16, 10))
                sns.heatmap(count_matrix.iloc[:, :-1], annot=True, fmt='d', cmap='Blues')
                plt.yticks(rotation=0)
                plt.ylabel('Ratio (Prediction / Actual, rounded to 0.1)')
                plt.xlabel('Predicted ROS (Rounded to 0.01)')
                plt.title(f'Prediction vs Actual Ratio — Store {selected_store}')
                st.pyplot(plt.gcf())
                plt.clf()
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    else:
        st.info("⬆️ Please upload a valid CSV file to continue.")

# Run Streamlit app
if __name__ == "__main__":
    main()
