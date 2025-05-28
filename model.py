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

    # Google Drive CSV download link
    file_id = "16mT7GtKKfNBxrSPp8A-CFJHtEdy4P102"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Load dataset
    df = pd.read_csv(url)

    # Store selection
    # Create a list of unique (store_code, city) pairs, excluding codes starting with "LKST01"
    store_city_pairs = df[['facility_code', 'City']].drop_duplicates()
    store_city_pairs['label'] = store_city_pairs['facility_code'] + " - " + store_city_pairs['City']
    store_city_pairs = store_city_pairs.sort_values('label', ascending=False)
    options = store_city_pairs['label'].tolist()
    selected_label = st.selectbox("Select Store Code and City", options)
    selected_store_code = selected_label.split(" - ")[0]


    if store_code:
        count_matrix = analyze_prediction_distribution(df, store_code)

        # Clean index to show only 1 decimal place for Ratio
        count_matrix.index = [f"{val:.1f}" for val in count_matrix.index]

        # Show table
        st.subheader(f"🧾 Distribution Table — Store {store_code}")
        st.dataframe(count_matrix)

        # Show heatmap
        st.subheader("🔥 Heatmap of Prediction vs Ratio")
        plt.figure(figsize=(16, 10))  # Bigger plot
        ax = sns.heatmap(count_matrix.iloc[:, :-1], annot=True, fmt='d', cmap='Blues')
        plt.yticks(rotation=0)
        plt.ylabel('Ratio (Prediction / Actual, rounded to 0.1)')
        plt.xlabel('Predicted ROS (Rounded to 0.01)')
        plt.title(f'Prediction vs Actual Ratio — Store {store_code}')
        plt.tight_layout()


        st.pyplot(plt.gcf())
        plt.clf()

# Run Streamlit app
if __name__ == "__main__":
    main()
