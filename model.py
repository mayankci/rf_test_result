import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    st.title("Store Prediction vs Actual Distribution")

    # Your Google Drive file ID
    file_id = "16mT7GtKKfNBxrSPp8A-CFJHtEdy4P102"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Load data
    df = pd.read_csv(url)

    store_codes = df['facility_code'].unique()
    store_code = st.selectbox("Select Store Code", sorted(store_codes))

    if store_code:
        count_matrix = analyze_prediction_distribution(df, store_code)

        st.subheader(f"Distribution Table for Store {store_code}")
        st.dataframe(count_matrix)

        st.subheader("Heatmap of Counts")
        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(count_matrix.iloc[:, :-1], annot=True, fmt='d', cmap='Blues')
        plt.yticks(rotation=0)  # Optional: rotate ticks for readability
        plt.ylabel('Ratio (Prediction / Actual, rounded to 0.1)')
        plt.xlabel('Predicted ROS (Rounded to 0.01)')
        plt.title(f'Prediction vs Actual Ratio Distribution — Store {store_code}')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()


if __name__ == "__main__":
    main()
