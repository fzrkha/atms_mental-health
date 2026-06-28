import streamlit as st
import pandas as pd
import plotly.express as px

from utils.classification import classification_data

st.set_page_config(
    page_title="MHA | Classification",
    layout="wide"
)

df = classification_data()

st.title("🏷️ Classification")

# =====================
# Pie Chart
# =====================

label_count = (
    df["auto_label"]
    .value_counts()
    .reset_index()
)

label_count.columns = [
    "Kategori",
    "Jumlah"
]

fig = px.pie(
    label_count,
    names="Kategori",
    values="Jumlah",
    hole=0.4
)

fig.update_traces(
    textinfo="percent+label"
)

col1, col2 = st.columns([2,1])

with col1:
    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    total = label_count["Jumlah"].sum()

    st.subheader("Persentase")

    for _, row in label_count.iterrows():

        persen = row["Jumlah"] / total * 100

        st.metric(
            row["Kategori"],
            f"{persen:.2f}%"
        )

st.divider()

# =====================
# Table
# =====================

st.dataframe(
    df[
        [
            "full_text",
            "clean_text",
            "distress_score",
            "auto_label"
        ]
    ],
    use_container_width=True
)