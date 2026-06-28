import streamlit as st
import pandas as pd

from utils.load_data import load_data
from utils.preprocessing import preprocessing_data

st.set_page_config(
    page_title="MHA | Dataset",
    layout="wide"
)

st.title("📄 Dataset")

df = load_data()

st.subheader("Preview Dataset")

st.dataframe(
    df,
    use_container_width=True,
    height=250
)

st.divider()

df = preprocessing_data()

st.subheader("⚙️ Hasil Preprocessing")

st.dataframe(
    df[["full_text", "clean_text"]],
    use_container_width=True,
    height=250
)

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Jumlah Baris",
    df.shape[0]
)

col2.metric(
    "Jumlah Kolom",
    df.shape[1]
)

col3.metric(
    "Missing Value",
    df.isnull().sum().sum()
)