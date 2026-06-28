import streamlit as st
import pandas as pd
import plotly.express as px

from utils.clustering import clustering_data

st.set_page_config(
    page_title="MHA | Clustering",
    layout="wide"
)

df = clustering_data()

st.title("📍 Clustering")

# ==========================
# t-SNE Scatter
# ==========================

fig = px.scatter(
    df,
    x="x_tsne",
    y="y_tsne",
    color="cluster_name",
    hover_data=[
        "full_text"
    ],
    color_discrete_map={
        "Tekanan Akademik":"blue",
        "Masalah Keluarga":"green",
        "Masalah Finansial":"orange",
        "OUTLIER":"gray"
    }
)

fig.update_traces(
    marker=dict(size=8)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================
# Persentase Cluster
# ==========================

cluster_count = (
    df["cluster_name"]
    .value_counts()
    .reset_index()
)

cluster_count.columns = [
    "Kategori",
    "Jumlah"
]

col1, col2 = st.columns([2,1])

with col1:

    fig2 = px.pie(
        cluster_count,
        names="Kategori",
        values="Jumlah",
        hole=0.4
    )

    fig2.update_traces(
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

with col2:

    total = cluster_count["Jumlah"].sum()

    st.subheader("Persentase")

    for _, row in cluster_count.iterrows():

        persen = row["Jumlah"] / total * 100

        st.metric(
            row["Kategori"],
            f"{persen:.2f}%"
        )

st.divider()

# ==========================
# Full Dataset
# ==========================

st.dataframe(
    df[
        [
            "full_text",
            "clean_text",
            "cluster_name"
        ]
    ],
    use_container_width=True
)