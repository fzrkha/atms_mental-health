import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.load_data import load_data
from utils.classification import classification_data
from utils.clustering import clustering_data
from utils.sna import sna_data
from utils.helper_network import helper_network_data

st.set_page_config(
    page_title="MHA | Dashboard",
    layout="wide"
)

st.title("Mental Health Analysis | MHA")

# =====================================================
# LOAD DATA
# =====================================================

raw_df = load_data()

class_df = classification_data()

cluster_df = clustering_data()

sna = sna_data()

network = helper_network_data()

# =====================================================
# METRIC
# =====================================================

total_curhatan = len(class_df)

curhat_darurat = len(

    class_df[
        class_df["auto_label"]
        ==
        "Butuh Pertolongan Segera"
    ]

)

total_akun = raw_df["username"].nunique()

if len(sna["community_summary"]) > 0:

    top_supporter = (
        "@"+network["helper_df"].iloc[0]["Username"]
        if not network["helper_df"].empty
        else "-"
    )

else:

    top_supporter = "-"

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Total Curhatan",
    total_curhatan
)

c2.metric(
    "Curhat Darurat",
    curhat_darurat
)

c3.metric(
    "Total Akun",
    total_akun
)

c4.metric(
    "Top Supporter",
    top_supporter
)

# =====================================================
# STATUS
# =====================================================

persen = curhat_darurat / total_curhatan

if persen >= 0.30:

    st.error(
        "🔴 Status Saat Ini : WASPADA"
    )

elif persen >= 0.15:

    st.warning(
        "🟠 Status Saat Ini : SIAGA"
    )

else:

    st.success(
        "🟢 Status Saat Ini : AMAN"
    )

st.divider()

# =====================================================
# DISTRIBUSI CLUSTER
# =====================================================

cluster_count = (

    cluster_df["cluster_name"]

    .value_counts()

    .reset_index()

)

cluster_count.columns = [

    "Kategori",

    "Jumlah"

]

fig = px.pie(

    cluster_count,

    names="Kategori",

    values="Jumlah",

    hole=0.4

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()

# =====================================================
# EKOSISTEM AKUN PENOLONG
# =====================================================

st.subheader("🌐 Ekosistem Akun Penolong")

st.caption(
    "Visualisasi hubungan akun yang memberikan dukungan kepada akun lain berdasarkan rule-based."
)

fig = go.Figure()

# =====================================================
# EDGE
# =====================================================

fig.add_trace(

    go.Scatter(

        x=network["edge_x"],

        y=network["edge_y"],

        mode="lines",

        line=dict(

            color="#BDBDBD",

            width=1

        ),

        hoverinfo="none",

        showlegend=False

    )

)

# =====================================================
# NODE
# =====================================================

labels = []

for text in network["node_text"]:

    label = (

        text

        .split("<br>")[0]

        .replace("<b>", "")

        .replace("</b>", "")

    )

    labels.append(label)

fig.add_trace(

    go.Scatter(

        x=network["node_x"],

        y=network["node_y"],

        mode="markers+text",

        text=labels,

        textposition="top center",

        hovertext=network["node_text"],

        hoverinfo="text",

        marker=dict(

            size=network["node_sizes"],

            color=network["node_colors"],

            line=dict(

                width=1,

                color="black"

            )

        ),

        showlegend=False

    )

)

fig.update_layout(

    template="plotly_white",

    height=750,

    hovermode="closest",

    margin=dict(

        l=20,

        r=20,

        t=20,

        b=20

    ),

    xaxis=dict(

        visible=False

    ),

    yaxis=dict(

        visible=False

    )

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()

# =====================================================
# RINGKASAN EKOSISTEM
# =====================================================

c1, c2 = st.columns(2)

# ==========================
# TOP PENOLONG
# ==========================

with c1:

    st.subheader("🤝 Top Akun Penolong")

    if network["helper_df"].empty:

        st.info(
            "Belum ditemukan akun penolong."
        )

    else:

        st.dataframe(

            network["helper_df"],

            use_container_width=True,

            hide_index=True,

            height=350

        )

# ==========================
# TOP PENERIMA DUKUNGAN
# ==========================

with c2:

    st.subheader("💙 Top Penerima Dukungan")

    if network["receiver_df"].empty:

        st.info(
            "Belum ditemukan akun penerima dukungan."
        )

    else:

        st.dataframe(

            network["receiver_df"],

            use_container_width=True,

            hide_index=True,

            height=350

        )

st.divider()

# =====================================================
# DETAIL RELASI PENOLONG
# =====================================================

st.subheader("🔗 Detail Relasi Akun Penolong")

if network["edge_df"].empty:

    st.warning(
        "Belum ada relasi akun penolong yang ditemukan."
    )

else:

    st.dataframe(

        network["edge_df"],

        use_container_width=True,

        height=400,

        hide_index=True

    )

    st.download_button(

        "📥 Download Relasi Akun Penolong",

        network["edge_df"].to_csv(
            index=False
        ),

        file_name="helper_network.csv",

        mime="text/csv"

    )