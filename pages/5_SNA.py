import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from utils.sna import sna_data

st.set_page_config(
    page_title="MHA | Social Network Analysis",
    layout="wide"
)

st.title("🕸 Social Network Analysis")

# ====================================
# LOAD
# ====================================

with st.spinner("Membangun jaringan sosial..."):

    result = sna_data()

G = result["graph"]

node_sizes = result["node_sizes"]

# ====================================
# INFO
# ====================================

c1, c2 = st.columns(2)

c1.metric(
    "Jumlah Node",
    G.number_of_nodes()
)

c2.metric(
    "Jumlah Edge",
    G.number_of_edges()
)

st.divider()

# ====================================
# NETWORK GRAPH
# ====================================

st.subheader("Network Graph")

fig, ax = plt.subplots(
    figsize=(15,10)
)

pos = nx.spring_layout(
    G,
    seed=42,
    k=0.4
)

nx.draw_networkx_nodes(

    G,

    pos,

    node_size=node_sizes,

    node_color="skyblue",

    edgecolors="black",

    linewidths=0.5

)

nx.draw_networkx_edges(

    G,

    pos,

    arrows=True,

    arrowstyle="-|>",

    arrowsize=10,

    edge_color="gray",

    alpha=0.6

)

nx.draw_networkx_labels(

    G,

    pos,

    font_size=7

)

plt.axis("off")

st.pyplot(fig)

st.divider()

# ====================================
# COMMUNITY SUMMARY
# ====================================

st.subheader("Deteksi Komunitas")

st.dataframe(

    result["community_summary"],

    use_container_width=True,

    height=400

)

st.divider()

# ====================================
# EDGE LIST
# ====================================

st.subheader("Edge List")

edge_df = pd.DataFrame(

    result["edges"],

    columns=[

        "Source",

        "Target"

    ]

)

st.dataframe(

    edge_df,

    use_container_width=True,

    height=400

)

st.download_button(

    "📥 Download Edge List",

    edge_df.to_csv(index=False),

    "edge_list.csv",

    "text/csv"

)