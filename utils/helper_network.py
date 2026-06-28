import re
import pandas as pd
import networkx as nx

from utils.load_data import load_data


def helper_network_data():

    # ==========================
    # LOAD DATA
    # ==========================

    df = load_data().copy()

    # ==========================
    # RULE BASED KATA SUPPORT
    # ==========================

    support_words = [

        "semangat",
        "kuat",
        "sabar",
        "gapapa",
        "ga apa",
        "jangan menyerah",
        "ayo",
        "peluk",
        "doa",
        "doakan",
        "aku ada",
        "tenang",
        "istirahat",
        "support",
        "tetap semangat",
        "kamu hebat",
        "stay strong",
        "bisa kok",
        "you're not alone"

    ]

    # ==========================
    # RELASI PENOLONG
    # ==========================

    edges = []

    helper_count = {}

    receiver_count = {}

    # ==========================
    # CARI AKUN PENOLONG
    # ==========================

    for _, row in df.iterrows():

        username = str(row["username"])

        text = str(row["full_text"]).lower()

        mentions = re.findall(
            r"@(\w+)",
            text
        )

        if len(mentions) == 0:
            continue

        # Cek apakah tweet mengandung kata support

        is_support = any(

            word in text

            for word in support_words

        )

        if not is_support:
            continue

        # Simpan relasi penolong -> penerima

        for target in mentions:

            edges.append(
                (
                    username,
                    target
                )
            )

            helper_count[username] = (

                helper_count.get(
                    username,
                    0
                ) + 1

            )

            receiver_count[target] = (

                receiver_count.get(
                    target,
                    0
                ) + 1

            )
    
    # ==========================
    # BUILD GRAPH
    # ==========================

    G = nx.DiGraph()

    G.add_edges_from(edges)

    G.remove_edges_from(
        nx.selfloop_edges(G)
    )

    # ==========================
    # POSISI NODE
    # ==========================

    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.8
    )

    # ==========================
    # UKURAN NODE
    # ==========================

    node_sizes = []

    node_colors = []

    for node in G.nodes():

        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)

        # ukuran node = banyak interaksi

        node_sizes.append(
            (out_degree + in_degree) * 250 + 300
        )

        # hijau = penolong
        # merah = penerima

        if out_degree > in_degree:

            node_colors.append("green")

        elif in_degree > out_degree:

            node_colors.append("red")

        else:

            node_colors.append("skyblue")

    # ==========================
    # EDGE COORDINATE
    # ==========================

    edge_x = []

    edge_y = []

    for source, target in G.edges():

        x0, y0 = pos[source]

        x1, y1 = pos[target]

        edge_x.extend(
            [
                x0,
                x1,
                None
            ]
        )

        edge_y.extend(
            [
                y0,
                y1,
                None
            ]
        )

    # ==========================
    # NODE COORDINATE
    # ==========================

    node_x = []

    node_y = []

    node_text = []

    for node in G.nodes():

        x, y = pos[node]

        node_x.append(x)

        node_y.append(y)

        node_text.append(
            f"""
            <b>@{node}</b><br>
            Memberi Dukungan : {G.out_degree(node)}<br>
            Menerima Dukungan : {G.in_degree(node)}
            """
        )

    # ==========================
    # TOP HELPER
    # ==========================

    helper_df = pd.DataFrame(

        {

            "Username": list(
                helper_count.keys()
            ),

            "Jumlah Bantuan": list(
                helper_count.values()
            )

        }

    )

    if not helper_df.empty:

        helper_df = (

            helper_df

            .sort_values(

                by="Jumlah Bantuan",

                ascending=False

            )

            .reset_index(drop=True)

        )

    # ==========================
    # TOP RECEIVER
    # ==========================

    receiver_df = pd.DataFrame(

        {

            "Username": list(
                receiver_count.keys()
            ),

            "Jumlah Dukungan": list(
                receiver_count.values()
            )

        }

    )

    if not receiver_df.empty:

        receiver_df = (

            receiver_df

            .sort_values(

                by="Jumlah Dukungan",

                ascending=False

            )

            .reset_index(drop=True)

        )

    # ==========================
    # EDGE TABLE
    # ==========================

    edge_df = pd.DataFrame(

        edges,

        columns=[

            "Source",

            "Target"

        ]

    )

    # ==========================
    # RETURN
    # ==========================

    return {

        "graph": G,

        "pos": pos,

        "edge_x": edge_x,
        "edge_y": edge_y,

        "node_x": node_x,
        "node_y": node_y,

        "node_sizes": node_sizes,

        "node_colors": node_colors,

        "node_text": node_text,

        "helper_df": helper_df,

        "receiver_df": receiver_df,

        "edge_df": edge_df

    }