import re
import pandas as pd
import networkx as nx
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from community import community_louvain

from utils.load_data import load_data
from utils.preprocessing import preprocessing_data

list_stopwords = stopwords.words("indonesian")


def sna_data():

    # ==========================
    # LOAD DATA
    # ==========================

    raw_df = load_data().copy()
    prep_df = preprocessing_data().copy()

    # Tambahkan clean_text ke data asli
    raw_df["clean_text"] = prep_df["clean_text"]

    df = raw_df.copy()

    # ==========================
    # AMBIL RELASI MENTION
    # ==========================

    edges = []

    for _, row in df.iterrows():

        mentions = re.findall(
            r"@(\w+)",
            str(row["full_text"])
        )

        for mention in mentions:

            edges.append(
                (
                    str(row["username"]),
                    mention
                )
            )

    # ==========================
    # BUILD GRAPH
    # ==========================

    G = nx.DiGraph()

    G.add_edges_from(edges)

    G.remove_edges_from(
        nx.selfloop_edges(G)
    )

    G_undirected = G.to_undirected()

    # ==========================
    # NODE SIZE
    # ==========================

    node_sizes = [

        max(
            300,
            G.degree(node) * 200
        )

        for node in G.nodes()

    ]

    # ==========================
    # COMMUNITY DETECTION
    # ==========================

    partition = community_louvain.best_partition(
        G_undirected
    )

    com_dict = defaultdict(list)

    for node, com in partition.items():

        com_dict[com].append(node)

    # ==========================
    # DEGREE CENTRALITY
    # ==========================

    degree = nx.degree_centrality(
        G_undirected
    )

    tokoh_dominan = {}

    for com, nodes in com_dict.items():

        tokoh_dominan[com] = max(
            nodes,
            key=lambda x: degree.get(x, 0)
        )

    # ==========================
    # TOPIK UNIGRAM
    # ==========================

    text_col = "clean_text"

    topik_unigram = {}

    for com, nodes in com_dict.items():

        teks = df[
            df["username"].isin(nodes)
        ][text_col].dropna().tolist()

        if len(teks) == 0:

            topik_unigram[com] = []

            continue

        try:

            vectorizer = CountVectorizer(
                stop_words=list_stopwords,
                max_features=3
            )

            vectorizer.fit(teks)

            topik_unigram[com] = list(
                vectorizer.get_feature_names_out()
            )

        except:

            topik_unigram[com] = []

    # ==========================
    # TOPIK BIGRAM
    # ==========================

    topik_bigram = {}

    for com, nodes in com_dict.items():

        teks = df[
            df["username"].isin(nodes)
        ][text_col].dropna().tolist()

        if len(teks) == 0:

            topik_bigram[com] = []

            continue

        try:

            vectorizer = CountVectorizer(
                ngram_range=(2, 2),
                stop_words=list_stopwords
            )

            X = vectorizer.fit_transform(teks)

            total = X.sum(axis=0)

            freq = [

                (
                    word,
                    total[0, idx]
                )

                for word, idx

                in vectorizer.vocabulary_.items()

            ]

            freq = sorted(
                freq,
                key=lambda x: x[1],
                reverse=True
            )

            topik_bigram[com] = [

                x[0]

                for x in freq[:2]

            ]

        except:

            topik_bigram[com] = []

    # ==========================
    # COMMUNITY SUMMARY
    # ==========================

    summary = []

    for com in sorted(com_dict):

        summary.append({

            "Community": com,

            "Jumlah Anggota": len(
                com_dict[com]
            ),

            "Tokoh Dominan": tokoh_dominan[com],

            "Topik Unigram": ", ".join(
                topik_unigram[com]
            ),

            "Topik Bigram": ", ".join(
                topik_bigram[com]
            )

        })

    summary_df = pd.DataFrame(summary)

    # ==========================
    # RETURN
    # ==========================

    return {

        "graph": G,

        "edges": edges,

        "node_sizes": node_sizes,

        "community_summary": summary_df

    }