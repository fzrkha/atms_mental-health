import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import preprocessing_data


def clustering_data():

    df = preprocessing_data().copy()

    # ==========================
    # Seed Topics (Diperluas)
    # ==========================

    SEED_TOPICS = {

        "Tekanan Akademik": [
            "tugas", "skripsi", "tesis", "kampus",
            "kuliah", "dosen", "ujian", "nilai",
            "ipk", "kelas", "praktikum", "sidang",
            "semester", "belajar", "deadline",
            "presentasi", "bimbingan", "ospek",
            "mahasiswa", "akademik"
        ],

        "Masalah Keluarga": [
            "ibu", "ayah", "orang tua", "keluarga",
            "rumah", "adik", "kakak", "suami",
            "istri", "cerai", "nikah", "mama",
            "papa", "anak", "kelahi", "toxic",
            "orangtua", "keluarga", "saudara"
        ],

        "Finansial": [
            "uang", "duit", "bayar", "hutang",
            "utang", "biaya", "kerja", "gaji",
            "miskin", "ekonomi", "tagihan",
            "cicilan", "listrik", "kontrak",
            "kos", "sewa", "beasiswa",
            "penghasilan", "pengangguran"
        ]
    }

    # ==========================
    # TF-IDF
    # ==========================

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(df["clean_text"])

    feature_names = np.array(
        tfidf.get_feature_names_out()
    )

    # ==========================
    # Expand Seed
    # ==========================

    def expand_seed(seed_words, top_n=5):

        seed_idx = [
            np.where(feature_names == word)[0][0]
            for word in seed_words
            if word in feature_names
        ]

        if len(seed_idx) == 0:
            return seed_words

        sims = cosine_similarity(
            tfidf_matrix.T[seed_idx],
            tfidf_matrix.T
        ).mean(axis=0)

        idx = sims.argsort()[
            -(len(seed_idx) + top_n):
        ][::-1]

        expanded = list(seed_words)

        for i in idx:

            word = feature_names[i]

            if word not in expanded:
                expanded.append(word)

        return expanded

    EXPANDED = {
        k: expand_seed(v)
        for k, v in SEED_TOPICS.items()
    }

    # ==========================
    # Build Centroid
    # ==========================

    centroids = []

    cluster_names = list(EXPANDED.keys())

    for topic in cluster_names:

        pseudo_doc = " ".join(
            EXPANDED[topic]
        )

        vec = tfidf.transform(
            [pseudo_doc]
        )

        centroids.append(
            vec.toarray()[0]
        )

    centroids = np.array(centroids)

    # ==========================
    # Weighted Keyword Scoring
    # ==========================

    TOPIC_WEIGHT = {

        "Tekanan Akademik": {
            "skripsi":5, "tesis":5, "kuliah":4, "kampus":4,
            "dosen":4, "ujian":4, "deadline":5,
            "tugas":4, "nilai":3, "ipk":3,
            "semester":3, "belajar":2,
            "presentasi":2, "sidang":4,
            "bimbingan":3, "mahasiswa":2
        },

        "Masalah Keluarga": {
            "ibu":5, "ayah":5, "orang tua":5,
            "orangtua":5, "keluarga":4,
            "rumah":3, "adik":3, "kakak":3,
            "suami":4, "istri":4,
            "anak":3, "cerai":5,
            "toxic":3, "saudara":3,
            "mama":5, "papa":5
        },

        "Finansial": {
            "uang":5, "duit":5,
            "hutang":5, "utang":5,
            "bayar":4, "gaji":4,
            "kerja":3, "biaya":4,
            "ekonomi":5, "tagihan":4,
            "cicilan":4, "kontrak":3,
            "kos":3, "sewa":3,
            "beasiswa":3,
            "penghasilan":4,
            "pengangguran":5
        }

    }

    final_cluster = []
    confidence = []

    for i in range(len(df)):

        text = df.iloc[i]["clean_text"].lower()

        scores = {}

        for topic, keywords in TOPIC_WEIGHT.items():

            score = 0

            for word, weight in keywords.items():

                if word in text:
                    score += weight

            scores[topic] = score

        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]

        # fallback jika tidak ada keyword
        if best_score == 0:

            sims = cosine_similarity(
                tfidf.transform([text]),
                centroids
            )[0]

            best_topic = cluster_names[np.argmax(sims)]
            best_score = float(np.max(sims))

        final_cluster.append(best_topic)
        confidence.append(best_score)

    # ==========================
    # Output
    # ==========================

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    embedding = tsne.fit_transform(tfidf_matrix.toarray())

    df["x_tsne"] = embedding[:,0]
    df["y_tsne"] = embedding[:,1]

    df["cluster_name"] = final_cluster
    df["confidence_score"] = confidence

    return df