import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from utils.classification import classification_data
from utils.clustering import clustering_data
from utils.preprocessing import clean_text


def model_data(test_df=None):

    # ==========================
    # DATA LATIH
    # ==========================

    train_class = classification_data().copy()
    train_cluster = clustering_data().copy()

    train_df = train_class.copy()

    train_df["cluster_name"] = train_cluster["cluster_name"]

    # ==========================
    # TRAIN MODEL
    # ==========================

    X = train_df["clean_text"]
    y = train_df["auto_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2)
    )

    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    svm = SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=42
    )

    svm.fit(
        X_train,
        y_train
    )

    y_pred = svm.predict(X_test)

    result = {

        "model":svm,
        "tfidf":tfidf,

        "accuracy":accuracy_score(
            y_test,
            y_pred
        ),

        "precision":precision_score(
            y_test,
            y_pred,
            average="weighted"
        ),

        "recall":recall_score(
            y_test,
            y_pred,
            average="weighted"
        ),

        "f1":f1_score(
            y_test,
            y_pred,
            average="weighted"
        ),

        "confusion_matrix":confusion_matrix(
            y_test,
            y_pred
        ),

        "classification_report":classification_report(
            y_test,
            y_pred,
            output_dict=True
        )
    }

    # ==========================
    # TIDAK ADA DATA UJI
    # ==========================

    if test_df is None:
        return result

    # ==========================
    # PREPROCESS DATA UJI
    # ==========================

    test_df = test_df.copy()

    test_df["clean_text"] = (
        test_df["clean_text"]
        .astype(str)
        .apply(clean_text)
    )

    # ==========================
    # PREDIKSI KLASIFIKASI
    # ==========================

    X_uji = tfidf.transform(
        test_df["clean_text"]
    )

    test_df["prediction"] = svm.predict(
        X_uji
    )

    test_df["confidence"] = (
        svm.predict_proba(X_uji)
        .max(axis=1)
    )

    # ==========================
    # PREDIKSI CLUSTER
    # ==========================

    topic_keywords = {

        "Tekanan Akademik":[
            "tugas","skripsi","tesis","kampus","kuliah",
            "dosen","ujian","nilai","ipk","deadline",
            "semester","belajar","sidang"
        ],

        "Masalah Keluarga":[
            "ibu","ayah","orang tua","keluarga",
            "adik","kakak","rumah","cerai",
            "anak","mama","papa"
        ],

        "Masalah Finansial":[
            "uang","duit","hutang","utang",
            "gaji","kerja","biaya",
            "tagihan","cicilan",
            "ekonomi","kos","sewa"
        ]

    }

    cluster_result = []

    for text in test_df["clean_text"]:

        score = {}

        for topic, words in topic_keywords.items():

            score[topic] = sum(
                1
                for w in words
                if w in text
            )

        cluster_result.append(
            max(score,key=score.get)
        )

    test_df["cluster_name"] = cluster_result

    # ==========================
    # EVALUASI DATA UJI
    # ==========================

    if "true_label" in test_df.columns:

        result["test_accuracy"] = accuracy_score(
            test_df["true_label"],
            test_df["prediction"]
        )

        result["test_precision"] = precision_score(
            test_df["true_label"],
            test_df["prediction"],
            average="weighted"
        )

        result["test_recall"] = recall_score(
            test_df["true_label"],
            test_df["prediction"],
            average="weighted"
        )

        result["test_f1"] = f1_score(
            test_df["true_label"],
            test_df["prediction"],
            average="weighted"
        )

        result["test_cm"] = confusion_matrix(
            test_df["true_label"],
            test_df["prediction"]
        )

        result["test_report"] = classification_report(
            test_df["true_label"],
            test_df["prediction"],
            output_dict=True
        )

    # ==========================
    # EVALUASI CLUSTER
    # ==========================

    if "true_cluster" in test_df.columns:

        result["cluster_accuracy"] = accuracy_score(
            test_df["true_cluster"],
            test_df["cluster_name"]
        )

        result["cluster_precision"] = precision_score(
            test_df["true_cluster"],
            test_df["cluster_name"],
            average="weighted"
        )

        result["cluster_recall"] = recall_score(
            test_df["true_cluster"],
            test_df["cluster_name"],
            average="weighted"
        )

        result["cluster_f1"] = f1_score(
            test_df["true_cluster"],
            test_df["cluster_name"],
            average="weighted"
        )

        result["cluster_cm"] = confusion_matrix(
            test_df["true_cluster"],
            test_df["cluster_name"]
        )

        result["cluster_report"] = classification_report(
            test_df["true_cluster"],
            test_df["cluster_name"],
            output_dict=True
        )

    result["prediction_data"] = test_df

    return result