import streamlit as st
import pandas as pd
import plotly.express as px

from utils.model import model_data

st.set_page_config(
    page_title="MHA | Model",
    layout="wide"
)

st.title("🧠 Mental Health Classification Model")

# ==========================
# Train Model
# ==========================

with st.spinner("Melatih model..."):

    result = model_data()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{result['accuracy']:.2%}"
)

col2.metric(
    "Precision",
    f"{result['precision']:.2%}"
)

col3.metric(
    "Recall",
    f"{result['recall']:.2%}"
)

col4.metric(
    "F1-Score",
    f"{result['f1']:.2%}"
)

# ==========================
# Classification Report
# ==========================

st.subheader("Classification Report")

report_df = (
    pd.DataFrame(result["classification_report"])
    .transpose()
)

report_df = report_df.round(3)

st.dataframe(
    report_df,
    use_container_width=True
)

st.divider()

# ==========================
# Confusion Matrix
# ==========================

st.subheader("Confusion Matrix")

cm_df = pd.DataFrame(
    result["confusion_matrix"],
    index=[
        "Actual Curhat Ringan",
        "Actual Butuh Pertolongan Segera"
    ],
    columns=[
        "Pred Curhat Ringan",
        "Pred Butuh Pertolongan Segera"
    ]
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="auto"
)

fig_cm.update_layout(
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label"
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)

# ==========================
# Upload Dataset Uji
# ==========================

uploaded_file = st.file_uploader(
    "Upload Dataset Uji",
    type=["csv"]
)

st.info(
    """
Upload dataset uji (.csv) yang memiliki kolom:

• full_text
• true_label
• true_cluster
"""
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.subheader("Preview Dataset Uji")

    st.dataframe(
        test_df,
        use_container_width=True,
        height=250
    )

    st.divider()

    # ==========================
    # Prediction
    # ==========================

    with st.spinner("Melakukan prediksi..."):

        hasil = model_data(test_df)

    st.success("Prediksi selesai!")

    # ==========================
    # Metric
    # ==========================

    if "test_accuracy" in hasil:

        st.subheader("Evaluasi Klasifikasi")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{hasil['test_accuracy']:.2%}"
        )

        c2.metric(
            "Precision",
            f"{hasil['test_precision']:.2%}"
        )

        c3.metric(
            "Recall",
            f"{hasil['test_recall']:.2%}"
        )

        c4.metric(
            "F1-Score",
            f"{hasil['test_f1']:.2%}"
        )

        st.subheader("Classification Report")

        test_report_df = (
            pd.DataFrame(
                hasil["test_report"]
            ).transpose()
        )

        test_report_df = test_report_df.round(3)

        st.dataframe(
            test_report_df,
            use_container_width=True
        )

    if "cluster_accuracy" in hasil:

        st.divider()

        st.subheader("Evaluasi Clustering")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{hasil['cluster_accuracy']:.2%}"
        )

        c2.metric(
            "Precision",
            f"{hasil['cluster_precision']:.2%}"
        )

        c3.metric(
            "Recall",
            f"{hasil['cluster_recall']:.2%}"
        )

        c4.metric(
            "F1-Score",
            f"{hasil['cluster_f1']:.2%}"
        )

        st.subheader("Clustering Report")

        cluster_report_df = (
            pd.DataFrame(
                hasil["cluster_report"]
            ).transpose()
        )

        cluster_report_df = cluster_report_df.round(3)

        st.dataframe(
            cluster_report_df,
            use_container_width=True
        )

    st.divider()

    # ==========================
    # Hasil Prediksi
    # ==========================

    st.subheader("Hasil Prediksi")

    st.dataframe(
        hasil["prediction_data"],
        use_container_width=True,
        height=500
    )

    st.download_button(
        "📥 Download Hasil",
        hasil["prediction_data"].to_csv(index=False),
        "hasil_prediksi.csv",
        "text/csv"
    )