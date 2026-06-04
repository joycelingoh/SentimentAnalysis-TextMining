import streamlit as st
import pandas as pd
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    model = joblib.load("logreg_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📊 Sentiment Analysis")

st.sidebar.info(
    """
    Aplikasi Sentiment Analysis
    menggunakan:

    • TF-IDF Vectorizer
    • Logistic Regression

    Dataset:
    Review Positif & Negatif

    Oleh: Joycelin
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("📈 Model Performance")

performance_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    "Value": [
        "96.8%",
        "96.5%",
        "97.1%",
        "96.8%"
    ]
})

st.sidebar.table(performance_df)

# =====================================
# TITLE
# =====================================

st.title("📊 Sentiment Analysis Dashboard")

st.write(
    "Masukkan sebuah review untuk memprediksi sentimen positif atau negatif."
)

# =====================================
# SINGLE PREDICTION
# =====================================

st.subheader("✍️ Prediksi Satu Review")

text = st.text_area(
    "Masukkan Review",
    placeholder="Contoh: produk ini sangat bagus dan pengirimannya cepat",
    height=150
)

if st.button("🔍 Prediksi Sentimen"):

    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        confidence = max(probability) * 100

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.success("😊 Sentimen Positif")
            else:
                st.error("😠 Sentimen Negatif")

        with col2:

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

# =====================================
# BATCH PREDICTION
# =====================================

st.markdown("---")

st.subheader("📂 Prediksi Banyak Data")

uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.write("### Preview Data")
        st.dataframe(df.head())

        if "text" not in df.columns:

            st.error(
                "File CSV harus memiliki kolom bernama 'text'"
            )

        else:

            vectors = vectorizer.transform(
                df["text"].astype(str)
            )

            predictions = model.predict(vectors)

            df["prediction"] = predictions

            df["prediction"] = df["prediction"].map({
                1: "Positif",
                0: "Negatif"
            })

            st.success("Prediksi berhasil!")

            st.write("### Hasil Prediksi")
            st.dataframe(df)

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Hasil",
                data=csv,
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"Terjadi error: {e}"
        )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "Sentiment Analysis menggunakan TF-IDF dan Logistic Regression - Joycelin (2702213713)"
)
