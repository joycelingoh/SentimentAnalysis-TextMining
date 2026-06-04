import streamlit as st
import joblib
import pandas as pd

# ======================
# CONFIG
# ======================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# ======================
# LOAD FILES
# ======================

vectorizer = joblib.load("vectorizer.pkl")

models = {
    "Logistic Regression": joblib.load("logreg_model.pkl"),
    "SVM": joblib.load("svm_model.pkl")
}

# ======================
# SIDEBAR
# ======================

st.sidebar.title("⚙️ Pengaturan")

selected_model = st.sidebar.selectbox(
    "Pilih Model",
    list(models.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Sentiment Analysis Indonesia**

    Dataset:
    Review teks berlabel positif dan negatif

    Model:
    - Logistic Regression
    - Support Vector Machine

    Dibuat oleh:
    Joycelin
    """
)

# ======================
# MAIN PAGE
# ======================

st.title("📊 Sentiment Analysis Dashboard")

st.markdown(
    """
    Masukkan sebuah kalimat atau review untuk diprediksi sentimennya.
    """
)

text = st.text_area(
    "Masukkan Review",
    height=150,
    placeholder="Contoh: produk ini sangat bagus dan pengirimannya cepat"
)

# ======================
# SINGLE PREDICTION
# ======================

if st.button("🔍 Prediksi Sentimen"):

    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:

        model = models[selected_model]

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

# ======================
# BATCH PREDICTION
# ======================

st.markdown("---")

st.subheader("📂 Prediksi Banyak Data")

uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Preview Data:")
    st.dataframe(df.head())

    if "text" not in df.columns:

        st.error(
            "CSV harus memiliki kolom bernama 'text'"
        )

    else:

        model = models[selected_model]

        vectors = vectorizer.transform(
            df["text"]
        )

        preds = model.predict(vectors)

        df["prediction"] = preds

        df["prediction"] = df["prediction"].map({
            1: "Positif",
            0: "Negatif"
        })

        st.success("Prediksi berhasil")

        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Hasil",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

# ======================
# FOOTER
# ======================

st.markdown("---")

st.caption(
    "Sentiment Analysis menggunakan TF-IDF + Machine Learning"
)
