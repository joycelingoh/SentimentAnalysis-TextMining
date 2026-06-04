import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("📊 Sentiment Analysis")
st.write("Masukkan kalimat untuk diprediksi sentimennya")

# Input
text = st.text_area(
    "Review",
    placeholder="Contoh: produknya bagus dan pengirimannya cepat"
)

if st.button("Prediksi Sentimen"):

    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu")
    else:

        vector = vectorizer.transform([text])

        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        confidence = max(prob) * 100

        if pred == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😠 Sentimen Negatif")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

# Sidebar
st.sidebar.header("Tentang")
st.sidebar.info(
    """
    Aplikasi Sentiment Analysis
    menggunakan TF-IDF dan Machine Learning.
    Oleh: Joycelin
    """
)
