import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Analisis Sentimen Bahasa Indonesia")

text = st.text_area(
    "Masukkan kalimat review"
)

if st.button("Prediksi"):

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)

    if prediction == 1:
        st.success("Sentimen Positif")
    else:
        st.error("Sentimen Negatif")

    st.write(
        f"Confidence: {max(probability[0])*100:.2f}%"
    )