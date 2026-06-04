import streamlit as st
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊"
)

vectorizer = joblib.load(
    "vectorizer.pkl"
)

tokenizer = joblib.load(
    "tokenizer.pkl"
)

models = {
    "Logistic Regression":
        joblib.load(
            "logreg_model.pkl"
        ),

    "SVM":
        joblib.load(
            "svm_model.pkl"
        ),

    "Random Forest":
        joblib.load(
            "random_forest.pkl"
        )
}

lstm_model = load_model(
    "lstm_model.keras"
)

st.title(
    "📊 Sentiment Analysis Indonesia"
)

selected_model = st.selectbox(
    "Pilih Model",
    [
        "Logistic Regression",
        "SVM",
        "Random Forest",
        "LSTM"
    ]
)

text = st.text_area(
    "Masukkan Review"
)

if st.button("Prediksi"):

    if selected_model == "LSTM":

        seq = tokenizer.texts_to_sequences(
            [text]
        )

        pad = pad_sequences(
            seq,
            maxlen=50
        )

        prob = lstm_model.predict(
            pad,
            verbose=0
        )[0][0]

        pred = 1 if prob >= 0.5 else 0

        confidence = prob if pred == 1 else (1-prob)

    else:

        vec = vectorizer.transform(
            [text]
        )

        model = models[
            selected_model
        ]

        pred = model.predict(
            vec
        )[0]

        prob = model.predict_proba(
            vec
        )[0]

        confidence = max(prob)

    if pred == 1:
        st.success(
            "😊 Positif"
        )
    else:
        st.error(
            "😠 Negatif"
        )

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )
