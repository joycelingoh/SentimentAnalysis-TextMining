import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "SVM":
        SVC(
            probability=True,
            kernel="linear"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

for name, model in models.items():

    model.fit(X_train_vec, y_train)

    pred = model.predict(X_test_vec)

    acc = accuracy_score(
        y_test,
        pred
    )

    print(f"{name}: {acc:.4f}")

    filename = (
        name.lower()
        .replace(" ","_")
        + ".pkl"
    )

    joblib.dump(
        model,
        filename
    )

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("Semua model ML berhasil disimpan")
