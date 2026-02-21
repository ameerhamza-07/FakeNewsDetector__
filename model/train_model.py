import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

true_df = pd.read_csv("True.csv")
false_df = pd.read_csv("Fake.csv")

true_df["label"] = "REAL"
false_df["label"] = "FAKE"

df = pd.concat([true_df, false_df], ignore_index=True)

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vectorized, y)

accuracy = model.score(X_vectorized, y)
print("Model Accuracy:", accuracy)

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")