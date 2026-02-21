from flask import Flask, render_template, request, jsonify
import joblib
from utils.ai_summary import generate_summary

app = Flask(__name__)

# Load model
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    article = request.json['article']

    vect = vectorizer.transform([article])
    prediction = model.predict(vect)[0]
    confidence = max(model.predict_proba(vect)[0]) * 100

    summary = generate_summary(article)

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "summary": summary
    })

if __name__ == "__main__":
    app.run(debug=True)