import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AI_API_KEY")
API_URL = os.getenv("AI_API_URL")

def generate_summary(text):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Summarize this news in simple words."},
            {"role": "user", "content": text}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Summary not available."