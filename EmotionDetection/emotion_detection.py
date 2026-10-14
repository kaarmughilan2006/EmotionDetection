"""IBM Watson NLP emotion detection client for Coursera project."""

import json
import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def emotion_detector(text_to_analyse):
    """Analyze text and return scores for five emotions plus dominant emotion."""
    payload = {"raw_document": {"text": text_to_analyse}}

    try:
        response = requests.post(URL, json=payload, headers=HEADERS, timeout=10)
    except requests.RequestException:
        return None

    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    if response.status_code == 200:
        try:
            response_dict = json.loads(response.text)
            emotions = response_dict["emotionPredictions"][0]["emotion"]
            anger_score = emotions["anger"]
            disgust_score = emotions["disgust"]
            fear_score = emotions["fear"]
            joy_score = emotions["joy"]
            sadness_score = emotions["sadness"]

            scores = {
                "anger": anger_score,
                "disgust": disgust_score,
                "fear": fear_score,
                "joy": joy_score,
                "sadness": sadness_score,
            }
            dominant_emotion = max(scores, key=scores.get)

            return {
                "anger": anger_score,
                "disgust": disgust_score,
                "fear": fear_score,
                "joy": joy_score,
                "sadness": sadness_score,
                "dominant_emotion": dominant_emotion,
            }
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return None

    return None
