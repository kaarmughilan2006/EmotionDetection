"""IBM Watson NLP emotion detection client used by the Coursera project."""

import requests

SERVICE_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTION_NAMES = ("anger", "disgust", "fear", "joy", "sadness")
HTTP_400_RESULT = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}


def emotion_detector(text_to_analyse):
    """Return Watson's emotion scores for ``text_to_analyse``.

    A 400 response returns the assignment-specified dictionary of ``None``
    values. Other unsuccessful HTTP responses, network errors, and malformed
    successful response bodies return ``None``.
    """
    if not isinstance(text_to_analyse, str) or not text_to_analyse.strip():
        return None

    headers = {"grpc-metadata-mm-model-id": MODEL_ID}
    payload = {"raw_document": {"text": text_to_analyse}}

    try:
        response = requests.post(
            SERVICE_URL,
            headers=headers,
            json=payload,
            timeout=8,
        )
    except requests.RequestException:
        return None

    if response.status_code == 400:
        return HTTP_400_RESULT.copy()
    if response.status_code != 200:
        return None

    try:
        emotions = response.json()["emotionPredictions"][0]["emotion"]
        scores = {name: emotions[name] for name in EMOTION_NAMES}
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    return {
        **scores,
        "dominant_emotion": max(scores, key=scores.__getitem__),
    }
