"""Client for IBM Watson NLP emotion detection."""

from typing import Any

import requests

SERVICE_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTION_NAMES = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze: str) -> dict[str, Any] | None:
    """Return Watson's five emotion scores and their dominant emotion.

    Return ``None`` for unsuccessful HTTP responses, request failures, or an
    unexpected service payload. The request is sent only for non-empty text.
    """
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return None

    headers = {"grpc-metadata-mm-model-id": MODEL_ID}
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(
            SERVICE_URL,
            headers=headers,
            json=payload,
            timeout=8,
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        body = response.json()
        emotions = body["emotionPredictions"][0]["emotion"]
        scores = {name: float(emotions[name]) for name in EMOTION_NAMES}
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    return {
        **scores,
        "dominant_emotion": max(scores, key=scores.__getitem__),
    }
