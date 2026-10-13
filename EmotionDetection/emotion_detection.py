"""Package-facing import of the single emotion detection implementation."""

from emotion_detection import EMOTION_NAMES, MODEL_ID, SERVICE_URL, emotion_detector

__all__ = ["EMOTION_NAMES", "MODEL_ID", "SERVICE_URL", "emotion_detector"]
