"""Flask web application for IBM Watson emotion detection."""

from typing import Any

from flask import Flask, jsonify, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.get("/")
def home() -> str:
    """Render the application home page."""
    return render_template("index.html")


@app.post("/analyze")
def analyze() -> tuple[str, int] | str:
    """Analyze submitted text and render its emotion scores."""
    text = request.form.get("text", "")
    if not text.strip():
        return render_template(
            "index.html", error="Please enter some text to analyze."
        ), 400

    result = emotion_detector(text)
    if result is None:
        return render_template(
            "index.html",
            error=(
                "Emotion analysis is currently unavailable. "
                "Please try again later."
            ),
        ), 503

    return render_template("index.html", result=result, submitted_text=text)


@app.get("/emotionDetector")
def emotion_detector_api() -> tuple[Any, int] | Any:
    """Return a JSON analysis for the course-compatible query parameter."""
    text = request.args.get("textToAnalyze", "")
    if not text.strip():
        return jsonify(error="Please enter some text to analyze."), 400

    result = emotion_detector(text)
    if result is None:
        return jsonify(error="Emotion analysis is currently unavailable."), 503

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
