"""Flask server for the IBM Coursera emotion detection project."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.get("/")
def home() -> str:
    """Render the text entry page."""
    return render_template("index.html")


@app.get("/emotionDetector")
def emotion_detector_route() -> str:
    """Return the assignment-formatted response for the query text."""
    text = request.args.get("textToAnalyze", "")
    if not text.strip():
        return "Invalid input! Try again."

    result = emotion_detector(text)
    if result is None:
        return "Emotion analysis unavailable. Please try again."

    return (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, 'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, 'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. The dominant emotion is "
        f"{result['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
