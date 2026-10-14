templates/
# Final Project

**Project Name: Final Project**

This project analyzes text with IBM Watson NLP's Emotion Detection service and provides the result through a Flask web application. The detector reports five emotions—anger, disgust, fear, joy, and sadness—and identifies the highest-scoring emotion as `dominant_emotion`.

## Technologies

Python 3, Requests, Flask, IBM Watson NLP Emotion Detection, `unittest`, and Pylint.

## Install dependencies

From the project root, create and activate a virtual environment, then install the runtime requirements:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pylint
```

Live analysis requires network access to the course Watson endpoint. Unit tests mock HTTP responses and do not require network access.

## Run the application

```sh
python3 server.py
```

Open http://localhost:5000. The course-compatible route is `/emotionDetector?textToAnalyze=...` and returns the formatted scores as text. Blank input returns `Invalid input! Try again.`

## Test the emotion detector

```sh
python3 -m unittest test_emotion_detection.py -v
```

Run static analysis with:

```sh
python3 -m pylint server.py
```
