# Emotion Detection Application

A small Flask application that sends English text to IBM Watson NLP's Emotion Detection service and displays the returned anger, disgust, fear, joy, and sadness scores, together with the dominant emotion. It includes a compatibility module, a Python package, and isolated unit tests that mock the external HTTP response.

**Repository:** https://github.com/kaarmughilan2006/EmotionDetection

## Technologies

- Python 3
- Requests
- Flask
- IBM Watson NLP Emotion Detection
- `unittest`
- Pylint

## Installation

Use Python 3.10 or later. From the project root, create and activate a virtual environment, then install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pylint
```

The Watson NLP service is an external course endpoint; internet access is required for live emotion analysis. The application returns an unavailable result if the service cannot be reached.

## Run the application

```sh
python3 server.py
```

Open http://127.0.0.1:5000. Submit text using the form. A course-compatible JSON endpoint is also available at `/emotionDetector?textToAnalyze=...`.

## Unit tests

The tests replace the HTTP request with mock responses so they are deterministic and do not depend on Watson being online:

```sh
python3 -m unittest test_emotion_detection.py -v
```

## Static code analysis

```sh
python3 -m pylint server.py
python3 -m pylint server.py emotion_detection.py EmotionDetection/emotion_detection.py test_emotion_detection.py
```

Both commands score 10.00/10 in this environment. Including `EmotionDetection/__init__.py` in the same Pylint command also reports Pylint's package-name convention warning because the assignment requires the package's exact CamelCase name; the package name is not changed to hide that warning.

## Project structure

```text
EmotionDetection/
├── __init__.py
└── emotion_detection.py
README.md
emotion_detection.py
requirements.txt
server.py
test_emotion_detection.py
templates/
└── index.html
submissions/
```
