"""Unit tests for the EmotionDetection package and Flask server."""

import unittest
from unittest.mock import Mock, patch
import requests

from EmotionDetection.emotion_detection import emotion_detector
import server


class TestEmotionDetectionStatements(unittest.TestCase):
    """Test the five required Coursera statements and their dominant emotions."""

    STATEMENT_EMOTIONS = {
        "I am glad this happened": {
            "anger": 0.02, "disgust": 0.01, "fear": 0.01, "joy": 0.95, "sadness": 0.01
        },
        "I am really mad about this": {
            "anger": 0.90, "disgust": 0.03, "fear": 0.02, "joy": 0.01, "sadness": 0.04
        },
        "I feel disgusted just hearing about this": {
            "anger": 0.04, "disgust": 0.88, "fear": 0.03, "joy": 0.01, "sadness": 0.04
        },
        "I am so sad about this": {
            "anger": 0.03, "disgust": 0.02, "fear": 0.02, "joy": 0.01, "sadness": 0.92
        },
        "I am really afraid that this will happen": {
            "anger": 0.02, "disgust": 0.02, "fear": 0.91, "joy": 0.01, "sadness": 0.04
        },
    }

    def _mock_post(self, url, *args, **kwargs):
        text = kwargs.get("json", {}).get("raw_document", {}).get("text", "")
        mock_resp = Mock()
        if text in self.STATEMENT_EMOTIONS:
            mock_resp.status_code = 200
            mock_resp.text = Mock()
            scores = self.STATEMENT_EMOTIONS[text]
            import json
            mock_resp.text = json.dumps({"emotionPredictions": [{"emotion": scores}]})
            return mock_resp
        mock_resp.status_code = 400
        return mock_resp

    def test_emotion_detector(self):
        """Verify the five benchmark statements produce their required dominant emotions."""
        with patch("requests.post", side_effect=self._mock_post):
            # 1. 'I am glad this happened' -> joy
            result_1 = emotion_detector("I am glad this happened")
            self.assertEqual(result_1["dominant_emotion"], "joy")

            # 2. 'I am really mad about this' -> anger
            result_2 = emotion_detector("I am really mad about this")
            self.assertEqual(result_2["dominant_emotion"], "anger")

            # 3. 'I feel disgusted just hearing about this' -> disgust
            result_3 = emotion_detector("I feel disgusted just hearing about this")
            self.assertEqual(result_3["dominant_emotion"], "disgust")

            # 4. 'I am so sad about this' -> sadness
            result_4 = emotion_detector("I am so sad about this")
            self.assertEqual(result_4["dominant_emotion"], "sadness")

            # 5. 'I am really afraid that this will happen' -> fear
            result_5 = emotion_detector("I am really afraid that this will happen")
            self.assertEqual(result_5["dominant_emotion"], "fear")


class TestEmotionDetectionEdgeCases(unittest.TestCase):
    """Test HTTP status code handling and response structure validation."""

    def test_http_400_status_code(self):
        """Verify status code 400 returns dictionary with all None values."""
        mock_resp = Mock(status_code=400)
        with patch("requests.post", return_value=mock_resp):
            result = emotion_detector("")
        expected = {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
        self.assertEqual(result, expected)

    def test_other_unsuccessful_status_code(self):
        """Verify non-200/non-400 status code returns None."""
        mock_resp = Mock(status_code=500)
        with patch("requests.post", return_value=mock_resp):
            result = emotion_detector("some text")
        self.assertIsNone(result)

    def test_request_exception(self):
        """Verify network request exceptions return None gracefully."""
        with patch("requests.post", side_effect=requests.RequestException("Network error")):
            result = emotion_detector("some text")
        self.assertIsNone(result)

    def test_malformed_json_response(self):
        """Verify malformed JSON responses return None safely."""
        mock_resp = Mock(status_code=200, text="not json")
        with patch("requests.post", return_value=mock_resp):
            result = emotion_detector("some text")
        self.assertIsNone(result)


class TestFlaskServerRoutes(unittest.TestCase):
    """Test the Flask application web routes and error handling."""

    def setUp(self):
        self.app = server.app.test_client()
        self.app.testing = True

    def test_index_route(self):
        """Verify the root page renders successfully."""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"NLP - Emotion Detection", response.data)

    def test_blank_input_handling(self):
        """Verify blank input returns the exact required error message."""
        mock_result = {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
        with patch("server.emotion_detector", return_value=mock_result):
            response = self.app.get("/emotionDetector?textToAnalyze=")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_data(as_text=True), "Invalid input! Try again.")

    def test_valid_input_formatting(self):
        """Verify successful emotion analysis returns formatted string."""
        mock_result = {
            "anger": 0.01,
            "disgust": 0.02,
            "fear": 0.03,
            "joy": 0.94,
            "sadness": 0.01,
            "dominant_emotion": "joy",
        }
        with patch("server.emotion_detector", return_value=mock_result):
            response = self.app.get("/emotionDetector?textToAnalyze=I%20am%20so%20happy")
            self.assertEqual(response.status_code, 200)
            text = response.get_data(as_text=True)
            self.assertIn("'joy': 0.94", text)
            self.assertIn("The dominant emotion is joy.", text)


if __name__ == "__main__":
    unittest.main()
