"""Unit tests for Watson response parsing and application error handling."""

import unittest
from unittest.mock import Mock, patch

import requests

import emotion_detection
import server


class EmotionDetectorTests(unittest.TestCase):
    """Verify formatted results using isolated, synthetic HTTP fixtures."""

    def setUp(self) -> None:
        """Build a complete mock response for each expected dominant emotion."""
        self.expected = {
            "I am glad this happened": "joy",
            "I am really mad about this": "anger",
            "I feel disgusted just hearing about this": "disgust",
            "I am so sad about this": "sadness",
            "I am really afraid that this will happen": "fear",
        }

    def make_response(self, dominant: str) -> Mock:
        """Construct a service-shaped fixture; scores are unit-test-only."""
        scores = {
            "anger": 0.10,
            "disgust": 0.10,
            "fear": 0.10,
            "joy": 0.10,
            "sadness": 0.10,
        }
        scores[dominant] = 0.60
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "emotionPredictions": [{"emotion": scores}]
        }
        return response

    def test_formatted_results_and_dominant_emotion(self) -> None:
        """Return exactly the five scores and the expected dominant label."""
        for text, dominant in self.expected.items():
            with self.subTest(text=text):
                response = self.make_response(dominant)
                with patch.object(
                    emotion_detection.requests, "post", return_value=response
                ) as post:
                    result = emotion_detection.emotion_detector(text)

                self.assertEqual(
                    set(result),
                    {
                        "anger",
                        "disgust",
                        "fear",
                        "joy",
                        "sadness",
                        "dominant_emotion",
                    },
                )
                self.assertEqual(result["dominant_emotion"], dominant)
                self.assertEqual(result[dominant], 0.60)
                post.assert_called_once()
                self.assertEqual(
                    post.call_args.kwargs["headers"][
                        "grpc-metadata-mm-model-id"
                    ],
                    emotion_detection.MODEL_ID,
                )

    def test_http_400_returns_none_without_parsing(self) -> None:
        """Do not parse an unsuccessful 400 response."""
        response = Mock(status_code=400)
        with patch.object(
            emotion_detection.requests, "post", return_value=response
        ):
            self.assertIsNone(emotion_detection.emotion_detector("invalid"))
        response.json.assert_not_called()

    def test_other_http_error_returns_none_without_parsing(self) -> None:
        """Treat all non-200 HTTP statuses as unavailable."""
        response = Mock(status_code=503)
        with patch.object(
            emotion_detection.requests, "post", return_value=response
        ):
            self.assertIsNone(emotion_detection.emotion_detector("text"))
        response.json.assert_not_called()

    def test_request_failure_returns_none(self) -> None:
        """Catch request transport errors without propagating them."""
        with patch.object(
            emotion_detection.requests,
            "post",
            side_effect=requests.RequestException("network unavailable"),
        ):
            self.assertIsNone(emotion_detection.emotion_detector("text"))

    def test_blank_text_does_not_call_service(self) -> None:
        """Reject empty and whitespace-only input before making a request."""
        with patch.object(emotion_detection.requests, "post") as post:
            self.assertIsNone(emotion_detection.emotion_detector("  \n"))
        post.assert_not_called()

    def test_unexpected_success_payload_returns_none(self) -> None:
        """Handle a malformed successful response without raising."""
        response = Mock(status_code=200)
        response.json.return_value = {"unexpected": "payload"}
        with patch.object(
            emotion_detection.requests, "post", return_value=response
        ):
            self.assertIsNone(emotion_detection.emotion_detector("text"))


class FlaskRouteTests(unittest.TestCase):
    """Check form validation and rendered results using Flask's test client."""

    def setUp(self) -> None:
        """Create a test client without starting the development server."""
        self.client = server.app.test_client()

    def test_home_page_loads(self) -> None:
        """Render the application form at the root URL."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Text to analyze", response.data)

    def test_blank_submission_skips_service(self) -> None:
        """Show a form error for blank input without calling Watson."""
        with patch.object(server, "emotion_detector") as detector:
            response = self.client.post("/analyze", data={"text": "  \n"})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Please enter some text", response.data)
        detector.assert_not_called()

    def test_unavailable_service_is_rendered_safely(self) -> None:
        """Show a safe error when the detector returns None."""
        with patch.object(server, "emotion_detector", return_value=None):
            response = self.client.post("/analyze", data={"text": "hello"})
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"currently unavailable", response.data)

    def test_successful_result_is_rendered(self) -> None:
        """Display all scores and the dominant emotion from the detector."""
        result = {
            "anger": 0.1,
            "disgust": 0.1,
            "fear": 0.1,
            "joy": 0.6,
            "sadness": 0.1,
            "dominant_emotion": "joy",
        }
        with patch.object(server, "emotion_detector", return_value=result):
            response = self.client.post("/analyze", data={"text": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Dominant emotion: joy", response.data)
        self.assertIn(b"sadness", response.data)


if __name__ == "__main__":
    unittest.main()
