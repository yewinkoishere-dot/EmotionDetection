import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy(self, mock_post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.01,
                    "disgust": 0.01,
                    "fear": 0.02,
                    "joy": 0.90,
                    "sadness": 0.06
                }
            }]
        }
        mock_post.return_value = response

        result = emotion_detector("I am happy")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger(self, mock_post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.90,
                    "disgust": 0.01,
                    "fear": 0.02,
                    "joy": 0.01,
                    "sadness": 0.06
                }
            }]
        }
        mock_post.return_value = response

        result = emotion_detector("I am angry")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_bad_request(self, mock_post):
        response = Mock()
        response.status_code = 400
        mock_post.return_value = response

        result = emotion_detector("Test")
        self.assertIsNone(result["dominant_emotion"])

    def test_blank_input(self):
        result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
