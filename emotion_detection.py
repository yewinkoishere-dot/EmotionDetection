
"""
Emotion detection module using the Watson NLP Emotion Predict API.
"""

import os
import requests

API_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
API_KEY = os.getenv("WATSON_API_KEY", "")


def emotion_detector(text_to_analyze):
    """
    Analyze text and return emotion scores and the dominant emotion.
    """
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
        "Content-Type": "application/json",
    }

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        if response.status_code == 400:
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            }

        response.raise_for_status()
        data = response.json()

        emotions = data["emotionPredictions"][0]["emotion"]
        scores = {
            "anger": emotions["anger"],
            "disgust": emotions["disgust"],
            "fear": emotions["fear"],
            "joy": emotions["joy"],
            "sadness": emotions["sadness"],
        }

        scores["dominant_emotion"] = max(
            scores,
            key=scores.get,
        )
        return scores

    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
