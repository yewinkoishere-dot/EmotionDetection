from flask import Flask, jsonify, render_template, request
from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/emotionDetector", methods=["GET"])
def detect_emotion():
    text_to_analyze = request.args.get("textToAnalyze", "")

    if not text_to_analyze.strip():
        return jsonify({
            "error": "Please enter some text to analyze."
        }), 400

    result = emotion_detector(text_to_analyze)

    if result["dominant_emotion"] is None:
        return jsonify({
            "error": "Unable to analyze the text."
        }), 400

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
