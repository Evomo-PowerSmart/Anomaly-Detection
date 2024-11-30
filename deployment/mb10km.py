from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)

model = tf.keras.models.load_model("deployment/model_mb10km.h5")

seq_length = 24

@app.route("/")
def home():
    return "Test run"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # input validation and parsing
        if "timestamp" not in data or "features" not in data:
            return jsonify({"error": "Input harus berisi 'timestamp' dan 'features'"}), 400
        
        timestamp = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
        
        features = np.array(data["features"])
        if len(features) != seq_length:
            return jsonify({"error": f"Panjang 'features' harus {seq_length}"}), 400

        features = features.reshape(1, seq_length, 1) 

        prediction = model.predict(features).flatten()[0]

        return jsonify({
            "timestamp": data["timestamp"],
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
