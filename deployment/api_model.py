from flask import Flask, request
from flask_restful import Resource, Api
import joblib
import numpy as np
from datetime import datetime

# Initialize Flask app and API
app = Flask(__name__)
api = Api(app)

# Load the trained model
ahu_model = joblib.load('AHU_2.pkl')

# -1 for anomaly and 1 is not anomaly
def define_anomaly(predictions):
    if isinstance(predictions, list):
        return [pred == -1 for pred in predictions]
    else:
        return predictions == -1

def process_timestamp(timestamp):
    try:
        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') 
        hour = dt.hour
        weekday = dt.weekday()  # Monday = 0, Sunday = 6
        return hour, weekday
    except ValueError as e:
        raise ValueError("Invalid timestamp format. 'YYYY-MM-DD HH:MM:SS'.") from e

# Define a Resource for predictions
class Predict_AHU(Resource):
    def post(self):
        try:
            # Parse JSON input
            input_data = request.json
            if not input_data:
                return {"error": "No input data provided"}, 400
            
            # Extract and process timestamp
            timestamp = input_data.get("timestamp")
            usage = input_data.get("usage")
            
            if timestamp is None or usage is None:
                return {
                    "error": "Missing required fields: 'timestamp' and 'usage'"
                }, 400

            # Process the timestamp to extract hour and weekday
            hour, weekday = process_timestamp(timestamp)
            
            # Prepare data for the model
            features = np.array([[usage, hour, weekday]])
            
            # Make a prediction
            prediction = ahu_model.predict(features)
            
            is_anomaly = bool(define_anomaly(prediction[0]))

            # Return the result
            return {
                "anomaly": is_anomaly
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500

# endpoint
api.add_resource(Predict_AHU, '/predict_ahu')


if __name__ == '__main__':
    app.run(debug=True)
