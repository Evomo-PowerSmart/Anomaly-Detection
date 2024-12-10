from flask import Flask, request
from flask_restful import Resource, Api
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)
api = Api(app)

# Load trained models
ahu_model = joblib.load('AHU_2.pkl')
chiller_model = joblib.load('chiller.pkl')
lift_witel_model = joblib.load('lift_witel.pkl')
lift_opmc_model = joblib.load('lift_opmc.pkl')

# Helper function for anomaly detection
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

# AHU Second Floor Predictions
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

# Chiller Model Predictions
class Predict_Chiller(Resource):
    def post(self):
        try:
            input_data = request.json
            if not input_data:
                return {"error": "No input data provided"}, 400

            timestamp = input_data.get("timestamp")
            usage = input_data.get("usage")

            if timestamp is None or usage is None:
                return {
                    "error": "Missing required fields: 'timestamp' and 'usage'"
                }, 400

            hour, weekday = process_timestamp(timestamp)
            features = np.array([[usage, hour, weekday]])
            
            prediction = chiller_model.predict(features)
            is_anomaly = bool(define_anomaly(prediction[0]))

            return {
                "anomaly": is_anomaly
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500

# Lift (Witel) Model Predictions
class Predict_Lift(Resource):
    def post(self):
        try:
            input_data = request.json
            if not input_data:
                return {"error": "No input data provided"}, 400

            timestamp = input_data.get("timestamp")
            usage = input_data.get("usage")

            if timestamp is None or usage is None:
                return {
                    "error": "Missing required fields: 'timestamp' and 'usage'"
                }, 400

            hour, weekday = process_timestamp(timestamp)
            features = np.array([[usage, hour, weekday]])
            
            prediction = lift_witel_model.predict(features)
            is_anomaly = bool(define_anomaly(prediction[0]))

            return {
                "anomaly": is_anomaly
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500

# Lift (OPMC) Model Predictions
class Predict_Lift_Opmc(Resource):
    def post(self):
        try:
            input_data = request.json
            if not input_data:
                return {"error": "No input data provided"}, 400

            timestamp = input_data.get("timestamp")
            usage = input_data.get("usage")

            if timestamp is None or usage is None:
                return {
                    "error": "Missing required fields: 'timestamp' and 'usage'"
                }, 400

            hour, weekday = process_timestamp(timestamp)
            features = np.array([[usage, hour, weekday]])
            
            prediction = lift_opmc_model.predict(features)
            is_anomaly = bool(define_anomaly(prediction[0]))

            return {
                "anomaly": is_anomaly
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500

# Endpoints
api.add_resource(Predict_AHU, '/predict_ahu')
api.add_resource(Predict_Chiller, '/predict_chiller')
api.add_resource(Predict_Lift, '/predict_lift_witel')
api.add_resource(Predict_Lift_Opmc, '/predict_lift_opmc')

if __name__ == '__main__':
    app.run(debug=True)
