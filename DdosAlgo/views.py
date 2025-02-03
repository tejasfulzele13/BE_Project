from django.shortcuts import render
import requests

# Create your views here.
def home(requests):
    return render(requests , 'home.html')

import joblib
import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import StandardScaler

# Load the saved model and preprocessing tools
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
protocol_encoder = joblib.load("protocol_encoder.pkl")
request_encoder = joblib.load("request_encoder.pkl")

@csrf_exempt
def classify_traffic(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Receive JSON data

            # Extract input features
            input_data = [
                data["Source Port"],
                data["Destination Port"],
                data["Packet Size"],
                data["Packet Rate"],
                data["Connection Duration"],
                protocol_encoder.transform([data["Protocol"]])[0],
                request_encoder.transform([data["Request Type"]])[0]
            ]

            # Scale input data
            input_data = np.array(input_data).reshape(1, -1)
            input_data = scaler.transform(input_data)

            # Predict attack type
            prediction = rf_model.predict(input_data)[0]

            # Convert numeric label back to attack type
            attack_classes = ["Botnet", "DDoS", "Normal"]
            result = attack_classes[prediction]

            return JsonResponse({"status": "success", "classification": result})

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return JsonResponse({"status": "error", "message": "Invalid request method"})
