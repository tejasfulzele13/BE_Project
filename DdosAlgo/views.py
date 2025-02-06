from django.shortcuts import render
import requests
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.http import JsonResponse

# Create your views here.
def home(requests):
    return render(requests , 'home.html')

def authenticate_user(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({"status": "success"})
        else:
            return JsonResponse({"status": "error", "message": "Invalid username or password"})

    return JsonResponse({"status": "error", "message": "Invalid request"})

def dashboard(request):
    return render(request, 'dashboard.html')  # Create this file for the next page after login


import pandas as pd
import json
import joblib
import logging
from django.http import JsonResponse
from sklearn.preprocessing import LabelEncoder
from django.views.decorators.csrf import csrf_exempt

# Set up logging
logger = logging.getLogger(__name__)

# Load the trained model and preprocessing objects
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
protocol_encoder = joblib.load("protocol_encoder.pkl")
request_encoder = joblib.load("request_encoder.pkl")

# Columns used during training for feature scaling
feature_columns = [
    "Source Port",
    "Destination Port",
    "Packet Size (bytes)",
    "Packet Rate (pps)",
    "Connection Duration (seconds)",
    "Protocol",
    "Request Type",
    "Anomaly Score"
]

@csrf_exempt  # Allow POST requests without CSRF token (for testing purposes)
def classify_attack(request):
    if request.method == 'POST':
        try:
            # Log the incoming request for debugging
            logger.debug(f"Received request data: {request.body}")

            # Load the input data
            data = json.loads(request.body)
            
            # Check if all required fields are present in the request
            missing_fields = [field for field in feature_columns if field not in data]
            if missing_fields:
                logger.error(f"Missing fields: {missing_fields}")
                return JsonResponse({"error": f"Missing fields: {', '.join(missing_fields)}"}, status=400)
            
            # Prepare the incoming data as a pandas DataFrame
            features = {
                "Source Port": [data["Source Port"]],
                "Destination Port": [data["Destination Port"]],
                "Packet Size (bytes)": [data["Packet Size (bytes)"]],
                "Packet Rate (pps)": [data["Packet Rate (pps)"]],
                "Connection Duration (seconds)": [data["Connection Duration (seconds)"]],
                "Protocol": [data["Protocol"]],
                "Request Type": [data["Request Type"]],
                "Anomaly Score": [data["Anomaly Score"]]
            }
            df = pd.DataFrame(features)

            # Encode categorical variables
            try:
                df["Protocol"] = protocol_encoder.transform(df["Protocol"])
                df["Request Type"] = request_encoder.transform(df["Request Type"])
            except KeyError as e:
                logger.error(f"Encoding error: {e}")
                return JsonResponse({"error": f"Encoding error: {e}"}, status=400)

            # Standardize the features (using the same scaler used during training)
            try:
                df_scaled = scaler.transform(df)
            except ValueError as e:
                logger.error(f"Standardization error: {e}")
                return JsonResponse({"error": f"Standardization error: {e}"}, status=400)

            # Predict using the trained model
            prediction = rf_model.predict(df_scaled)

            # Convert the prediction from int64 to a regular int
            prediction = int(prediction[0])

            # Return the predicted attack type
            return JsonResponse({"Prediction": prediction})

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)

