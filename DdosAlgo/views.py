from django.shortcuts import render
import requests
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.http import JsonResponse
import pandas as pd
import json
import joblib
import logging
from django.http import JsonResponse
from sklearn.preprocessing import LabelEncoder
from django.views.decorators.csrf import csrf_exempt

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

  





logger = logging.getLogger(__name__)


rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
protocol_encoder = joblib.load("protocol_encoder.pkl")
request_encoder = joblib.load("request_encoder.pkl")


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


from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import pandas as pd
import logging
from .models import AttackLog  # Import the model

logger = logging.getLogger(__name__)

@csrf_exempt  
def classify_attack(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

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

            df["Protocol"] = protocol_encoder.transform(df["Protocol"])
            df["Request Type"] = request_encoder.transform(df["Request Type"])
            df_scaled = scaler.transform(df)

            prediction = rf_model.predict(df_scaled)
            prediction = int(prediction[0])

            if prediction == 2:
                prediction_label = "Normal Traffic"
            elif prediction == 1:
                prediction_label = "DDOS Attack"
            else:
                prediction_label = "Botnet Attack"

            # Save to database
            AttackLog.objects.create(
                source_port=data["Source Port"],
                destination_port=data["Destination Port"],
                packet_size=data["Packet Size (bytes)"],
                packet_rate=data["Packet Rate (pps)"],
                connection_duration=data["Connection Duration (seconds)"],
                protocol=data["Protocol"],
                request_type=data["Request Type"],
                anomaly_score=data["Anomaly Score"],
                prediction=prediction_label
            )

            return JsonResponse({"Prediction": prediction_label})

        except json.JSONDecodeError as e:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)



from django.http import JsonResponse
from .models import AttackLog

def get_traffic_logs(request):
    logs = AttackLog.objects.order_by('-timestamp')[:50]  # Get last 50 logs
    log_list = [
        {
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "source_port": log.source_port,
            "destination_port": log.destination_port,
            "packet_size": log.packet_size,
            "packet_rate": log.packet_rate,
            "connection_duration": log.connection_duration,
            "protocol": log.protocol,
            "request_type": log.request_type,
            "anomaly_score": log.anomaly_score,
            "prediction": log.prediction,
        }
        for log in logs
    ]
    return JsonResponse({"logs": log_list})



from django.shortcuts import render

def traffic_dashboard(requests):
    return render(requests, "dashboard.html")

from django.http import JsonResponse
from .models import AttackLog

def clear_logs(request):
    if request.method == "POST":
        AttackLog.objects.all().delete()  # Delete all logs
        return JsonResponse({"message": "Logs cleared successfully"})
    return JsonResponse({"error": "Invalid request method"}, status=405)
