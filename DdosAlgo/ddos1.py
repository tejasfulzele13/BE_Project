import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import subprocess
import time

# Load the dataset
file_path = 'Datasets.csv'  # Replace with your file's path
data = pd.read_csv(file_path, low_memory=False)

# Encode categorical columns
label_encoder = LabelEncoder()
for col in ['Sender_IP', 'Target_IP', 'Transport_Protocol']:
    data[col] = label_encoder.fit_transform(data[col])

# Define features and target
features = [
    'Sender_IP', 'Sender_Port', 'Target_IP', 'Target_Port',
    'Transport_Protocol', 'Duration', 'AvgDuration', 'PBS', 
    'AvgPBS', 'TBS', 'PBR', 'AvgPBR', 'TBR', 'Missed_Bytes', 
    'Packets_Sent', 'Packets_Received', 'SRPR'
]
target = 'Label'

X = data[features]
y = data[target]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest model with balanced class weights
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Helper functions
def prevent_ddos_attack(sender_ip):
    """Block the offending IP address using firewall."""
    try:
        print(f"Blocking IP: {sender_ip}")
        subprocess.run(["iptables", "-A", "INPUT", "-s", sender_ip, "-j", "DROP"], check=True)
    except Exception as e:
        print(f"Failed to block IP: {sender_ip}. Error: {e}")

def handle_unseen_labels(value, encoder):
    """Handle unseen labels by adding them dynamically to the encoder."""
    value = str(value)
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

def monitor_traffic(new_data, model, scaler):
    """Simulate real-time traffic monitoring and DDoS prevention."""
    new_data_encoded = new_data.copy()
    try:
        sender_ip = str(new_data['Sender_IP'].iloc[0])
        
        # Handle unseen labels
        new_data_encoded['Sender_IP'] = handle_unseen_labels(sender_ip, label_encoder)
        new_data_encoded['Target_IP'] = handle_unseen_labels(str(new_data_encoded['Target_IP'].iloc[0]), label_encoder)
        new_data_encoded['Transport_Protocol'] = handle_unseen_labels(str(new_data_encoded['Transport_Protocol'].iloc[0]), label_encoder)

        # Scale features
        X_new = scaler.transform(new_data_encoded[features])

        # Predict and act
        prediction = model.predict(X_new)
        print(f"Prediction: {prediction}, Actual data: {new_data.iloc[0]}")
        if prediction[0] == 1:
            print("DDoS attack detected!")
            prevent_ddos_attack(sender_ip)
        else:
            print("Normal traffic.")
    except Exception as e:
        print(f"Error in monitoring traffic: {e}")

# Simulate new traffic data (Replace this with live data in production)
test_traffic = {
    'Sender_IP': '192.168.2.112', 'Sender_Port': 0, 'Target_IP': '75.126.101.175', 'Target_Port': 443,
    'Transport_Protocol': 1, 'Duration': 4.28, 'AvgDuration': 3, 'PBS': 1174, 
    'AvgPBS': 0, 'TBS': 1894, 'PBR': 11462, 'AvgPBR': 2, 'TBR': 12462, 'Missed_Bytes': 0, 
    'Packets_Sent': 8, 'Packets_Received': 15, 'SRPR': 0.833333
}
test_traffic_df = pd.DataFrame([test_traffic])

# Simulate real-time monitoring
try:
    while True:
        monitor_traffic(test_traffic_df, model, scaler)
        time.sleep(5)  # Check every 5 seconds
except KeyboardInterrupt:
    print("Stopping real-time monitoring.")
