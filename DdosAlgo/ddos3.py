import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import subprocess

# Define functions for unseen labels and DDoS prevention
def handle_unseen_labels(value, encoder):
    """
    Handle unseen categorical values for LabelEncoder by assigning a default class.
    """
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

def prevent_ddos_attack(sender_ip):
    """
    Block the offending IP address using Windows Firewall.
    """
    try:
        print(f"Blocking IP: {sender_ip}")
        subprocess.run(
            ["netsh", "advfirewall", "firewall", "add", "rule", "name=Block_IP", 
             "dir=in", "action=block", f"remoteip={sender_ip}"],
            check=True
        )
    except Exception as e:
        print(f"Failed to block IP: {sender_ip}. Error: {e}")

def monitor_traffic(new_data, model, scaler):
    """
    Simulate real-time traffic monitoring and DDoS prevention.
    """
    # Preprocess incoming data
    new_data_encoded = new_data.copy()
    sender_ip = str(new_data['Sender_IP'].iloc[0])  # Ensure it's a single value
    
    # Handle unseen IPs and other categorical data
    new_data_encoded['Sender_IP'] = handle_unseen_labels(sender_ip, label_encoder)
    new_data_encoded['Target_IP'] = handle_unseen_labels(str(new_data_encoded['Target_IP'].iloc[0]), label_encoder)
    new_data_encoded['Transport_Protocol'] = handle_unseen_labels(str(new_data_encoded['Transport_Protocol'].iloc[0]), label_encoder)

    # Scale features (use DataFrame to preserve column names)
    X_new = pd.DataFrame([new_data_encoded[updated_features].values[0]], columns=updated_features)
    X_new_scaled = scaler.transform(X_new)

    # Predict and act
    prediction = model.predict(X_new_scaled)
    if prediction[0] == 1:
        print("DDoS attack detected!")
        prevent_ddos_attack(sender_ip)
    else:
        print("Normal traffic.")

# Load the dataset
file_path = 'Datasets.csv'  # Replace with your file path
dataset = pd.read_csv(file_path)

# Columns to use as features for the model
updated_features = [
    'Sender_IP', 'Target_IP', 'Transport_Protocol', 'Duration', 
    'Packets_Sent', 'Packets_Received'
]

# Encode categorical features
label_encoder = LabelEncoder()
for col in ['Sender_IP', 'Target_IP', 'Transport_Protocol']:
    dataset[col] = label_encoder.fit_transform(dataset[col].astype(str))

# Prepare features (X) and target (y)
X = dataset[updated_features]
y = dataset['Label']

# Scale the numerical features
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Train a Random Forest model on the dataset
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Test the model with an example from the dataset
for i in range(1,1000000):
    test_example = dataset.iloc[0:i][updated_features]  # First row for testing
    monitor_traffic(test_example, model, scaler)
 