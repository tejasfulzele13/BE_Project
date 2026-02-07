import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import subprocess

# Sample features used in the model
features = ['Sender_IP', 'Target_IP', 'Transport_Protocol', 'Packet_Size', 'Packet_Count', 'Duration']

# Dummy LabelEncoder and Scaler for demonstration
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Mock data for training the model (replace with your actual dataset)
data = pd.DataFrame({
    'Sender_IP': ['192.168.1.1', '192.168.1.2', '192.168.1.3'],
    'Target_IP': ['10.0.0.1', '10.0.0.2', '10.0.0.3'],
    'Transport_Protocol': ['TCP', 'UDP', 'ICMP'],
    'Packet_Size': [100, 200, 150],
    'Packet_Count': [10, 15, 12],
    'Duration': [0.5, 0.7, 0.6],
    'DDoS': [0, 1, 0]
}) 

# Encode categorical features
for col in ['Sender_IP', 'Target_IP', 'Transport_Protocol']:
    data[col] = label_encoder.fit_transform(data[col])

# Separate features and target
X = data[features]
y = data['DDoS']

# Scale features
scaler.fit(X)
X_scaled = scaler.transform(X)

# Train a simple Random Forest model
model = RandomForestClassifier()
model.fit(X_scaled, y)

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
    X_new = pd.DataFrame([new_data_encoded[features].values[0]], columns=features)
    X_new_scaled = scaler.transform(X_new)

    # Predict and act
    prediction = model.predict(X_new_scaled)
    if prediction[0] == 1:
        print("DDoS attack detected!")
        prevent_ddos_attack(sender_ip)
    else:
        print("Normal traffic.")

# Example traffic data for testing (replace with live traffic data)
test_traffic_df = pd.DataFrame({
    'Sender_IP': ['192.168.2.112'],
    'Target_IP': ['10.0.0.5'],
    'Transport_Protocol': ['TCP'],
    'Packet_Size': [200],
    'Packet_Count': [20],
    'Duration': [0.9]
})

# Monitor traffic
monitor_traffic(test_traffic_df, model, scaler)
