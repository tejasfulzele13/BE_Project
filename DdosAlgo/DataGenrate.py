import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate random IP addresses
def generate_ip():
    return f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}"

# Define parameters
num_rows = 100000
start_time = datetime(2023, 10, 1, 12, 0, 0)
end_time = datetime(2023, 10, 2, 12, 0, 0)
protocols = ['TCP', 'UDP', 'ICMP', 'SCTP', 'IGMP']
request_types = ['GET', 'POST', 'SYN', 'ACK', 'PUT', 'PATCH', 'DELETE']
ports = [
    80, 443, 22, 53, 8080, 3389, 445, 23, 25, 3306, 5432, 21, 21, 110, 995, 139, 8081, 254, 143, 25
]

# Generate synthetic data
timestamps = pd.date_range(start=start_time, end=end_time, periods=num_rows)

# Convert timestamps to numpy array and shuffle
timestamps = np.array(timestamps)
np.random.shuffle(timestamps)

# Create the DataFrame
data = {
    'Timestamp': pd.to_datetime(timestamps),
    'Source IP': [generate_ip() for _ in range(num_rows)],
    'Destination IP': [generate_ip() for _ in range(num_rows)],
    'Source Port': np.random.randint(1024, 65535, num_rows),
    'Destination Port': np.random.choice(ports, num_rows),
    'Packet Size (bytes)': np.random.choice([64, 128, 256, 512, 768, 1024, 2048, 4096], num_rows),
    'Packet Rate (pps)': np.random.randint(50, 500, num_rows),  # Increase packet rate range for better attack diversity
    'Connection Duration (seconds)': np.random.randint(1, 30, num_rows),  # Increase duration range for more variability
    'Protocol': np.random.choice(protocols, num_rows),
    'Request Type': np.random.choice(request_types, num_rows),
    'Anomaly Score': np.round(np.random.uniform(0, 1, num_rows), 2),
    'Attack Type': np.random.choice(['Normal', 'DDoS', 'Botnet'], num_rows, p=[0.8, 0.1, 0.1])  # Balance Attack Types
}

df = pd.DataFrame(data)

# Apply attack patterns
df.loc[
    (df['Packet Rate (pps)'] > 200) & 
    (df['Connection Duration (seconds)'] < 10) & 
    (df['Anomaly Score'] > 0.6), 
    'Attack Type'
] = 'DDoS'

df.loc[
    (df['Packet Size (bytes)'] > 512) & 
    (df['Destination Port'].isin([22, 3389, 445, 23, 25, 3306, 5432])) & 
    (df['Anomaly Score'] > 0.7),
    'Attack Type'
] = 'Botnet'

# Save to CSV
df.to_csv('network_traffic_100k_with_patterns.csv', index=False)
print("Dataset with patterns generated and saved as 'network_traffic_100k_with_patterns.csv'")
