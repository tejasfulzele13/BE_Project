import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = 'Datasets.csv'  # Replace with your file's path
data = pd.read_csv(file_path, low_memory=False)

# Encode categorical columns
label_encoder = LabelEncoder()
data['Sender_IP'] = label_encoder.fit_transform(data['Sender_IP'])
data['Target_IP'] = label_encoder.fit_transform(data['Target_IP'])
data['Transport_Protocol'] = label_encoder.fit_transform(data['Transport_Protocol'])

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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)


# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)
