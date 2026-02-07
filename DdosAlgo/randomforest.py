import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = "Dataset.csv"  # Update with your actual path
df = pd.read_csv(file_path)

# Drop non-relevant columns
df_cleaned = df.drop(columns=["Timestamp", "Source IP", "Destination IP"])

# Encode categorical variables
label_encoders = {}
for col in ["Protocol", "Request Type", "Attack Type"]:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

# Split features and target
X = df_cleaned.drop(columns=["Attack Type"])
y = df_cleaned["Attack Type"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoders["Attack Type"].classes_)

# Print results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

import joblib

# Save the trained Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save the scaler for preprocessing
joblib.dump(label_encoders["Protocol"], "protocol_encoder.pkl")
joblib.dump(label_encoders["Request Type"], "request_encoder.pkl")
