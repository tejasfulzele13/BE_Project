
import pandas as pd
import json
from django.http import JsonResponse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = "Dataset01.csv"  # Adjust path as needed
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

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define multiple models (excluding XGBoost)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    #"Logistic Regression": LogisticRegression(max_iter=1000),
   #"SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    #"Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = round(accuracy * 100, 2)  # Convert to percentage
print(results)
# Django view function to return model results
def model_results(request):
    return JsonResponse(results)