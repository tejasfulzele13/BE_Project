import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load the dataset
file_path = "Dataset01.csv"
df = pd.read_csv(file_path)

# Drop non-relevant columns
df_cleaned = df.drop(columns=["Timestamp", "Source IP", "Destination IP"])

# Encode categorical variablesb
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

# Compute Class Weights (Alternative to SMOTE)
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Focal Loss for better handling of imbalanced classes
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
    
    def forward(self, inputs, targets):
        # Get log softmax
        logpt = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-logpt)
        
        # Compute the Focal loss
        loss = (1 - pt) ** self.gamma * logpt
        
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        
        if self.reduce:
            return loss.mean()
        else:
            return loss

# Define an Improved ResNet Model
class ResNetTabular(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ResNetTabular, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

        self.shortcut = nn.Linear(input_dim, 256)  # Residual connection
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.shortcut.weight)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # Add residual connection
        out = self.relu(out)

        out = self.fc3(out)
        return out

# Initialize Model
input_dim = X_train.shape[1]
num_classes = len(label_encoders["Attack Type"].classes_)
model = ResNetTabular(input_dim, num_classes).to(device)

# Loss, Optimizer, and Learning Rate Scheduler
criterion = FocalLoss(alpha=class_weights_tensor.to(device), gamma=2)  # Use Focal Loss
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # L2 regularization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)

# Move tensors to GPU if available
X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

# Training loop with Early Stopping
num_epochs = 150  # Increased epochs for better learning
batch_size = 512  # Mini-batch training
best_loss = float("inf")
patience = 20  # Stop training if no improvement in 20 epochs
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Compute average loss
    epoch_loss /= (len(X_train_tensor) // batch_size)
    scheduler.step(epoch_loss)  # Adjust learning rate

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Early Stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered!")
        break

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = torch.argmax(y_pred_tensor, axis=1).cpu().numpy()  # Convert predictions to numpy

# Convert y_test_tensor to numpy for comparison
y_test_numpy = y_test_tensor.cpu().numpy()  # Convert y_test to numpy

accuracy = accuracy_score(y_test_numpy, y_pred)  # Compare numpy arrays
report = classification_report(y_test_numpy, y_pred, target_names=label_encoders["Attack Type"].classes_)

print("\nFinal Accuracy:", accuracy)
print("\nClassification Report:\n", report)
