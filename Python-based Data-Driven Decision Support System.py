import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
# Create synthetic data
np.random.seed(42)
num_patients = 1000

# Generate features
age = np.random.randint(30, 80, num_patients)
gender = np.random.choice([0, 1], num_patients)  # 0: Female, 1: Male
blood_pressure = np.random.randint(80, 180, num_patients)
cholesterol = np.random.randint(150, 300, num_patients)
heart_rate = np.random.randint(60, 100, num_patients)
smoking = np.random.choice([0, 1], num_patients)
diabetes = np.random.choice([0, 1], num_patients)

# Generate target variable (risk of heart disease)
risk_score = (
    age * 0.3 +
    gender * 5 +
    blood_pressure * 0.2 +
    cholesterol * 0.1 +
    heart_rate * 0.1 +
    smoking * 10 +
    diabetes * 15 +
    np.random.normal(0, 10, num_patients)
)

risk_threshold = np.percentile(risk_score, 75)
heart_disease = (risk_score > risk_threshold).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'BloodPressure': blood_pressure,
    'Cholesterol': cholesterol,
    'HeartRate': heart_rate,
    'Smoking': smoking,
    'Diabetes': diabetes,
    'HeartDisease': heart_disease
})

data.head()
# Features and target variable
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Feature importance
importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 6))
feat_imp.plot(kind='bar')
plt.title('Feature Importance')
plt.ylabel('Importance Score')
plt.show()
# Decision Support Function
def predict_risk(input_data):
    """
    Predicts the risk of heart disease for a patient.

    Parameters:
    - input_data: dict with keys as feature names

    Returns:
    - risk_prediction: int (0: Low Risk, 1: High Risk)
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Feature scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    risk_prediction = model.predict(input_scaled)[0]

    return risk_prediction