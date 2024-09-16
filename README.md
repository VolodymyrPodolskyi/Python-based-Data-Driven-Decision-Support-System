# Python-based-Data-Driven-Decision-Support-System
Python-based Data-Driven Decision Support System for patient care that incorporates predictive analytics algorithms. This system uses synthetic patient data to predict the risk of a certain medical condition (e.g., heart disease) based on various health indicators.
    Data Generation: We create synthetic patient data with features like age, gender, blood pressure, etc., and a target variable indicating the risk of heart disease.
    Data Preprocessing: The data is split into training and testing sets, and features are scaled for better model performance.
    Model Building: A Random Forest Classifier is used to build the predictive model.
    Evaluation: The model's performance is evaluated using classification metrics and a confusion matrix.
    Feature Importance: We visualize which features contribute most to the prediction.
    Decision Support Function: A function predict_risk takes patient data as input and returns a risk prediction.
    Example Usage: We demonstrate how to use the system to predict the risk for a new patient.

The model's accuracy and effectiveness depend on the quality and quantity of data. In real-world applications, you should use real patient data (ensuring compliance with all privacy laws and regulations) and consider more sophisticated models and validation techniques.
Example Usage
# Sample patient data
new_patient = {
    'Age': 55,
    'Gender': 1,           # Male
    'BloodPressure': 140,
    'Cholesterol': 250,
    'HeartRate': 85,
    'Smoking': 1,          # Smoker
    'Diabetes': 0          # Non-diabetic
}

# Predict risk
risk = predict_risk(new_patient)

# Output result
risk_str = 'High Risk' if risk == 1 else 'Low Risk'
print(f"The patient is at {risk_str} of heart disease.")

Output:
Classification Report:
              precision    recall  f1-score   support

           0       0.81      0.79      0.80        99
           1       0.80      0.81      0.80       101

    accuracy                           0.80       200
   macro avg       0.80      0.80      0.80       200
weighted avg       0.80      0.80      0.80       200

The patient is at High Risk of heart disease.
