import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(file_path, scaler=None):
    """
    Preprocess the data by removing 'Attrition', handling categorical variables,
    and scaling numerical features.
    
    If a scaler is provided, use it for scaling; otherwise, fit a new scaler.
    
    Returns:
        X: Preprocessed features
        y: Target variable
        scaler: Fitted scaler
    """
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()  # Strip spaces from column names

    if 'EmployeeID' in data.columns:
        data.drop(['EmployeeID'], axis=1, inplace=True)

    # Separate target variable
    y = data['Attrition']
    X = data.drop(['Attrition'], axis=1)

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if scaler is None:
        scaler = RobustScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        X[numerical_cols] = scaler.transform(X[numerical_cols])

    return X, y, scaler

def load_first_n_entries(file_path, n=2000):
    """
    Load the first n entries from the CSV file and strip column names.
    """
    data = pd.read_csv(file_path, nrows=n)
    data.columns = data.columns.str.strip()  # Strip spaces from column names
    return data

def main():
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'datas', 'final_merged_data_with_work_metrics_delete.csv')
    model_path = os.path.join(current_dir, '..', 'saved_models', 'best_model_0.keras')  # Ensure correct extension

    # Debug: Print paths and existence
    print(f"Data Path: {data_path}")
    print(f"Data Exists: {os.path.exists(data_path)}")
    print(f"Model Path: {model_path}")
    print(f"Model Exists: {os.path.exists(model_path)}")

    # Verify if the data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please ensure the file exists.")
        return

    # Verify if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please ensure the model is trained and saved.")
        return

    # Load the first 2000 entries
    first_2000_data = load_first_n_entries(data_path, n=2000)

    # Debug: Print column names to verify 'Attrition' exists
    print("Columns in the first 2000 entries:", first_2000_data.columns.tolist())

    # Separate actual Attrition values
    try:
        actual_attrition = first_2000_data['Attrition'].values
    except KeyError:
        print("'Attrition' column not found in the first 2000 entries. Please check the CSV file.")
        return

    # Drop 'Attrition' for input features
    input_data = first_2000_data.drop(['Attrition'], axis=1)

    # Preprocess the input data
    # Load the entire dataset to fit the scaler
    X_full, _, scaler = preprocess_data(data_path)
    # Preprocess the first 2000 entries using the fitted scaler
    X_preprocessed, _, _ = preprocess_data(data_path, scaler=scaler)
    X_first_2000 = X_preprocessed.iloc[:2000]

    # Load the trained model
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Make predictions
    predictions_prob = model.predict(X_first_2000)
    predictions = (predictions_prob > 0.5).astype(int).flatten()

    # Calculate evaluation metrics
    accuracy = accuracy_score(actual_attrition, predictions)
    auc = roc_auc_score(actual_attrition, predictions_prob)
    cm = confusion_matrix(actual_attrition, predictions)
    report = classification_report(actual_attrition, predictions, digits=4)

    # Display evaluation metrics
    print("=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Plot Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Leave', 'Leave'], yticklabels=['Not Leave', 'Leave'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()