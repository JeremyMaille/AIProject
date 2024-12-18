import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from reseau_neurone.parallel_trainer import ParallelTrainer
import os
import json
from datetime import datetime
import sys

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

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
    scaler = RobustScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler

def main():
    # Load and preprocess data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'datas', 'final_merged_data_with_work_metrics_delete.csv')
    trainer = ParallelTrainer()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data_path)

    # Start training
    results = trainer.train_parallel(X_train, y_train, X_test, y_test)

    if not results:
        print("No successful models were trained. Please check the training logs for issues.")
        sys.exit(1)

    # Sort results by AUC
    results.sort(key=lambda x: x['metrics']['auc'], reverse=True)

    # Save top 5 models
    best_models_dir = os.path.join('saved_models')
    os.makedirs(best_models_dir, exist_ok=True)
    for i, result in enumerate(results[:5]):
        src_model_path = result['model_path']
        dest_model_path = os.path.join(best_models_dir, f"best_model_{i}.h5")

        try:
            os.rename(src_model_path, dest_model_path)
            print(f"Best Model {i} saved to {dest_model_path}")
        except Exception as e:
            print(f"Error saving Best Model {i}: {str(e)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results')
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"training_results_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Training results saved to {result_file}")

    # Feature importance analysis
    importances = trainer.feature_importance(X_train.columns)
    if importances is not None:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        importances.sort_values().plot(kind='barh')
        plt.title('Feature Importance')
        feature_importance_path = os.path.join(results_dir, f"feature_importance_{timestamp}.png")
        plt.savefig(feature_importance_path)
        plt.close()
        print(f"Feature importance plot saved to {feature_importance_path}")
    else:
        print("Feature importance analysis was not performed.")

if __name__ == "__main__":
    main()