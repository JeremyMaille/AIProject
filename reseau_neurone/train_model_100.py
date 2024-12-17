import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Preprocess Data
def preprocess_data(file_path):
    data = pd.read_csv("datas\\final_merged_data.csv")
    data.columns = data.columns.str.strip()

    if 'Attrition' not in data.columns:
        raise KeyError("The 'Attrition' column is missing.")

    if 'EmployeeID' in data.columns:
        data.drop(['EmployeeID'], axis=1, inplace=True)

    # Convert boolean columns to integers
    bool_columns = data.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        data[col] = data[col].astype(int)

    # Get mappings for categorical variables
    cat_columns = data.select_dtypes(include=['object']).columns
    mappings = {}

    for col in cat_columns:
        data[col] = data[col].astype('category')
        mappings[col] = data[col].cat.categories.tolist()
        data[col] = data[col].cat.codes

    # Get ranges for numerical variables
    num_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    num_columns.remove('Attrition')
    ranges = {}

    for col in num_columns:
        min_val = data[col].min()
        max_val = data[col].max()
        ranges[col] = [min_val, max_val]

    # Combine mappings and ranges into the desired format
    mapping_values = {}

    # List of columns in the desired order
    column_names = ['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education',
                    'EducationField', 'Gender', 'JobLevel', 'JobRole', 'MaritalStatus',
                    'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
                    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                    'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                    'JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction',
                    'JobSatisfaction', 'WorkLifeBalance', 'AverageHoursWorked']

    for col in column_names:
        if col in mappings:
            value = mappings[col]
        elif col in ranges:
            value = ranges[col]
        else:
            value = None  # If the column doesn't exist
        mapping_values[col] = value

    # Create DataFrame with the mappings
    mapping_df = pd.DataFrame([mapping_values])
    mapping_df.to_csv('mapping_values.csv', index=False)

    # Continue with preprocessing
    X = data.drop(['Attrition'], axis=1)
    y = data['Attrition']

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler

# Create Model
def create_model(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Train Single Model
def train_single_model(i, X_train, y_train, X_test, y_test, input_dim, epochs):
    print(f"Training model {i + 1}...")
    model = create_model(input_dim)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model {i + 1}: Accuracy: {accuracy * 100:.2f}%, Loss: {loss * 100:.2f}%")
    return {'model': model, 'accuracy': accuracy, 'loss': loss, 'history': history.history}

# Train Multiple Models in Parallel
def train_multiple_models(file_path, num_models=100, epochs=50):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_path)
    input_dim = X_train.shape[1]

    # Train models in parallel
    results = Parallel(n_jobs=-1)(
        delayed(train_single_model)(i, X_train, y_train, X_test, y_test, input_dim, epochs)
        for i in range(num_models)
    )

    # Sort models by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    # Print accuracy and loss of each model
    for i, result in enumerate(results):
        print(f"Model {i + 1}: Accuracy: {result['accuracy'] * 100:.2f}%, Loss: {result['loss'] * 100:.2f}%")

    return results[:5], input_dim

# Plot Evolution
def plot_top_models(results):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    for i, result in enumerate(results):
        history = result['history']
        ax[0].plot(history['accuracy'], label=f'Model {i + 1}')
        ax[1].plot(history['val_loss'], label=f'Model {i + 1}')

    ax[0].set_title('Training Accuracy Evolution')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].set_title('Validation Loss Evolution')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('top_models_evolution.png')
    plt.show()

# Main Function
def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    file_path = '/mnt/data/final_merged_data.csv'

    # Train multiple models
    top_results, input_dim = train_multiple_models(file_path, num_models=60, epochs=50)

    # Save the best models
    os.makedirs('models/', exist_ok=True)
    for i, result in enumerate(top_results):
        accuracy = result['accuracy'] * 100
        result['model'].save(f'models/best_model_{i + 1}_{accuracy:.2f}.keras')

    # Plot the evolution of top models
    plot_top_models(top_results)

if __name__ == "__main__":
    main()
