import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
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

    bool_columns = data.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        data[col] = data[col].astype(int)

    cat_columns = data.select_dtypes(include=['object']).columns
    if len(cat_columns) > 0:
        data = pd.get_dummies(data, columns=cat_columns)

    X = data.drop(['Attrition'], axis=1)
    y = data['Attrition']

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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

# Train Multiple Models
def train_multiple_models(file_path, num_models=100, epochs=50):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_path)
    
    results = []
    models = []

    for i in range(num_models):
        print(f"Training model {i + 1}/{num_models}...")
        model = create_model(X_train.shape[1])

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
        results.append({'model': model, 'accuracy': accuracy, 'history': history.history})
        models.append(model)

    # Sort models by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    return results[:5], X_train.shape[1]

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
    top_results, input_dim = train_multiple_models(file_path, num_models=100, epochs=50)

    # Save the best models
    for i, result in enumerate(top_results):
        result['model'].save(f'best_model_{i + 1}.h5')

    # Plot the evolution of top models
    plot_top_models(top_results)

if __name__ == "__main__":
    main()
