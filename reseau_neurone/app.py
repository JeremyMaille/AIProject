# ------------------------------------------
# IMPORTING REQUIRED LIBRARIES
# ------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

# ------------------------------------------
# LOADING AND ENCODING DATA
# ------------------------------------------
data = pd.read_csv("datas\\FINAL_MERGED_DATA.CSV")

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# ------------------------------------------
# FEATURE SELECTION AND DATA SPLIT
# ------------------------------------------
X = data.drop(columns=['Attrition'])
y = data['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------
# DATA STANDARDIZATION
# ------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------
# BUILDING AND TRAINING THE MODEL
# ------------------------------------------
mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# ------------------------------------------
# MODEL EVALUATION
# ------------------------------------------
y_pred = mlp.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------
# STREAMLIT USER INTERFACE
# ------------------------------------------
st.title("Prédiction de l'Attrition des Employés")

# ------------------------------------------
# USER INPUTS
# ------------------------------------------
input_data = {}
for column in X.columns:
    if data[column].dtype == 'float64' or data[column].dtype == 'int64':
        input_data[column] = st.number_input(column, value=float(data[column].mean()))
    else:
        input_data[column] = st.selectbox(column, options=data[column].unique())

# ------------------------------------------
# PREDICTION PREPARATION
# ------------------------------------------
input_df = pd.DataFrame([input_data])
input_df = scaler.transform(input_df)

prediction = mlp.predict(input_df)
prediction_proba = mlp.predict_proba(input_df)

# ------------------------------------------
# DISPLAYING RESULTS
# ------------------------------------------
if prediction[0] == 1:
    st.write("L'employé est susceptible de quitter l'entreprise.")
else:
    st.write("L'employé n'est pas susceptible de quitter l'entreprise.")

st.write(f"Probabilité de quitter l'entreprise : {prediction_proba[0][1]:.2f}")

# ------------------------------------------
# MODEL ACCURACY ON RANDOM SAMPLE
# ------------------------------------------
if st.button("Vérifier la précision du modèle"):
    sample_data = data.sample(n=150, random_state=42)
    sample_X = sample_data.drop(columns=['Attrition'])
    sample_y = sample_data['Attrition']
    sample_X = scaler.transform(sample_X)
    
    sample_predictions = mlp.predict(sample_X)
    accuracy = accuracy_score(sample_y, sample_predictions)
    
    st.write(f"Précision du modèle sur un échantillon de 150 employés : {accuracy:.2f}")
