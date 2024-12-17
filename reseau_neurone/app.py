import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

# Charger les données
data = pd.read_csv("datas\\FINAL_MERGED_DATA.CSV")

# Encoder les variables catégoriques
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Sélectionner les caractéristiques et la cible
X = data.drop(columns=['Attrition'])
y = data['Attrition']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construire le modèle de réseau de neurones
mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Évaluer les performances du modèle
y_pred = mlp.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Interface utilisateur avec Streamlit
st.title("Prédiction de l'Attrition des Employés")

# Entrées utilisateur pour les variables
input_data = {}
for column in X.columns:
    if data[column].dtype == 'float64' or data[column].dtype == 'int64':
        input_data[column] = st.number_input(column, value=float(data[column].mean()))
    else:
        input_data[column] = st.selectbox(column, options=data[column].unique())

# Préparer les données pour la prédiction
input_df = pd.DataFrame([input_data])
input_df = scaler.transform(input_df)

# Faire la prédiction
prediction = mlp.predict(input_df)
prediction_proba = mlp.predict_proba(input_df)

# Afficher le résultat
if prediction[0] == 1:
    st.write("L'employé est susceptible de quitter l'entreprise.")
else:
    st.write("L'employé n'est pas susceptible de quitter l'entreprise.")

st.write(f"Probabilité de quitter l'entreprise : {prediction_proba[0][1]:.2f}")

# Vérifier la précision du modèle avec un échantillon aléatoire de 150 employés
if st.button("Vérifier la précision du modèle"):
    sample_data = data.sample(n=150, random_state=42)
    sample_X = sample_data.drop(columns=['Attrition'])
    sample_y = sample_data['Attrition']
    sample_X = scaler.transform(sample_X)
    
    sample_predictions = mlp.predict(sample_X)
    accuracy = accuracy_score(sample_y, sample_predictions)
    
    st.write(f"Précision du modèle sur un échantillon de 150 employés : {accuracy:.2f}")