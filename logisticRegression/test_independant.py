import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Charger les données et le modèle sauvegardé
best_model = joblib.load("graphique/best_model.pkl")
X_balanced, y_balanced = joblib.load("graphique/data_balanced.pkl")

# Générer de nouvelles données indépendantes
X_independent, y_independent = make_classification(
    n_samples=1000,  # Nombre de nouvelles observations
    n_features=X_balanced.shape[1],  # Même nombre de features
    n_informative=10,  # Nombre de features informatives
    n_redundant=5,  # Nombre de features redondantes
    n_classes=2,  # Deux classes : 0 et 1
    weights=[0.7, 0.3],  # Classement déséquilibré similaire
    random_state=42
)

# Prédictions sur l'ensemble de test indépendant
y_independent_pred = best_model.predict(X_independent)
y_independent_pred_prob = best_model.predict_proba(X_independent)[:, 1]

# Métriques sur l'ensemble indépendant
accuracy = accuracy_score(y_independent, y_independent_pred)
recall = recall_score(y_independent, y_independent_pred)
f1 = f1_score(y_independent, y_independent_pred)
conf_matrix = confusion_matrix(y_independent, y_independent_pred)
report = classification_report(y_independent, y_independent_pred)

# Résultats
print("\nÉvaluation sur l'ensemble de test indépendant :")
print(f"Accuracy : {accuracy:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("Confusion Matrix :\n", conf_matrix)
print("Classification Report :\n", report)

# Courbe ROC sur l'ensemble indépendant
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_independent, y_independent_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Independent Test')
plt.legend(loc="lower right")
plt.grid()
plt.savefig("graphique/independent_test_roc_curve.png")
plt.show()

# Sauvegarder le rapport
with open("graphique/independent_test_report.txt", "w") as f:
    f.write("Évaluation sur l'ensemble de test indépendant:\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Recall : {recall:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# Distribution des scores sur l'ensemble indépendant
plt.figure(figsize=(10, 6))
plt.hist(y_independent_pred_prob, bins=20, color='blue', alpha=0.7, label='Predicted Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Distribution des probabilités prédites - Ensemble de test indépendant')
plt.grid()
plt.legend()
plt.savefig("graphique/independent_test_prob_distribution.png")
plt.show()
