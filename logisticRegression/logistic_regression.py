import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_curve, auc, log_loss, confusion_matrix, 
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)

# Chargement des données
merged_data = pd.read_csv("final_merged_data_with_work_metrics_delete.csv")

# Préparation des données
y = merged_data["Attrition"]
X = merged_data.drop(columns=["Attrition", "EmployeeID"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèle de régression logistique
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Prédictions et probabilités
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Métriques de base
accuracy = accuracy_score(y_test, y_pred)

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

print(f"AUC de la courbe ROC : {roc_auc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("Courbe ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("graphique/roc_curve.png")
plt.show()

# Validation croisée
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print("Résultats de la validation croisée :")
print(f"Scores : {cv_scores}")
print(f"Moyenne des scores : {np.mean(cv_scores):.4f}")
print(f"Écart-type des scores : {np.std(cv_scores):.4f}")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# Log Loss
loss_train = log_loss(y_train, model.predict_proba(X_train)[:, 1])
loss_test = log_loss(y_test, y_pred_prob)

print(f"Log Loss - Entraînement : {loss_train:.4f}")
print(f"Log Loss - Test         : {loss_test:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix")
plt.savefig("graphique/confusion_matrix.png")
plt.show()

# Courbes de Log Loss et Accuracy
train_accuracies = []
test_accuracies = []
loss_train_values = []
loss_test_values = []

for fraction in np.linspace(0.1, 1.0, 10):
    X_partial = X_train[:int(fraction * len(X_train))]
    y_partial = y_train[:int(fraction * len(y_train))]

    model.fit(X_partial, y_partial)

    y_partial_pred = model.predict(X_partial)
    y_partial_pred_prob = model.predict_proba(X_partial)[:, 1]

    train_accuracies.append(accuracy_score(y_partial, y_partial_pred))
    loss_train_values.append(log_loss(y_partial, y_partial_pred_prob))

    y_test_pred = model.predict(X_test)
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]

    test_accuracies.append(accuracy_score(y_test, y_test_pred))
    loss_test_values.append(log_loss(y_test, y_test_pred_prob))

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0.1, 1.0, 10), loss_train_values, label="Train Loss", marker="o")
plt.plot(np.linspace(0.1, 1.0, 10), loss_test_values, label="Test Loss", marker="o")
plt.title("Log Loss pour l'entraînement et le test")
plt.xlabel("Fraction des données d'entraînement utilisées")
plt.ylabel("Log Loss")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig("graphique/log_loss_curve.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0.1, 1.0, 10), train_accuracies, label="Train Accuracy", marker="o")
plt.plot(np.linspace(0.1, 1.0, 10), test_accuracies, label="Test Accuracy", marker="o")
plt.title("Accuracy pour l'entraînement et le test")
plt.xlabel("Fraction des données d'entraînement utilisées")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig("graphique/accuracy_curve.png")
plt.show()

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': abs(model.coef_[0])
}).sort_values(by='Coefficient', ascending=False, key=abs)

top_10_features = coefficients.head(10)

plt.figure(figsize=(12, 8))
plt.barh(top_10_features['Feature'], top_10_features['Coefficient'], color='blue')
plt.title("Top 10 des paramètres les plus influents sur l'attrition")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.gca().invert_yaxis()  
plt.grid(axis='x')
plt.savefig("graphique/top_10_influential_features.png")
plt.show()

top_10_people = X[:10]
top_10_people_scaled = scaler.transform(top_10_people)

predictions = model.predict(top_10_people_scaled)
probabilities = model.predict_proba(top_10_people_scaled)[:, 1]

print("Prédictions pour les 10 premières personnes :")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Personne {i+1}: Quittera l'entreprise : {'Oui' if pred == 1 else 'Non'}, Probabilité : {prob:.2f}")
