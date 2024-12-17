import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import log_loss, accuracy_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

merged_data = pd.read_csv("final_merged_data_with_work_metrics_delete.csv")

y = merged_data["Attrition"]
X = merged_data.drop(columns=["Attrition", "EmployeeID"])

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)
X = pd.DataFrame(X_poly, columns=feature_names)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

scaler = StandardScaler()
X_balanced = scaler.fit_transform(X_balanced)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=1000, random_state=42), param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Meilleurs hyperparamètres :", grid_search.best_params_)

joblib.dump(best_model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(poly, "poly_features.pkl")

loss_train = []
loss_test = []
accuracy_train = []
accuracy_test = []

for i in range(1, 11):
    best_model.fit(X_train[:i * len(X_train) // 10], y_train[:i * len(y_train) // 10])
    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]
    loss_train.append(log_loss(y_train, y_train_pred_prob))
    loss_test.append(log_loss(y_test, y_test_pred_prob))
    accuracy_train.append(accuracy_score(y_train, best_model.predict(X_train)))
    accuracy_test.append(accuracy_score(y_test, best_model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), loss_train, label="Train Loss", marker="o")
plt.plot(range(1, 11), loss_test, label="Test Loss", marker="o")
plt.title("Courbe de la perte (Loss) pour l'entraînement et le test")
plt.xlabel("Fraction de données utilisées pour l'entraînement")
plt.ylabel("Log Loss")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig("graphique/loss_train_test_curve.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), accuracy_train, label="Train Accuracy", marker="o")
plt.plot(range(1, 11), accuracy_test, label="Test Accuracy", marker="o")
plt.title("Courbe de l'accuracy pour l'entraînement et le test")
plt.xlabel("Fraction de données utilisées pour l'entraînement")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig("graphique/accuracy_train_test_curve.png")
plt.show()

coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': best_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(15, 10))
plt.barh(coefficients['Feature'].head(10), coefficients['Coefficient'].head(10))
plt.title("Top 10 des paramètres les plus influents sur l'attrition")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.grid()
plt.savefig("graphique/top_10_influential_features.png")
plt.show()

cv_scores = cross_val_score(best_model, X_balanced, y_balanced, cv=5, scoring='accuracy')
print("Résultats de la validation croisée :")
print(f"Scores : {cv_scores}")
print(f"Moyenne des scores : {np.mean(cv_scores):.4f}")
print(f"Écart-type des scores : {np.std(cv_scores):.4f}")

y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_test_pred = best_model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("Courbe ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("graphique/roc_curve.png")
plt.show()

conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.8, fignum=1)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("graphique/confusion_matrix.png")
plt.show()

precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

top_10_people = merged_data.drop(columns=["Attrition", "EmployeeID"]).iloc[:10]
top_10_people_poly = poly.transform(top_10_people)
top_10_people_scaled = scaler.transform(top_10_people_poly)

predictions = best_model.predict(top_10_people_scaled)
probabilities = best_model.predict_proba(top_10_people_scaled)[:, 1]

print("Prédictions pour les 10 premières personnes :")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Personne {i+1}: Quittera l'entreprise : {'Oui' if pred == 1 else 'Non'}, Probabilité : {prob:.2f}")
