# ------------------------------------------
# IMPORTING REQUIRED LIBRARIES
# ------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import log_loss, accuracy_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# ------------------------------------------
# PREPARING OUTPUT DIRECTORY
# ------------------------------------------
output_dir = "graphique"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ------------------------------------------
# LOADING DATA
# ------------------------------------------
merged_data = pd.read_csv("../datas/final_merged_data_with_work_metrics_delete.csv")

y = merged_data["Attrition"]
X = merged_data.drop(columns=["Attrition", "EmployeeID"])

# ------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)
X = pd.DataFrame(X_poly, columns=feature_names)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

scaler = StandardScaler()
X_balanced = scaler.fit_transform(X_balanced)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# ------------------------------------------
# RANDOMIZED SEARCH FOR HYPERPARAMETERS
# ------------------------------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

# ------------------------------------------
# SAVING BEST MODEL AND TRANSFORMERS
# ------------------------------------------
best_model = random_search.best_estimator_
print("Best Hyperparameters :", random_search.best_params_)

joblib.dump(best_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(poly, "poly_features.pkl")

# ------------------------------------------
# EVALUATION METRICS
# ------------------------------------------
loss_train = []
loss_test = []
accuracy_train = []
accuracy_test = []

epochs = np.arange(1, 51, 5)
for epoch in epochs:
    if epoch <= 0:
        continue
    model = RandomForestClassifier(n_estimators=epoch, random_state=42, **{k: v for k, v in random_search.best_params_.items() if k != 'n_estimators'})
    model.fit(X_train, y_train)
    
    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]
    
    loss_train.append(log_loss(y_train, y_train_pred_prob))
    loss_test.append(log_loss(y_test, y_test_pred_prob))
    
    accuracy_train.append(accuracy_score(y_train, model.predict(X_train)))
    accuracy_test.append(accuracy_score(y_test, model.predict(X_test)))

min_len = min(len(epochs), len(loss_train))
epochs = epochs[:min_len]
loss_train = loss_train[:min_len]
loss_test = loss_test[:min_len]
accuracy_train = accuracy_train[:min_len]
accuracy_test = accuracy_test[:min_len]

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_train, label="Train Loss", marker="o")
plt.plot(epochs, loss_test, label="Test Loss", marker="o")
plt.title("Loss Curve for Training and Test")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/loss_curve_epochs.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy_train, label="Train Accuracy", marker="o")
plt.plot(epochs, accuracy_test, label="Test Accuracy", marker="o")
plt.title("Accuracy Curve for Training and Test")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/accuracy_curve_epochs.png")
plt.show()

# ------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------
importances = best_model.feature_importances_
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
coefficients.head(10).plot(kind='bar', x='Feature', y='Importance', legend=False, color='skyblue')
plt.title("Top 10 Most Influential Parameters on Attrition")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{output_dir}/top_10_influential_features.png")
plt.show()

# ------------------------------------------
# CROSS-VALIDATION SCORES
# ------------------------------------------
cv_scores = cross_val_score(best_model, X_balanced, y_balanced, cv=3, scoring='accuracy')
print("Cross-validation results :")
print(f"Scores : {cv_scores}")
print(f"Average score : {np.mean(cv_scores):.4f}")
print(f"Standard deviation of scores : {np.std(cv_scores):.4f}")

# ------------------------------------------
# ROC CURVE
# ------------------------------------------
y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_test_pred = best_model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(f"{output_dir}/roc_curve.png")
plt.show()

# ------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.8, fignum=1)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.show()

# ------------------------------------------
# CLASSIFICATION METRICS
# ------------------------------------------
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# ------------------------------------------
# PREDICTIONS FOR TOP 10 PEOPLE
# ------------------------------------------
top_10_people = merged_data.drop(columns=["Attrition", "EmployeeID"]).iloc[:10]
top_10_people_poly = poly.transform(top_10_people)
top_10_people_scaled = scaler.transform(top_10_people_poly)

predictions = best_model.predict(top_10_people_scaled)
probabilities = best_model.predict_proba(top_10_people_scaled)[:, 1]

print("Predictions for the first 10 people :")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Person {i+1}: Will leave the company : {'Oui' if pred == 1 else 'Non'}, Probability : {prob:.2f}")
