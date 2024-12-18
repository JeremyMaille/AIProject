# ------------------------------------------
# IMPORTING REQUIRED LIBRARIES
# ------------------------------------------
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

# ------------------------------------------
# DATA LOADING AND PREPARATION
# ------------------------------------------
merged_data = pd.read_csv("../datas/final_merged_data_with_work_metrics_delete.csv")
y = merged_data["Attrition"]
X = merged_data.drop(columns=["Attrition", "EmployeeID"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
total_test_size = 1446  
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=total_test_size / len(X_scaled), random_state=42
)

# ------------------------------------------
# MODEL INITIALIZATION AND TRAINING
# ------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------
# MODEL PREDICTIONS AND EVALUATION
# ------------------------------------------
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)

# ------------------------------------------
# ROC CURVE AND AUC SCORE
# ------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

print(f"AUC of the ROC Curve: {roc_auc:.4f}")
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("graphique/roc_curve.png")
plt.show()

# ------------------------------------------
# CROSS-VALIDATION RESULTS
# ------------------------------------------
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print("Cross-Validation Results:")
print(f"Scores: {cv_scores}")
print(f"Mean Score: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}")

# ------------------------------------------
# PRECISION, RECALL, AND F1-SCORE
# ------------------------------------------
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# ------------------------------------------
# LOG LOSS FOR TRAINING AND TEST DATA
# ------------------------------------------
loss_train = log_loss(y_train, model.predict_proba(X_train)[:, 1])
loss_test = log_loss(y_test, y_pred_prob)
print(f"Log Loss - Training: {loss_train:.4f}")
print(f"Log Loss - Test    : {loss_test:.4f}")

# ------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# ------------------------------------------
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix")
plt.savefig("graphique/confusion_matrix.png")
plt.show()

# ------------------------------------------
# TRAINING AND TEST ACCURACY CURVES
# ------------------------------------------
train_accuracies = []
test_accuracies = []
loss_train_values = []
loss_test_values = []

epochs = range(0, 50, 2)  
for epoch in epochs:
    model = LogisticRegression(max_iter=epoch, random_state=42)
    model.fit(X_train, y_train)

    y_partial_pred = model.predict(X_train)
    y_partial_pred_prob = model.predict_proba(X_train)[:, 1]

    train_accuracies.append(accuracy_score(y_train, y_partial_pred))
    loss_train_values.append(log_loss(y_train, y_partial_pred_prob))

    y_test_pred = model.predict(X_test)
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]

    test_accuracies.append(accuracy_score(y_test, y_test_pred))
    loss_test_values.append(log_loss(y_test, y_test_pred_prob))

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_train_values, label="Train Loss", marker="o")
plt.plot(epochs, loss_test_values, label="Test Loss", marker="o")
plt.title("Training and Test Log Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig("graphique/log_loss_curve_epochs.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(epochs, test_accuracies, label="Test Accuracy", marker="o")
plt.title("Training and Test Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig("graphique/accuracy_curve_epochs.png")
plt.show()

# ------------------------------------------
# TOP 10 MOST INFLUENTIAL FEATURES
# ------------------------------------------
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': abs(model.coef_[0])
}).sort_values(by='Coefficient', ascending=False, key=abs)

top_10_features = coefficients.head(10)

plt.figure(figsize=(12, 8))
plt.barh(top_10_features['Feature'], top_10_features['Coefficient'], color='blue')
plt.title("Top 10 Influential Features for Attrition")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.savefig("graphique/top_10_influential_features.png")
plt.show()

# ------------------------------------------
# PREDICTIONS FOR THE FIRST 10 PEOPLE
# ------------------------------------------
top_10_people = X[:10]
top_10_people_scaled = scaler.transform(top_10_people)

predictions = model.predict(top_10_people_scaled)
probabilities = model.predict_proba(top_10_people_scaled)[:, 1]

print("Predictions for the First 10 People:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    prob = (prob - 0.5) * 2
    print(f"Person {i+1}: Will Leave the Company: {'Yes' if pred == 1 else 'No'}, Probability: {prob:.2f}")
