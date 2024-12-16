import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import os
import joblib

# Charger le modèle et les données
best_model = joblib.load("graphique/best_model.pkl")
X_balanced, y_balanced = joblib.load("graphique/data_balanced.pkl")
print("Modèle et données chargés.")


# Créer un dossier pour enregistrer les graphiques
os.makedirs("graphique", exist_ok=True)

# Validation croisée avec scoring multiple
cv_results = cross_validate(
    estimator=best_model,
    X=X_balanced,
    y=y_balanced,
    cv=5,  # Nombre de folds
    scoring=['accuracy', 'f1', 'recall', 'precision'],  # Métriques à évaluer
    return_train_score=True
)

# Résumé des résultats
print("Validation croisée (5 folds) :")
print(f"Accuracy moyenne : {np.mean(cv_results['test_accuracy']):.4f} ± {np.std(cv_results['test_accuracy']):.4f}")
print(f"F1-Score moyen : {np.mean(cv_results['test_f1']):.4f} ± {np.std(cv_results['test_f1']):.4f}")
print(f"Recall moyen : {np.mean(cv_results['test_recall']):.4f} ± {np.std(cv_results['test_recall']):.4f}")
print(f"Precision moyenne : {np.mean(cv_results['test_precision']):.4f} ± {np.std(cv_results['test_precision']):.4f}")

# Distribution des scores sur les folds
plt.figure(figsize=(10, 6))
plt.plot(cv_results['test_accuracy'], label='Accuracy', marker='o')
plt.plot(cv_results['test_f1'], label='F1-Score', marker='o')
plt.plot(cv_results['test_recall'], label='Recall', marker='o')
plt.plot(cv_results['test_precision'], label='Precision', marker='o')
plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Validation croisée - Scores par Fold")
plt.legend()
plt.grid()
plt.savefig("graphique/cross_validation_scores.png")
plt.show()
