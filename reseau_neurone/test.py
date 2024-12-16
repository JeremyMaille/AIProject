import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Charger les données
data = pd.read_csv('datas\\FINAL_MERGED_DATA.CSV')

# Séparer les caractéristiques et la cible
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# 2. Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Construire le modèle avec plus de paramètres
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# 5. Compiler le modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Configurer le EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 7. Entraîner le modèle et stocker l'historique
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# 8. Créer l'animation
import numpy as np
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

epochs = range(1, len(history.history['accuracy']) + 1)

def animate(i):
    ax1.clear()
    ax2.clear()

    # Précision
    ax1.plot(epochs[:i], history.history['accuracy'][:i], label='Entraînement')
    ax1.plot(epochs[:i], history.history['val_accuracy'][:i], label='Validation')
    ax1.set_title('Précision du modèle')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Précision')
    ax1.legend()

    # Perte
    ax2.plot(epochs[:i], history.history['loss'][:i], label='Entraînement')
    ax2.plot(epochs[:i], history.history['val_loss'][:i], label='Validation')
    ax2.set_title('Perte du modèle')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Perte')
    ax2.legend()

ani = FuncAnimation(fig, animate, frames=len(epochs), interval=200)

# Sauvegarder l'animation en GIF
ani.save('evolution_performance.gif', writer='pillow')

plt.close()