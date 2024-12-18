from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def create_model(self, input_dim):
        model = Sequential([
            Dense(self.config.layer_sizes[0], activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(self.config.dropout_rates[0]),
            Dense(self.config.layer_sizes[1], activation='relu'),
            BatchNormalization(),
            Dropout(self.config.dropout_rates[1]),
            Dense(self.config.layer_sizes[2], activation='relu'),
            BatchNormalization(),
            Dropout(self.config.dropout_rates[2]),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        return model

    def train_single_model(self, X_train, y_train, X_test, y_test, epochs=50):
        model = self.create_model(X_train.shape[1])

        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(y_train),
                                             y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))

        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max')
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=0  # Silent training to reduce console clutter
        )

        return model, history