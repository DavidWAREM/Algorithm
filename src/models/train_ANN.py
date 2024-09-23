import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ANNModel:
    def __init__(self, input_shape, learning_rate=0.001):
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        self.model = self.build_ann_model(input_shape, learning_rate)

    def build_ann_model(self, input_shape, learning_rate):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        self.logger.info("ANN model built and compiled successfully.")
        return model

    def train(self, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, patience=10):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs,
                                 batch_size=batch_size, callbacks=[early_stopping], verbose=1)
        self.logger.info("Model training completed.")
        return history

    def save_model(self, file_name="ann_model.h5"):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'models')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, file_name)
        self.model.save(file_path)
        self.logger.info(f"Model saved to {file_path}")

    @staticmethod
    def load_model(file_name="ann_model.h5"):
        # Initialize logger inside the static method
        logger = logging.getLogger(__name__)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'models')
        file_path = os.path.join(results_dir, file_name)
        model = load_model(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model
