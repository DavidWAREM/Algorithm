import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ANNModel:
    def __init__(self, input_shape, learning_rate=0.001):
        """
        Initializes the ANNModel class.

        Args:
            input_shape (int): Number of features in the input data.
            learning_rate (float): Learning rate for the optimizer.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.model = self.build_ann_model(input_shape, learning_rate)

    def build_ann_model(self, input_shape, learning_rate):
        """
        Builds and compiles the ANN model with increased complexity.

        Args:
            input_shape (int): Number of features in the input data.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: Compiled ANN model.
        """
        model = Sequential()

        # Eingabeschicht
        model.add(Dense(256, input_shape=(input_shape,)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Versteckte Schicht 1
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Versteckte Schicht 2
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Versteckte Schicht 3
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Versteckte Schicht 4
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Ausgabeschicht
        model.add(Dense(1, activation='linear'))

        # Kompilieren des Modells
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])

        self.logger.info("ANN model built and compiled successfully with increased complexity.")
        model.summary(print_fn=lambda x: self.logger.debug(x))
        return model

    def train(self, X_train, y_train, validation_split=0.2, epochs=1000, batch_size=32, patience=10):
        """
        Trains the ANN model with Early Stopping.

        Args:
            X_train (np.array): Training features.
            y_train (pd.Series): Training target.
            validation_split (float): Fraction of training data to be used as validation.
            epochs (int): Maximum number of epochs for training.
            batch_size (int): Number of samples per gradient update.
            patience (int): Number of epochs with no improvement after which training will be stopped.

        Returns:
            tf.keras.callbacks.History: Training history.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        self.logger.info("Model training completed.")
        return history

    def save_model(self, file_name="ann_model.h5"):
        """
        Saves the trained model to disk.

        Args:
            file_name (str): Name of the file to save the model.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'models')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, file_name)
        self.model.save(file_path)
        self.logger.info(f"Model saved to {file_path}")

    @staticmethod
    def load_trained_model(file_name="ann_model.h5"):
        """
        Loads a trained ANN model from disk.

        Args:
            file_name (str): Name of the file to load the model from.

        Returns:
            tf.keras.Model: Loaded ANN model.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'models')
        file_path = os.path.join(results_dir, file_name)
        model = load_model(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model
