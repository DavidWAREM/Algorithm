import torch
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt  # Importiere Matplotlib für die Visualisierung
import logging


class GCNTester:
    """
        A class used to test a Graph Convolutional Network (GCN) model.

        Attributes
        ----------
        model : torch.nn.Module
            The GCN model to be tested.
        device : torch.device
            The device on which the model and data are located (e.g., CPU or GPU).

        Methods
        -------
        __init__(self, model, device)
            Initializes the GCNTester with a model and a device.

        test_model(self, test_data)
            Tests the model using the provided test data and returns various performance metrics.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        logging.info("GCNTester initialized with the provided model and device.")

    def test_model(self, test_data):
        logging.info("Starting model testing...")
        self.model.eval()
        total_mse = 0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for idx, (data, y) in enumerate(test_data):
                data = data.to(self.device)
                y = y.to(self.device)

                logging.debug(f"Testing dataset {idx + 1} on the model.")
                out = self.model(data)

                mse = torch.nn.functional.mse_loss(out, y)
                total_mse += mse.item()
                logging.debug(f"MSE for dataset {idx + 1}: {mse.item()}")

                all_preds.extend(out.cpu().numpy())  # Vorhersagen sammeln
                all_true.extend(y.cpu().numpy())  # Wahre Werte sammeln

        # Berechne RMSE und R²-Score
        rmse = np.sqrt(mean_squared_error(all_true, all_preds))
        r2 = r2_score(all_true, all_preds)
        logging.info(f"Testing completed. Total MSE: {total_mse}, RMSE: {rmse}, R²: {r2}")

        # Visualisierung: True vs Predicted Werte
        self.plot_predictions(np.array(all_true), np.array(all_preds))

        return total_mse, rmse, r2

    def plot_predictions(self, y_true, y_pred):
        """
        Plots the true values vs the predicted values for visual inspection.

        Parameters:
        y_true (array): True target values.
        y_pred (array): Predicted values from the model.
        """
        logging.info("Plotting true vs predicted values.")
        plt.figure(figsize=(10, 5))
        plt.scatter(y_true, y_pred, alpha=0.7)  # Scatter plot der wahren vs vorhergesagten Werte
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r',
                 linewidth=2)  # Linie für perfekte Vorhersage
        plt.xlabel('True Values')  # X-Achse Label
        plt.ylabel('Predicted Values')  # Y-Achse Label
        plt.title('GCN - True vs Predicted')  # Titel des Plots
        plt.show()  # Zeigt den Plot an
        logging.info("Plot displayed successfully.")
