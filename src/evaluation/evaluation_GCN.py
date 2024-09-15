import torch
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class GCNTester:
    """
        class GCNTester:

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

    def test_model(self, test_data):
        self.model.eval()
        total_mse = 0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for data, y in test_data:
                data = data.to(self.device)
                y = y.to(self.device)

                out = self.model(data)
                mse = torch.nn.functional.mse_loss(out, y)
                total_mse += mse.item()

                all_preds.extend(out.cpu().numpy())  # Vorhersagen sammeln
                all_true.extend(y.cpu().numpy())  # Wahre Werte sammeln

        # Berechne RMSE und R²-Score
        rmse = np.sqrt(mean_squared_error(all_true, all_preds))
        r2 = r2_score(all_true, all_preds)

        return total_mse, rmse, r2
