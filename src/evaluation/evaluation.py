# src/evaluation/evaluation.py
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from src.evaluation.plots import plot_roc_curve, plot_precision_recall, plot_confusion_matrix
from src.train.train import test
import logging

class Evaluator:
    """
    Evaluator class handles model evaluation and metric calculations.
    """
    def __init__(self, model, device, logger):
        """
        Initializes the Evaluator.

        Args:
            model (torch.nn.Module): The trained model.
            device (torch.device): Device to run the evaluation on.
            logger (logging.Logger): Logger instance.
        """
        self.model = model
        self.device = device
        self.logger = logger

    def evaluate(self, test_loader):
        """
        Evaluates the model on the test data.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            tuple: AUC score, accuracy score.
        """
        try:
            y_true, y_pred = test(test_loader, self.model, self.device)
            self.logger.info("Test predictions obtained.")

            # Calculate ROC AUC
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                self.logger.warning("y_true contains only one class. ROC AUC cannot be computed.")
                auc_score = None
            else:
                auc_score = roc_auc_score(y_true, y_pred)
                self.logger.info(f"ROC AUC Score: {auc_score:.4f}")

            # Calculate Accuracy
            y_pred_binary = (y_pred >= 0.5).astype(int)
            accuracy = accuracy_score(y_true, y_pred_binary)
            self.logger.info(f"Accuracy Score: {accuracy:.4f}")

            # Generate plots
            plot_roc_curve(y_true, y_pred)
            plot_precision_recall(y_true, y_pred)
            plot_confusion_matrix(y_true, y_pred_binary)
            self.logger.info("Evaluation plots generated.")

            return auc_score, accuracy

        except Exception as e:
            self.logger.error(f"An error occurred during evaluation: {e}")
            raise
