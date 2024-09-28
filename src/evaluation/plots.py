# src/evaluation/plots.py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns
import logging

def plot_roc_curve(y_true, y_pred):
    """
    Plots the ROC curve.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted probabilities.
    """
    logger = logging.getLogger(__name__)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend()
        plt.savefig('results/figures/roc_curve.png')
        plt.close()
        logger.info("ROC curve plotted and saved.")
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")

def plot_precision_recall(y_true, y_pred):
    """
    Plots the Precision-Recall curve.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted probabilities.
    """
    logger = logging.getLogger(__name__)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.figure()
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig('results/figures/precision_recall_curve.png')
        plt.close()
        logger.info("Precision-Recall curve plotted and saved.")
    except Exception as e:
        logger.error(f"Error plotting Precision-Recall curve: {e}")

def plot_confusion_matrix(y_true, y_pred_binary):
    """
    Plots the confusion matrix.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred_binary (np.ndarray): Predicted binary labels.
    """
    logger = logging.getLogger(__name__)
    try:
        cm = confusion_matrix(y_true, y_pred_binary)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig('results/figures/confusion_matrix.png')
        plt.close()
        logger.info("Confusion matrix plotted and saved.")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
