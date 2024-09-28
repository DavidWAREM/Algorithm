import numpy as np
import logging
from sklearn.metrics import roc_auc_score, accuracy_score
from src.evaluation.plots import plot_roc_curve, plot_precision_recall, plot_confusion_matrix

from src.train.train import test

def evaluate_model(test_loader, model, device, logger):
    y_true, y_pred = test(test_loader, model, device)

    # Überprüfen, ob y_true mindestens zwei Klassen enthält
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning("y_true enthält nur eine Klasse. ROC AUC Score kann nicht berechnet werden.")
        auc_score = None
    else:
        # Metriken berechnen
        auc_score = roc_auc_score(y_true, y_pred)

    # Binäre Vorhersagen
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)

    logger.info(f'Genauigkeit: {accuracy:.4f}')
    if auc_score is not None:
        logger.info(f'AUC: {auc_score:.4f}')

    # Plots erzeugen
    plot_roc_curve(y_true, y_pred)
    plot_precision_recall(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred_binary)

    return auc_score, accuracy, model
