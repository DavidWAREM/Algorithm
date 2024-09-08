import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

class XGBoostModelEvaluator:
    @staticmethod
    def evaluate_and_visualize(X_test, y_test, model_file="xgboost_model.model"):
        # Lade das gespeicherte XGBoost-Modell
        model = xgb.Booster()
        model.load_model(model_file)

        # Erstelle DMatrix für Testdaten
        dtest = xgb.DMatrix(X_test)

        # Mache Vorhersagen
        y_pred = model.predict(dtest)

        # Berechne Metriken
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        # Plot True vs Predicted
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('XGBoost - True vs Predicted')
        plt.show()

if __name__ == "__main__":
    # Erstelle Dummy-Daten
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    # Datensatz aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluierung des Modells
    XGBoostModelEvaluator.evaluate_and_visualize(X_test, y_test)
