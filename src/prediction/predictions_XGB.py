import xgboost as xgb
import numpy as np

class XGBoostPrediction:
    def __init__(self, model_file="xgboost_model.model"):
        # Lade das gespeicherte XGBoost-Modell
        self.model = xgb.Booster()
        self.model.load_model(model_file)

    def predict(self, X):
        # Erstelle DMatrix für Vorhersagen
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)

if __name__ == "__main__":
    # Testdaten für Vorhersagen
    X_test = np.random.rand(100, 20)

    # Mache Vorhersagen mit XGBoost
    predictor = XGBoostPrediction()
    predictions = predictor.predict(X_test)

    print("Predictions:", predictions)
