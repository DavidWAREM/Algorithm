import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class XGBoostModel:
    def __init__(self):
        # Verwende 'device' statt 'gpu_hist'
        self.params = {
            'objective': 'reg:squarederror',
            'device': 'cuda',  # Verwende GPU für das Training
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'max_depth': 7,
            'n_estimators': 600,
            'subsample': 0.9
        }
        self.model = xgb.XGBRegressor(**self.params)

    def hyperparameter_tuning(self, X_train, y_train):
        # Definiere das Hyperparameter-Raster für das Tuning
        param_grid = {
            'max_depth': [5, 6, 7],  # Feinere Abstufung in der Baumtiefe
            'learning_rate': [0.05, 0.1, 0.15],  # Kleinere Lernraten für stabilere Updates
            'n_estimators': [400, 500, 600],  # Mehr Bäume für bessere Genauigkeit
            'subsample': [0.7, 0.8, 0.9],  # Leicht reduziertes Subsampling zur Vermeidung von Overfitting
            'colsample_bytree': [0.7, 0.8, 0.9]  # Mehr Varianz bei den ausgewählten Features pro Baum
        }

        # Verwende GridSearchCV für das Tuning
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_train, y_train)

        # Speichere das beste Modell und die besten Hyperparameter
        self.model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

    def train(self, X_train, y_train):
        # Training des Modells
        self.model.fit(X_train, y_train)

    def save_model(self, file_name="xgboost_model.model"):
        # Speichere das trainierte Modell
        self.model.save_model(file_name)

    def load_model(self, file_name="xgboost_model.model"):
        # Lade das trainierte Modell
        self.model = xgb.Booster()
        self.model.load_model(file_name)

    def evaluate(self, X_test, y_test):
        # Vorhersagen machen
        y_pred = self.model.predict(X_test)

        # Metriken berechnen
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
