import xgboost as xgb
import numpy as np

# Dummy-Daten erstellen
X = np.random.rand(1000, 10)
y = np.random.rand(1000)

# DMatrix für XGBoost erstellen
dtrain = xgb.DMatrix(X, label=y)

# Setze die Parameter für GPU-Nutzung
param = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',  # Standard tree method
    'device': 'cuda'        # Wechselt zu GPU
}

# Modell trainieren
bst = xgb.train(param, dtrain, num_boost_round=10)

print("XGBoost Modelltraining abgeschlossen. GPU wurde verwendet.")
