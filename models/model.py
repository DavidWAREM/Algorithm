import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Funktion zum Laden der Daten aus einer Excel-Datei
def load_data_from_excel(file_path):
    data = pd.read_excel(file_path)
    return data

# Pfad zum Ordner mit den Excel-Dateien
folder_path = r"C:\Users\d.muehlfeld\PycharmProjects\temp\generated_excel_files"

# Liste zum Speichern der Daten aus allen Excel-Dateien
all_data = []

# Durchlaufen aller Excel-Dateien im Ordner
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        data = load_data_from_excel(file_path)
        if 'A' not in data.columns or 'B' not in data.columns:
            print(f"Columns 'A' or 'B' not found in file: {file_name}")
            continue
        all_data.append((file_name, data))

# Sicherstellen, dass es mindestens 10 Dateien gibt
if len(all_data) < 10:
    raise ValueError("Not enough files. Ensure there are at least 10 Excel files in the directory.")

# Aufteilen der Daten in Trainings- und Testdateien
np.random.seed(42)
np.random.shuffle(all_data)
train_files = all_data[:8]
test_files = all_data[8:]

# StandardScaler initialisieren
scaler = StandardScaler()

# Training mit 8 Dateien
X_train_list = []
y_train_list = []
for file_name, data in train_files:
    X_train = data[['A']].values  # Unabhängige Variable
    y_train = data['B'].values    # Abhängige Variable
    X_train_list.append(X_train)
    y_train_list.append(y_train)

# Kombinieren der Trainingsdaten und Skalieren
X_train_combined = np.vstack(X_train_list)
y_train_combined = np.concatenate(y_train_list)
X_train_scaled = scaler.fit_transform(X_train_combined)

# Erstellen des linearen Regressionsmodells für inkrementelles Lernen
model = SGDRegressor(max_iter=10000, tol=1e-4, learning_rate='adaptive', eta0=0.01)

# Trainieren des Modells
model.fit(X_train_scaled, y_train_combined)

# Sammeln der Vorhersagen und der tatsächlichen Werte für alle Testdateien
all_y_test = []
all_y_pred = []

# Evaluieren des Modells mit den verbleibenden 2 Dateien
for file_name, data in test_files:
    X_test = data[['A']].values  # Unabhängige Variable
    y_test = data['B'].values    # Abhängige Variable
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

    # Ausgeben der Koeffizienten und Leistungsmetriken für die Testdatei
    print(f"Results for file: {file_name}")

    # Berechnung des nicht skalierten Intercepts und Koeffizienten
    unscaled_intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
    unscaled_coefficients = model.coef_ / scaler.scale_

    print(f"Scaled Intercept: {model.intercept_}")
    print(f"Scaled Coefficients: {model.coef_}")
    print(f"Unscaled Intercept: {unscaled_intercept}")
    print(f"Unscaled Coefficients: {unscaled_coefficients}")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")

# Plotten der gesammelten Ergebnisse
plt.figure()
plt.scatter(all_y_test, all_y_pred, color='black', label='Actual vs Predicted')
plt.plot([min(all_y_test), max(all_y_test)], [min(all_y_test), max(all_y_test)], color='blue', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual B')
plt.ylabel('Predicted B')
plt.title('Combined Results for Validation Data')
plt.legend()
plt.show()
