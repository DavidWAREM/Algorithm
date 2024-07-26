import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generiere synthetische Daten
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 100 Datenpunkte zwischen 0 und 10
y = 2 * X**2 + 3 * X + 5 + np.random.randn(100, 1) * 10  # Quadratische Beziehung mit etwas Rauschen

# Lineares Modell ohne polynomiale Features
model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)

# Lineares Modell mit polynomialen Features (Grad 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)

# Visualisierung der Daten und der Modellanpassungen
plt.figure(figsize=(14, 7))

# Originaldaten und lineare Anpassung
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Daten')
plt.plot(X, y_pred_linear, color='red', linewidth=2, label='Lineare Anpassung')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Lineares Modell ohne polynomiale Features')
plt.legend()

# Originaldaten und polynomiale Anpassung
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Daten')
plt.plot(X, y_pred_poly, color='red', linewidth=2, label='Polynomiale Anpassung (Grad 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Lineares Modell mit polynomialen Features')
plt.legend()

plt.tight_layout()
plt.show()
