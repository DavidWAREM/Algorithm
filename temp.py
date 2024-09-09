import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.experimental import optimizers
import numpy as np
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Aktivierungsfunktion: ReLU
def relu(x):
    return jnp.maximum(0, x)

# ANN Modell
def init_model_params(input_size, hidden_sizes, output_size, key):
    keys = jax.random.split(key, len(hidden_sizes) + 1)
    params = []

    # Eingabe -> erste verborgene Schicht
    params.append((jax.random.normal(keys[0], (input_size, hidden_sizes[0])),
                   jnp.zeros(hidden_sizes[0])))

    # Verborgene Schichten
    for i in range(1, len(hidden_sizes)):
        params.append((jax.random.normal(keys[i], (hidden_sizes[i-1], hidden_sizes[i])),
                       jnp.zeros(hidden_sizes[i])))

    # Letzte Schicht
    params.append((jax.random.normal(keys[-1], (hidden_sizes[-1], output_size)),
                   jnp.zeros(output_size)))

    return params

# Vorwärtsdurchlauf
def forward_pass(params, x):
    for w, b in params[:-1]:
        x = relu(jnp.dot(x, w) + b)  # ReLU Aktivierung
    final_w, final_b = params[-1]
    return jnp.dot(x, final_w) + final_b  # Lineare Ausgabe

# Verlustfunktion: Mittlerer quadratischer Fehler (MSE)
def loss_fn(params, x, y):
    preds = forward_pass(params, x)
    return jnp.mean((preds - y) ** 2)

# Gradientenbasierte Optimierung
@jit
def update(params, x, y, opt_state):
    grads = grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optimizers.apply_updates(params, updates)
    return new_params, opt_state

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Beispiel-Daten
    X = np.random.rand(1000, 10)  # Simulierte Feature-Daten
    y = np.random.rand(1000)      # Simulierte Ziel-Daten

    # Aufteilen des Datensatzes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Merkmale skalieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialisiere Modellparameter
    input_size = X_train_scaled.shape[1]
    hidden_sizes = [64, 32]
    output_size = 1
    key = jax.random.PRNGKey(0)
    params = init_model_params(input_size, hidden_sizes, output_size, key)

    # Initialisiere den Optimierer (SGD)
    learning_rate = 0.001
    optimizer = optimizers.sgd(learning_rate)
    opt_state = optimizer.init(params)

    # Training
    epochs = 10
    batch_size = 32
    for epoch in range(epochs):
        for i in range(0, X_train_scaled.shape[0], batch_size):
            X_batch = X_train_scaled[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            params, opt_state = update(params, X_batch, y_batch, opt_state)

        # Evaluierung des Verlusts nach jeder Epoche
        train_loss = loss_fn(params, X_train_scaled, y_train)
        logging.info(f"Epoch {epoch+1}, Loss: {train_loss}")

    # Vorhersagen auf dem Testset
    test_preds = forward_pass(params, X_test_scaled)
    test_loss = loss_fn(params, X_test_scaled, y_test)
    logging.info(f"Test Loss: {test_loss}")
