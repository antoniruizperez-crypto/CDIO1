import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# 1. CARGA DE DATOS HISTÓRICOS Y GROUND TRUTH 2025
# Cargar la serie procesada previamente (o recomputarla rápido)
CSV_PATH = "projects/costa_principal/output/estimated_erosion_accretion/shoreline_distances.csv"
df = pd.read_csv(CSV_PATH, parse_dates=["date"])
pivot_df = df.pivot(index="date", columns="transect_id", values="distance_m")
ts = pivot_df.mean(axis=1).resample("2W").mean().interpolate(method="linear")

# 2. PREPARACIÓN PARA LSTM (Sliding Window & Normalización)
scaler = MinMaxScaler(feature_range=(0, 1))
ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

WINDOW_SIZE = 12  # Usar las últimas 12 observaciones (~6 meses) para predecir la siguiente

def create_sliding_windows(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

X_train, y_train = create_sliding_windows(ts_scaled, WINDOW_SIZE)

# 3. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO LSTM
model = Sequential([
    Input(shape=(WINDOW_SIZE, 1)),
    LSTM(50, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
print("\n--- Entrenando Modelo LSTM ---")
history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# 4. PRONÓSTICO PARA ENERO-JUNIO 2025 (Autoregresivo / Multi-step)
HORIZON = 12  # 12 pasos quincenales = 6 meses (Ene - Jun 2025)
future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=14), periods=HORIZON, freq="2W")

# Predicción Autoregresiva LSTM
current_window = ts_scaled[-WINDOW_SIZE:].copy().reshape(1, WINDOW_SIZE, 1)
lstm_preds_scaled = []

for _ in range(HORIZON):
    pred = model.predict(current_window, verbose=0)
    lstm_preds_scaled.append(pred[0, 0])
    current_window = np.append(current_window[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

lstm_preds = scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1)).flatten()

# 5. PRONÓSTICO MODELO PARAMÉTRICO CON INTERVALO DE CONFIANZA 95%
# Ajustamos el modelo Sigmoide + FFT como ejemplo
t_hist = np.arange(len(ts)).reshape(-1, 1)
k = 0.5
t_jump_idx = np.searchsorted(ts.index, pd.to_datetime("2020-02-01"))
sigmoid_hist = 1 / (1 + np.exp(-k * (t_hist.flatten() - t_jump_idx))).reshape(-1, 1)
X_sig_hist = np.hstack([t_hist, sigmoid_hist])

lr = tf.keras.optimizers.Adam() # dummy reference
from sklearn.linear_model import LinearRegression
model_param = LinearRegression().fit(X_sig_hist, ts.values)
trend_fit = model_param.predict(X_sig_hist)

# Residuos e Intervalo de Confianza (95%)
residuals = ts.values - trend_fit
sigma_res = np.std(residuals)

# Extrapolación a 2025
t_future = np.arange(len(ts), len(ts) + HORIZON).reshape(-1, 1)
sigmoid_fut = 1 / (1 + np.exp(-k * (t_future.flatten() - t_jump_idx))).reshape(-1, 1)
X_sig_fut = np.hstack([t_future, sigmoid_fut])

trend_pred = model_param.predict(X_sig_fut)

# Intervalo de confianza creciente con el horizonte h
h = np.arange(1, HORIZON + 1)
ci_margin = 1.96 * sigma_res * np.sqrt(1 + h / len(ts))
param_preds = trend_pred  # En este horizonte asumimos tendencia proyectada + ciclo

# 6. VISUALIZACIÓN DEL PRONÓSTICO Y COMPARACIÓN (2017-2025)
plt.figure(figsize=(13, 6))
plt.plot(ts.index, ts.values, label="Histórico Observado (2017-2024)", color="black")
plt.plot(future_dates, param_preds, label="Pronóstico Paramétrico (Sigmoide)", color="blue", linestyle="--")
plt.fill_between(future_dates, param_preds - ci_margin, param_preds + ci_margin, color="blue", alpha=0.2, label="IC 95% (Paramétrico)")
plt.plot(future_dates, lstm_preds, label="Pronóstico LSTM", color="red", linestyle="-.")

plt.title("Pronóstico de la Línea de Costa Jan-Jun 2025: Paramétrico vs. LSTM")
plt.ylabel("Distancia (m)")
plt.xlabel("Fecha")
plt.legend()
plt.grid(True)
plt.savefig("forecast_2025_comparison.png")
plt.show()

print("\n¡Procesamiento y pronóstico completados con éxito!")