import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# 1. CARGA Y PREPARACIÓN DE DATOS (Parte 1)
CSV_PATH = "projects/costa_principal/output/estimated_erosion_accretion/shoreline_distances.csv"
df = pd.read_csv(CSV_PATH, parse_dates=["date"])

# Crear tabla pivote: Filas = Fechas, Columnas = Transectos
pivot_df = df.pivot(index="date", columns="transect_id", values="distance_m")

# A. Filtrar transectos con > 30% de NaNs
nan_threshold = 0.30
valid_transects = pivot_df.columns[pivot_df.isna().mean() <= nan_threshold]
pivot_filtered = pivot_df[valid_transects].copy()

# B. Filtrar Outliers por transecto usando el método IQR
for col in pivot_filtered.columns:
    Q1 = pivot_filtered[col].quantile(0.25)
    Q3 = pivot_filtered[col].quantile(0.75)
    IQR = Q3 - Q1
    pivot_filtered[col] = pivot_filtered[col].mask(
        (pivot_filtered[col] < (Q1 - 1.5 * IQR)) | (pivot_filtered[col] > (Q3 + 1.5 * IQR))
    )

# C. Serie temporal agregada (media entre transectos válidos)
ts_raw = pivot_filtered.mean(axis=1)

# D. Resampleado (ej. Quincenal '2W' o Mensual 'MS') e Interpolación lineal
ts = ts_raw.resample("2W").mean().interpolate(method="linear")
print(f"Serie temporal procesada: {len(ts)} muestras entre {ts.index.min().date()} y {ts.index.max().date()}")

# 2. DINÁMICA DE EROSIÓN / ACRECIÓN (Parte 2)
# Derivadas (Velocidad y Aceleración)
dt = 1/24  # Suponiendo resampleado quincenal (~24 muestras por año)
velocity = np.gradient(ts.values, dt)       # m/año
acceleration = np.gradient(velocity, dt)   # m/año²

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axs[0].plot(ts.index, ts.values, label="Posición Costa (m)", color="b")
axs[0].set_ylabel("Distancia (m)")
axs[0].set_title("Evolución de la Línea de Costa y Dinámica de Erosión")
axs[0].grid(True)

axs[1].plot(ts.index, velocity, label="Velocidad (m/año)", color="orange")
axs[1].axhline(0, color="black", linestyle="--", alpha=0.5)
axs[1].set_ylabel("Velocidad (m/año)")
axs[1].grid(True)

axs[2].plot(ts.index, acceleration, label="Aceleración (m/año²)", color="red")
axs[2].axhline(0, color="black", linestyle="--", alpha=0.5)
axs[2].set_ylabel("Aceleración (m/año²)")
axs[2].set_xlabel("Fecha")
axs[2].grid(True)
plt.tight_layout()
plt.savefig("dinamica_erosion.png")
plt.show()

# 3. MODELADO PARAMÉTRICO (3 Opciones de Tendencia + FFT)
t = np.arange(len(ts)).reshape(-1, 1)

# A. Modelo 1: Tendencia Lineal + FFT
model_lin = LinearRegression().fit(t, ts.values)
trend_lin = model_lin.predict(t)

# B. Modelo 2: Tendencia Step (Gloria - Feb 2020) + FFT
t_jump_idx = np.searchsorted(ts.index, pd.to_datetime("2020-02-01"))
step = (t.flatten() >= t_jump_idx).astype(int).reshape(-1, 1)
X_step = np.hstack([t, step])
model_step = LinearRegression().fit(X_step, ts.values)
trend_step = model_step.predict(X_step)

# C. Modelo 3: Tendencia Sigmoide + FFT
k = 0.5  # Velocidad de transición
sigmoid = 1 / (1 + np.exp(-k * (t.flatten() - t_jump_idx))).reshape(-1, 1)
X_sig = np.hstack([t, sigmoid])
model_sig = LinearRegression().fit(X_sig, ts.values)
trend_sig = model_sig.predict(X_sig)

# Función para descomponer armónicos mediante FFT
def fit_fft_cyclical(ts_detrended, n_harmonics=2):
    fft_vals = fft(ts_detrended)
    freqs = fftfreq(len(ts_detrended))
    mags = np.abs(fft_vals)
    
    pos_mask = freqs > 0
    top_indices = np.argsort(mags[pos_mask])[-n_harmonics:]
    top_freqs = freqs[pos_mask][top_indices]
    
    fft_filtered = np.zeros_like(fft_vals)
    for f in top_freqs:
        idx = np.where(freqs == f)[0]
        idx_neg = np.where(freqs == -f)[0]
        fft_filtered[idx] = fft_vals[idx]
        fft_filtered[idx_neg] = fft_vals[idx_neg]
        
    return np.real(ifft(fft_filtered))

# Reconstrucción de cada modelo
cyclical_lin = fit_fft_cyclical(ts.values - trend_lin)
reconstructed_lin = trend_lin + cyclical_lin

cyclical_step = fit_fft_cyclical(ts.values - trend_step)
reconstructed_step = trend_step + cyclical_step

cyclical_sig = fit_fft_cyclical(ts.values - trend_sig)
reconstructed_sig = trend_sig + cyclical_sig

# 4. EVALUACIÓN Y COMPARACIÓN DE MODELOS
def get_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

metrics = {
    "Linear + FFT": get_metrics(ts.values, reconstructed_lin),
    "Step + FFT": get_metrics(ts.values, reconstructed_step),
    "Sigmoid + FFT": get_metrics(ts.values, reconstructed_sig),
}

print("\n--- RESULTADOS DE LOS MODELOS PARAMÉTRICOS ---")
for m_name, m_vals in metrics.items():
    print(f"{m_name:15s} | MAE: {m_vals['MAE']:.3f} m | RMSE: {m_vals['RMSE']:.3f} m | R2: {m_vals['R2']:.3f}")

# Gráfica comparativa
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts.values, label="Observado", color="gray", alpha=0.6)
plt.plot(ts.index, reconstructed_lin, label=f"Lineal+FFT (R²={metrics['Linear + FFT']['R2']:.2f})")
plt.plot(ts.index, reconstructed_step, label=f"Step+FFT (R²={metrics['Step + FFT']['R2']:.2f})")
plt.plot(ts.index, reconstructed_sig, label=f"Sigmoide+FFT (R²={metrics['Sigmoid + FFT']['R2']:.2f})")
plt.title("Comparación de Modelos Paramétricos para la Línea de Costa")
plt.ylabel("Distancia (m)")
plt.xlabel("Fecha")
plt.legend()
plt.grid(True)
plt.savefig("comparacion_modelos_parametricos.png")
plt.show()