"""
Procesamiento por lotes de imágenes Sentinel-2 (nivel L2A).

Para cada .zip encontrado en CARPETA_ENTRADA:
1. Localiza la banda SCL (20m) dentro del zip sin descomprimirlo.
2. Calcula el % de nubosidad usando la clasificación de escena.
3. Si pasa el filtro de nubosidad, genera un compuesto RGB de 10m
   (B04/B03/B02) para visualización de costa.
4. Registra el resultado de cada zip en un CSV.
5. Mueve el zip procesado a otra carpeta para no repetirlo.

Requiere: rasterio, numpy, pillow
    pip install rasterio numpy pillow
(rasterio depende de GDAL; en la mayoría de sistemas se instala
 automáticamente junto con el paquete)
"""

import os
import csv
from datetime import datetime

import numpy as np
import rasterio
from PIL import Image
import zipfile

import matplotlib
matplotlib.use("Agg")  # backend sin interfaz: nunca abre ventana, no bloquea el bucle
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

CARPETA_ENTRADA = "Imagenes/zips-entrada"
CARPETA_PROCESADOS = "Imagenes/zips_procesados"
CARPETA_SALIDA = "Imagenes/resultados"
CSV_LOG = "Imagenes/registro_procesamiento.csv"

# % máximo de nubosidad aceptado para que la imagen pase el filtro
UMBRAL_NUBOSIDAD = 10.0

# Clases SCL (Scene Classification Layer) que se consideran "nube"
# 8 = nube probabilidad media, 9 = nube probabilidad alta, 10 = cirros
CLASES_NUBE = {8, 9, 10}
CLASE_NODATA = 0

# Bandas necesarias para el compuesto de costa (10m nativo)
BANDAS_COSTA = ["B04", "B03", "B02"]  # Rojo, Verde, Azul

# Umbral conservador de NDWI para considerar un píxel como agua
# (por encima de 0 para reducir falsos positivos de humedad/sombra)
UMBRAL_NDWI_AGUA = 0.2


# ---------------------------------------------------------------------------
# Localización de archivos dentro del zip
# ---------------------------------------------------------------------------

def buscar_archivo_en_zip(zip_path, patron):
    """
    Busca dentro del .zip (sin extraerlo) el primer archivo cuyo nombre
    contenga 'patron'. Devuelve la ruta interna o None si no lo encuentra.
    """
    with zipfile.ZipFile(zip_path) as z:
        for nombre in z.namelist():
            if patron in nombre:
                return nombre
    return None


def ruta_vsizip(zip_path, ruta_interna):
    """
    Construye la ruta especial que usa GDAL/rasterio para leer un archivo
    dentro de un .zip como si fuera un archivo normal en disco, sin
    necesidad de descomprimirlo.
    """
    return f"/vsizip/{os.path.abspath(zip_path)}/{ruta_interna}"


# ---------------------------------------------------------------------------
# Cálculo de nubosidad a partir de la SCL
# ---------------------------------------------------------------------------

def calcular_porcentaje_nubosidad(zip_path, ruta_scl):
    """
    Lee la banda SCL y calcula el % de píxeles clasificados como nube
    sobre el total de píxeles válidos (excluyendo 'no data').

    Devuelve (porcentaje_total, desglose_por_clase).
    """
    with rasterio.open(ruta_vsizip(zip_path, ruta_scl)) as src:
        scl = src.read(1)

    validos = scl != CLASE_NODATA
    total_validos = int(np.sum(validos))

    if total_validos == 0:
        raise ValueError("La SCL no tiene píxeles válidos (todo 'no data')")

    es_nube = np.isin(scl, list(CLASES_NUBE)) & validos
    porcentaje_total = float(np.sum(es_nube)) / total_validos * 100

    desglose = {
        clase: float(np.sum((scl == clase) & validos)) / total_validos * 100
        for clase in CLASES_NUBE
    }

    return porcentaje_total, desglose


# ---------------------------------------------------------------------------
# Lectura de bandas de 10m
# ---------------------------------------------------------------------------

def leer_banda(zip_path, ruta_interna):
    """
    Lee una banda dentro del zip y la devuelve como array float32.
    Se usa tanto para el compuesto de costa como para el NDWI, para no
    leer la misma banda (p.ej. B03) dos veces desde el zip.
    """
    with rasterio.open(ruta_vsizip(zip_path, ruta_interna)) as src:
        return src.read(1).astype(np.float32)


# ---------------------------------------------------------------------------
# Generación del compuesto de costa (10m)
# ---------------------------------------------------------------------------

def _normalizar_banda(banda, percentil=99):
    """
    Estira el contraste de una banda de reflectancia (valores altos, no
    0-255) a una escala visualizable 0-255, usando un percentil para
    evitar que unos pocos píxeles extremos aplasten el resto de la imagen.
    """
    maximo = np.percentile(banda, percentil)
    if maximo <= 0:
        maximo = 1.0
    banda_norm = np.clip(banda / maximo, 0, 1)
    return (banda_norm * 255).astype(np.uint8)


def generar_visual_costa(bandas, salida_path):
    """
    Arma un compuesto RGB a partir de bandas ya leídas (dict banda -> array
    float32) y lo guarda como imagen.
    """
    canales = [_normalizar_banda(bandas[nombre]) for nombre in BANDAS_COSTA]
    rgb = np.dstack(canales)
    Image.fromarray(rgb).save(salida_path)


# ---------------------------------------------------------------------------
# Cálculo y visualización de NDWI (índice de agua)
# ---------------------------------------------------------------------------

def calcular_ndwi(green, nir):
    """
    NDWI = (Verde - NIR) / (Verde + NIR)

    Ambas bandas deben venir ya en float32 (importante: si se restan como
    enteros sin signo, como venían originalmente de rasterio, se produce
    overflow y el resultado se corrompe justo en los píxeles de agua).
    """
    numerador = green - nir
    denominador = green + nir
    ndwi = np.divide(
        numerador,
        denominador,
        out=np.zeros_like(green, dtype=np.float32),
        where=denominador != 0,
    )
    return ndwi


def calcular_porcentaje_agua(ndwi, umbral=UMBRAL_NDWI_AGUA):
    """
    % de píxeles cuyo NDWI supera el umbral conservador de agua.
    """
    return float(np.mean(ndwi > umbral)) * 100


def generar_visual_ndwi(ndwi, salida_path):
    """
    Guarda una imagen coloreada del NDWI con colorbar. Usa savefig (nunca
    show) para no bloquear el procesamiento por lotes.
    """
    plt.figure()
    plt.imshow(ndwi, cmap="RdYlBu")
    plt.colorbar(label="NDWI")
    plt.title("NDWI")
    plt.axis("off")
    plt.savefig(salida_path, bbox_inches="tight", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Orquestador por imagen
# ---------------------------------------------------------------------------

def procesar_zip(zip_path, carpeta_salida):
    """
    Procesa un único .zip de principio a fin y devuelve un dict con el
    resultado (para el CSV de registro). Cualquier error se captura aquí
    para no detener el procesamiento del resto del lote.
    """
    nombre_zip = os.path.basename(zip_path)
    resultado = {
        "zip": nombre_zip,
        "fecha_proceso": datetime.now().isoformat(timespec="seconds"),
        "porcentaje_nubosidad": None,
        "paso_filtro": None,
        "archivo_salida": None,
        "archivo_ndwi": None,
        "porcentaje_agua_ndwi": None,
        "error": None,
    }

    try:
        ruta_scl = buscar_archivo_en_zip(zip_path, "_SCL_20m.jp2")
        if ruta_scl is None:
            raise FileNotFoundError("No se encontró la banda SCL (_SCL_20m.jp2) en el zip")

        porcentaje, _desglose = calcular_porcentaje_nubosidad(zip_path, ruta_scl)
        resultado["porcentaje_nubosidad"] = round(porcentaje, 2)
        resultado["paso_filtro"] = porcentaje <= UMBRAL_NUBOSIDAD

        if resultado["paso_filtro"]:
            # Localiza en el zip todas las bandas de 10m que necesitamos:
            # las de costa (RGB) más B08 (NIR) para el NDWI.
            bandas_necesarias = list(dict.fromkeys(BANDAS_COSTA + ["B08"]))
            rutas_bandas = {}
            for banda in bandas_necesarias:
                ruta = buscar_archivo_en_zip(zip_path, f"_{banda}_10m.jp2")
                if ruta is None:
                    raise FileNotFoundError(f"No se encontró la banda {banda} a 10m en el zip")
                rutas_bandas[banda] = ruta

            # Se lee cada banda una sola vez y se reutiliza (B03 sirve
            # tanto para el RGB de costa como para el NDWI).
            bandas = {banda: leer_banda(zip_path, ruta) for banda, ruta in rutas_bandas.items()}

            nombre_base = os.path.splitext(nombre_zip)[0]

            nombre_costa = nombre_base + "_costa.jpg"
            generar_visual_costa(bandas, os.path.join(carpeta_salida, nombre_costa))
            resultado["archivo_salida"] = nombre_costa

            ndwi = calcular_ndwi(bandas["B03"], bandas["B08"])
            nombre_ndwi = nombre_base + "_ndwi.jpg"
            generar_visual_ndwi(ndwi, os.path.join(carpeta_salida, nombre_ndwi))
            resultado["archivo_ndwi"] = nombre_ndwi
            resultado["porcentaje_agua_ndwi"] = round(calcular_porcentaje_agua(ndwi), 2)

    except Exception as e:
        resultado["error"] = str(e)

    return resultado


# ---------------------------------------------------------------------------
# Registro CSV
# ---------------------------------------------------------------------------

def escribir_registro(csv_path, filas):
    """
    Añade filas al CSV de registro, creando el encabezado si el archivo
    todavía no existe.
    """
    campos = [
        "zip", "fecha_proceso", "porcentaje_nubosidad",
        "paso_filtro", "archivo_salida",
        "archivo_ndwi", "porcentaje_agua_ndwi", "error",
    ]
    existe = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if not existe:
            writer.writeheader()
        for fila in filas:
            writer.writerow(fila)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

    if not os.path.isdir(CARPETA_ENTRADA):
        print(f"No existe la carpeta de entrada: {CARPETA_ENTRADA}")
        return

    zips = sorted(f for f in os.listdir(CARPETA_ENTRADA) if f.lower().endswith(".zip"))

    if not zips:
        print(f"No se encontraron archivos .zip en {CARPETA_ENTRADA}")
        return

    resultados = []
    for nombre in zips:
        zip_path = os.path.join(CARPETA_ENTRADA, nombre)
        print(f"Procesando {nombre}...")

        resultado = procesar_zip(zip_path, CARPETA_SALIDA)
        resultados.append(resultado)

        if resultado["error"]:
            print(f"  ERROR: {resultado['error']}")
        else:
            print(
                f"  Nubosidad: {resultado['porcentaje_nubosidad']}% "
                f"- Pasa filtro: {resultado['paso_filtro']}"
            )

        os.rename(zip_path, os.path.join(CARPETA_PROCESADOS, nombre))

    escribir_registro(CSV_LOG, resultados)
    print(f"\nProcesamiento completo. Registro guardado en: {CSV_LOG}")


if __name__ == "__main__":
    main()
