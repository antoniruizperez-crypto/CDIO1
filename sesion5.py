"""
Procesamiento por lotes de imágenes Sentinel-2.

Para cada .zip encontrado en CARPETA_ENTRADA:
1. Detecta si el producto es L1C o L2A (por nombre de archivo o XML).
2. Si es L2A: localiza la SCL (20m), calcula el % de nubosidad por
   píxel y, si pasa el filtro, genera compuesto de costa (10m RGB),
   NDWI (imagen coloreada + GeoTIFF continuo) y una máscara binaria
   tierra/agua (waterbody.tif) en la carpeta del proyecto de costa
   (Code/projects/<NOMBRE_PROYECTO>/output/estimated_waterbodies_images/),
   lista para extract_shorelines.py.
3. Si es L1C: lee el % de nubosidad global desde el XML de metadatos
   (Cloud_Coverage_Assessment). Solo se usa para registro/filtrado;
   no se generan imágenes, por la menor fiabilidad de la reflectancia
   TOA sin corrección atmosférica y la falta de máscara por píxel.
4. Registra el resultado de cada zip en un CSV.
5. Mueve el zip procesado a otra carpeta para no repetirlo.

Requiere: rasterio, numpy, pillow, matplotlib, pyproj, shapely, pandas
    pip install rasterio numpy pillow matplotlib pyproj shapely pandas
(rasterio depende de GDAL; en la mayoría de sistemas se instala
 automáticamente junto con el paquete)

Debe ejecutarse con el directorio de trabajo en la raíz del proyecto
(la carpeta que contiene tanto Code/ como Imagenes/), y shoreline_utils.py
debe estar en la misma carpeta que este script (Code/).
"""

import os
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
import zipfile

import matplotlib
matplotlib.use("Agg")  # backend sin interfaz: nunca abre ventana, no bloquea el bucle
import matplotlib.pyplot as plt

# shoreline_utils.py debe vivir en la misma carpeta que este script
import shoreline_utils as utils

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

CARPETA_ENTRADA = "Imagenes/zips_entrada"
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
# (se usa tanto para la estadística informativa como para el waterbody.tif)
UMBRAL_NDWI_AGUA = 0.2

# Proyecto de costa (carpeta bajo Code/projects/) usado por extract_shorelines.py
# y calculate_erosion.py vía shoreline_utils.project_path
NOMBRE_PROYECTO = "costa_principal"
CARPETA_WATERBODY = utils.project_path(NOMBRE_PROYECTO) / "output" / "estimated_waterbodies_images"


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
# Detección de nivel de procesamiento (L1C / L2A)
# ---------------------------------------------------------------------------

def detectar_nivel_procesamiento(zip_path):
    """
    Determina si el producto es L1C o L2A. Primero mira el nombre del zip
    (convención oficial de la ESA: 'MSIL1C' / 'MSIL2A'), y si no es
    concluyente, busca el XML de metadatos correspondiente dentro del zip.

    Devuelve "L1C", "L2A" o "DESCONOCIDO".
    """
    nombre = os.path.basename(zip_path)

    if "MSIL2A" in nombre:
        return "L2A"
    if "MSIL1C" in nombre:
        return "L1C"

    # Nombre no concluyente (p.ej. si el archivo fue renombrado):
    # se busca el XML de metadatos propio de cada nivel.
    if buscar_archivo_en_zip(zip_path, "MTD_MSIL2A.xml"):
        return "L2A"
    if buscar_archivo_en_zip(zip_path, "MTD_MSIL1C.xml"):
        return "L1C"

    return "DESCONOCIDO"


def leer_nubosidad_xml_l1c(zip_path, ruta_xml):
    """
    Lee el % de nubosidad global de la escena desde el XML de metadatos
    de un producto L1C (campo 'Cloud_Coverage_Assessment').

    Es un único valor para todo el granulado (no una máscara por píxel),
    calculado por la ESA.
    """
    with zipfile.ZipFile(zip_path) as z:
        contenido = z.read(ruta_xml)

    root = ET.fromstring(contenido)
    for elemento in root.iter():
        etiqueta = elemento.tag.split("}")[-1]  # quita el namespace XML
        if etiqueta == "Cloud_Coverage_Assessment":
            return float(elemento.text)

    raise ValueError("No se encontró 'Cloud_Coverage_Assessment' en el XML del producto L1C")


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

def leer_banda(zip_path, ruta_interna, con_profile=False):
    """
    Lee una banda dentro del zip y la devuelve como array float32.
    Se usa tanto para el compuesto de costa como para el NDWI, para no
    leer la misma banda (p.ej. B03) dos veces desde el zip.

    Si con_profile=True, devuelve también el profile de rasterio (CRS,
    transform, dimensiones) de esa banda, necesario para poder escribir
    después un .tif georreferenciado (ndwi.tif, waterbody.tif).
    """
    with rasterio.open(ruta_vsizip(zip_path, ruta_interna)) as src:
        datos = src.read(1).astype(np.float32)
        if con_profile:
            return datos, src.profile.copy()
        return datos


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


def guardar_ndwi_tif(ndwi, salida_path, profile):
    """
    Guarda el NDWI continuo (-1 a 1) como GeoTIFF de una banda, reutilizando
    el profile (CRS/transform) de la banda de origen. Es un producto de
    trazabilidad/inspección, no lo requiere extract_shorelines.py.
    """
    # El profile viene de una banda .jp2 (driver JP2OpenJPEG), que no admite
    # float32 ni otros tipos que sí soporta GeoTIFF. Hay que forzar el driver
    # a GTiff explícitamente; la extensión del nombre de archivo no basta,
    # GDAL usa el driver indicado en el profile.
    perfil_salida = profile.copy()
    perfil_salida.update(driver="GTiff", dtype=rasterio.float32, count=1)
    with rasterio.open(salida_path, "w", **perfil_salida) as dst:
        dst.write(ndwi.astype(np.float32), 1)


def generar_waterbody(ndwi, salida_path, profile, umbral=UMBRAL_NDWI_AGUA):
    """
    Genera la máscara binaria tierra/agua que espera extract_shorelines.py:
    una sola banda uint8 con valores exactos 0 (tierra) / 1 (agua),
    georreferenciada con el mismo profile que la banda de origen.
    """
    waterbody = (ndwi > umbral).astype(np.uint8)
    perfil_salida = profile.copy()
    perfil_salida.update(driver="GTiff", dtype=rasterio.uint8, count=1, nodata=None)
    with rasterio.open(salida_path, "w", **perfil_salida) as dst:
        dst.write(waterbody, 1)


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
        "nivel_procesamiento": None,
        "porcentaje_nubosidad": None,
        "paso_filtro": None,
        "archivo_salida": None,
        "archivo_ndwi": None,
        "archivo_ndwi_tif": None,
        "archivo_waterbody": None,
        "porcentaje_agua_ndwi": None,
        "observaciones": None,
        "error": None,
    }

    try:
        nivel = detectar_nivel_procesamiento(zip_path)
        resultado["nivel_procesamiento"] = nivel

        if nivel == "L2A":
            _procesar_l2a(zip_path, carpeta_salida, resultado)

        elif nivel == "L1C":
            _procesar_l1c(zip_path, resultado)

        else:
            raise ValueError("No se pudo determinar el nivel de procesamiento (ni L1C ni L2A)")

    except Exception as e:
        resultado["error"] = str(e)

    return resultado


def _procesar_l2a(zip_path, carpeta_salida, resultado):
    """
    Flujo completo para productos L2A: nubosidad vía SCL y, si pasa el
    filtro, generación de compuesto de costa y NDWI.
    """
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

        # Se lee cada banda una sola vez y se reutiliza (B03 sirve tanto
        # para el RGB de costa como para el NDWI). El profile (CRS,
        # transform) se captura de la primera banda leída y es el mismo
        # para todas, al ser todas bandas nativas de 10m del mismo tile.
        bandas = {}
        profile_10m = None
        for banda, ruta in rutas_bandas.items():
            if profile_10m is None:
                bandas[banda], profile_10m = leer_banda(zip_path, ruta, con_profile=True)
            else:
                bandas[banda] = leer_banda(zip_path, ruta)

        nombre_zip = resultado["zip"]
        nombre_base = os.path.splitext(nombre_zip)[0]

        nombre_costa = nombre_base + "_costa.jpg"
        generar_visual_costa(bandas, os.path.join(carpeta_salida, nombre_costa))
        resultado["archivo_salida"] = nombre_costa

        ndwi = calcular_ndwi(bandas["B03"], bandas["B08"])

        nombre_ndwi = nombre_base + "_ndwi.jpg"
        generar_visual_ndwi(ndwi, os.path.join(carpeta_salida, nombre_ndwi))
        resultado["archivo_ndwi"] = nombre_ndwi
        resultado["porcentaje_agua_ndwi"] = round(calcular_porcentaje_agua(ndwi), 2)

        # NDWI continuo como GeoTIFF (trazabilidad; no lo usa extract_shorelines.py)
        nombre_ndwi_tif = nombre_base + "_ndwi.tif"
        guardar_ndwi_tif(ndwi, os.path.join(carpeta_salida, nombre_ndwi_tif), profile_10m)
        resultado["archivo_ndwi_tif"] = nombre_ndwi_tif

        # Máscara binaria tierra/agua, en la carpeta que espera extract_shorelines.py.
        # El nombre del zip se conserva intacto (solo se le añade un sufijo) para
        # que shoreline_utils.extract_date siga encontrando la fecha en el nombre.
        nombre_waterbody = nombre_base + "_waterbody.tif"
        ruta_waterbody = CARPETA_WATERBODY / nombre_waterbody
        generar_waterbody(ndwi, ruta_waterbody, profile_10m)
        resultado["archivo_waterbody"] = nombre_waterbody


def _procesar_l1c(zip_path, resultado):
    """
    Flujo reducido para productos L1C: sin SCL disponible, se usa el %
    de nubosidad global del XML de metadatos solo para registro y
    filtrado. No se generan imágenes de costa ni NDWI, porque la
    reflectancia L1C (TOA, sin corrección atmosférica) y la ausencia de
    una máscara de nubes por píxel hacen que esos productos sean menos
    fiables que un L2A equivalente.
    """
    ruta_xml = buscar_archivo_en_zip(zip_path, "MTD_MSIL1C.xml")
    if ruta_xml is None:
        raise FileNotFoundError("No se encontró el XML de metadatos MTD_MSIL1C.xml en el zip")

    porcentaje = leer_nubosidad_xml_l1c(zip_path, ruta_xml)
    resultado["porcentaje_nubosidad"] = round(porcentaje, 2)
    resultado["paso_filtro"] = porcentaje <= UMBRAL_NUBOSIDAD
    resultado["observaciones"] = (
        "L1C: % de nubosidad global desde XML (sin máscara por píxel); "
        "no se generan imágenes de costa ni NDWI para este nivel."
    )


# ---------------------------------------------------------------------------
# Registro CSV
# ---------------------------------------------------------------------------

def escribir_registro(csv_path, filas):
    """
    Añade filas al CSV de registro, creando el encabezado si el archivo
    todavía no existe.
    """
    campos = [
        "zip", "fecha_proceso", "nivel_procesamiento",
        "porcentaje_nubosidad", "paso_filtro", "archivo_salida",
        "archivo_ndwi", "archivo_ndwi_tif", "archivo_waterbody",
        "porcentaje_agua_ndwi", "observaciones", "error",
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
    os.makedirs(CARPETA_WATERBODY, exist_ok=True)

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
                f"  Nivel: {resultado['nivel_procesamiento']} "
                f"- Nubosidad: {resultado['porcentaje_nubosidad']}% "
                f"- Pasa filtro: {resultado['paso_filtro']}"
            )

        os.rename(zip_path, os.path.join(CARPETA_PROCESADOS, nombre))

    escribir_registro(CSV_LOG, resultados)
    print(f"\nProcesamiento completo. Registro guardado en: {CSV_LOG}")


if __name__ == "__main__":
    main()