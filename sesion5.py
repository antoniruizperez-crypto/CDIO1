"""
Procesamiento por lotes de imágenes Sentinel-2 L2A.

Para cada .zip de CARPETA_ENTRADA:
 1. Detecta si es L1C o L2A.
 2. L2A: recorta la escena al AOI (no lee los 100x100 km completos), calcula el
    % de nube/sombra SOLO dentro del AOI usando la SCL y, si pasa el filtro:
      - convierte a reflectancia aplicando el offset BOA del XML (los productos
        con baseline >= 04.00 llevan +1000 que sesga el NDWI si no se resta),
      - calcula el NDWI enmascarando nubes, sombras y datos inválidos (NaN),
      - guarda el NDWI continuo (GeoTIFF float32, entrada de extract_shorelines.py),
        una máscara agua/tierra de inspección, el RGB y una imagen del NDWI.
 3. L1C: solo registra la nubosidad global del XML.
 4. Anota cada zip en el CSV de registro INMEDIATAMENTE (no al final).
 5. Mueve el zip a zips_procesados, o a zips_error si falló (para reintentarlo).

Uso:  python sesion5.py [-p proyecto]
Requiere: rasterio numpy pillow matplotlib pyproj shapely pandas geopandas scipy scikit-image
"""

import argparse
import csv
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import geometry_mask
from rasterio.windows import Window
from scipy.ndimage import binary_dilation

matplotlib.use("Agg")  # sin ventanas: no bloquea el bucle
import matplotlib.pyplot as plt

import config
import shoreline_utils as utils

BANDAS_COSTA = ["B04", "B03", "B02"]  # R, G, B (10 m)

CAMPOS_LOG = [
    "zip", "fecha_proceso", "nivel_procesamiento", "porcentaje_nubosidad_aoi",
    "cobertura_aoi", "paso_filtro", "umbral_ndwi_preview", "porcentaje_agua_aoi",
    "archivo_rgb", "archivo_ndwi_jpg", "archivo_ndwi_tif", "archivo_waterbody",
    "observaciones", "error",
]


# ---------------------------------------------------------------------------
# Acceso a archivos dentro del zip
# ---------------------------------------------------------------------------

def buscar_archivo_en_zip(zip_path, patron):
    """Primer archivo del zip (sin extraerlo) cuyo nombre contiene 'patron', o None."""
    with zipfile.ZipFile(zip_path) as z:
        for nombre in z.namelist():
            if patron in nombre:
                return nombre
    return None


def ruta_vsizip(zip_path, ruta_interna):
    """Ruta /vsizip/ para que GDAL/rasterio lea un archivo dentro del zip sin descomprimir."""
    return f"/vsizip/{os.path.abspath(zip_path)}/{ruta_interna}"


# ---------------------------------------------------------------------------
# Nivel de procesamiento y metadatos
# ---------------------------------------------------------------------------

def detectar_nivel_procesamiento(zip_path):
    nombre = os.path.basename(zip_path)
    if "MSIL2A" in nombre:
        return "L2A"
    if "MSIL1C" in nombre:
        return "L1C"
    if buscar_archivo_en_zip(zip_path, "MTD_MSIL2A.xml"):
        return "L2A"
    if buscar_archivo_en_zip(zip_path, "MTD_MSIL1C.xml"):
        return "L1C"
    return "DESCONOCIDO"


def leer_nubosidad_xml_l1c(zip_path, ruta_xml):
    """% de nubosidad global de un L1C (Cloud_Coverage_Assessment)."""
    with zipfile.ZipFile(zip_path) as z:
        root = ET.fromstring(z.read(ruta_xml))
    for el in root.iter():
        if el.tag.split("}")[-1] == "Cloud_Coverage_Assessment":
            return float(el.text)
    raise ValueError("No se encontró 'Cloud_Coverage_Assessment' en el XML del producto L1C")


def leer_offset_y_escala_boa(zip_path):
    """
    (offset, escala) para pasar de DN a reflectancia: refl = (DN + offset) / escala.
    Se lee de MTD_MSIL2A.xml (BOA_ADD_OFFSET, BOA_QUANTIFICATION_VALUE). Si no
    está, se deduce de la baseline del nombre (N0400 o superior -> offset -1000).
    """
    escala, offsets = 10000.0, []
    ruta_xml = buscar_archivo_en_zip(zip_path, "MTD_MSIL2A.xml")
    if ruta_xml:
        with zipfile.ZipFile(zip_path) as z:
            root = ET.fromstring(z.read(ruta_xml))
        for el in root.iter():
            etiqueta = el.tag.split("}")[-1]
            if etiqueta == "BOA_ADD_OFFSET" and el.text:
                offsets.append(float(el.text))
            elif etiqueta == "BOA_QUANTIFICATION_VALUE" and el.text:
                escala = float(el.text)
    if offsets:
        return offsets[0], escala
    m = re.search(r"_N(\d{4})_", os.path.basename(zip_path))
    return (-1000.0 if (m and int(m.group(1)) >= 400) else 0.0), escala


# ---------------------------------------------------------------------------
# Recorte al AOI
# ---------------------------------------------------------------------------

def cargar_aoi(proyecto):
    ruta = utils.project_path(proyecto) / "input" / "polygon.geojson"
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el polígono del AOI: {ruta}")
    return gpd.read_file(ruta).set_crs(4326, allow_override=True)


def ventana_aoi(src, aoi_proj, margen):
    """Ventana (en píxeles de src) que contiene el AOI más un margen, recortada a la escena."""
    minx, miny, maxx, maxy = aoi_proj.total_bounds
    inv = ~src.transform
    c0, r0 = inv * (minx, maxy)
    c1, r1 = inv * (maxx, miny)
    col0 = max(int(np.floor(min(c0, c1))) - margen, 0)
    row0 = max(int(np.floor(min(r0, r1))) - margen, 0)
    col1 = min(int(np.ceil(max(c0, c1))) + margen, src.width)
    row1 = min(int(np.ceil(max(r0, r1))) + margen, src.height)
    if col1 <= col0 or row1 <= row0:
        raise ValueError("El AOI no intersecta con esta escena")
    return Window(col0, row0, col1 - col0, row1 - row0)


def leer_banda_ventana(zip_path, ruta_interna, ventana):
    with rasterio.open(ruta_vsizip(zip_path, ruta_interna)) as src:
        return src.read(1, window=ventana).astype(np.float32)


def leer_scl_ventana(zip_path, ruta_scl, ventana, ancho10, alto10):
    """SCL (20 m) recortada y llevada a la rejilla de 10 m de la ventana (vecino más cercano)."""
    with rasterio.open(ruta_vsizip(zip_path, ruta_scl)) as s20:
        if s20.width * 2 != ancho10 or s20.height * 2 != alto10:
            raise ValueError("La SCL de 20 m no es exactamente la mitad de la rejilla de 10 m")
        col0, row0 = int(ventana.col_off), int(ventana.row_off)
        c0, r0 = col0 // 2, row0 // 2
        c1 = -(-(col0 + int(ventana.width)) // 2)
        r1 = -(-(row0 + int(ventana.height)) // 2)
        scl20 = s20.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    scl = np.repeat(np.repeat(scl20, 2, axis=0), 2, axis=1)
    oc, orow = col0 - 2 * c0, row0 - 2 * r0
    return scl[orow:orow + int(ventana.height), oc:oc + int(ventana.width)]


# ---------------------------------------------------------------------------
# Nubes
# ---------------------------------------------------------------------------

def analizar_nubes(scl, aoi_mask):
    """
    % de nube/sombra y cobertura de datos SOLO dentro del AOI.
    Devuelve (porcentaje_nube, cobertura, mascara_invalida_ensanchada).
    """
    sin_datos = scl == 0
    validos_aoi = aoi_mask & ~sin_datos
    n_aoi = max(int(aoi_mask.sum()), 1)
    cobertura = float(validos_aoi.sum()) / n_aoi
    es_nube = np.isin(scl, list(config.CLASES_NUBE))
    n_validos = int(validos_aoi.sum())
    pct = float(es_nube[validos_aoi].sum()) / n_validos * 100 if n_validos else 100.0

    invalido = np.isin(scl, list(config.CLASES_INVALIDAS))
    if config.DILATACION_NUBE_PX > 0 and invalido.any():
        invalido = binary_dilation(invalido, iterations=config.DILATACION_NUBE_PX)
    return pct, cobertura, invalido


# ---------------------------------------------------------------------------
# NDWI e imágenes
# ---------------------------------------------------------------------------

def calcular_ndwi(green, nir):
    """
    NDWI = (Verde - NIR) / (Verde + NIR), en float32. Si el denominador es 0 el
    píxel queda en NaN (inválido), no en 0: un 0 se habría tomado por tierra.
    """
    num, den = green - nir, green + nir
    ndwi = np.full(green.shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=ndwi, where=den != 0)
    return ndwi


def _normalizar_banda(banda, percentil=99):
    """Estira el contraste a 0-255 con un percentil, ignorando NaN."""
    finitos = banda[np.isfinite(banda) & (banda > 0)]
    maximo = np.percentile(finitos, percentil) if finitos.size else 1.0
    if maximo <= 0:
        maximo = 1.0
    return (np.clip(np.nan_to_num(banda) / maximo, 0, 1) * 255).astype(np.uint8)


def generar_visual_costa(bandas, salida_path):
    canales = [_normalizar_banda(bandas[n]) for n in BANDAS_COSTA]
    Image.fromarray(np.dstack(canales)).save(salida_path)


def generar_visual_ndwi(ndwi, salida_path):
    plt.figure()
    plt.imshow(ndwi, cmap="RdYlBu", vmin=-1, vmax=1)
    plt.colorbar(label="NDWI")
    plt.title("NDWI")
    plt.axis("off")
    plt.savefig(salida_path, bbox_inches="tight", dpi=150)
    plt.close()


def guardar_ndwi_tif(ndwi, salida_path, profile, tags):
    """NDWI continuo (NaN = inválido) como GeoTIFF; es la entrada de extract_shorelines.py."""
    p = profile.copy()
    p.update(driver="GTiff", dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(salida_path, "w", **p) as dst:
        dst.write(ndwi.astype(np.float32), 1)
        dst.update_tags(**tags)


def guardar_waterbody_preview(ndwi, umbral, salida_path, profile):
    """Máscara de inspección: 0 tierra, 1 agua, 255 inválido (no es la que usa la extracción)."""
    p = profile.copy()
    p.update(driver="GTiff", dtype=rasterio.uint8, count=1, nodata=255)
    m = np.where(np.isfinite(ndwi), (ndwi > umbral).astype(np.uint8), 255).astype(np.uint8)
    with rasterio.open(salida_path, "w", **p) as dst:
        dst.write(m, 1)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def procesar_zip(zip_path, proyecto, aoi_gdf):
    """Procesa un zip; cualquier error se captura para no parar el lote."""
    resultado = {k: None for k in CAMPOS_LOG}
    resultado["zip"] = os.path.basename(zip_path)
    resultado["fecha_proceso"] = datetime.now().isoformat(timespec="seconds")
    try:
        nivel = detectar_nivel_procesamiento(zip_path)
        resultado["nivel_procesamiento"] = nivel
        if nivel == "L2A":
            _procesar_l2a(zip_path, proyecto, aoi_gdf, resultado)
        elif nivel == "L1C":
            _procesar_l1c(zip_path, resultado)
        else:
            raise ValueError("No se pudo determinar el nivel de procesamiento (ni L1C ni L2A)")
    except Exception as e:  # noqa: BLE001 - se registra y se sigue con el lote
        resultado["error"] = f"{type(e).__name__}: {e}"
    return resultado


def _procesar_l2a(zip_path, proyecto, aoi_gdf, resultado):
    ruta_scl = buscar_archivo_en_zip(zip_path, "_SCL_20m.jp2")
    if ruta_scl is None:
        raise FileNotFoundError("No se encontró la banda SCL (_SCL_20m.jp2) en el zip")
    bandas_nec = list(dict.fromkeys(BANDAS_COSTA + ["B08"]))
    rutas = {}
    for b in bandas_nec:
        rutas[b] = buscar_archivo_en_zip(zip_path, f"_{b}_10m.jp2")
        if rutas[b] is None:
            raise FileNotFoundError(f"No se encontró la banda {b} a 10 m en el zip")

    with rasterio.open(ruta_vsizip(zip_path, rutas["B03"])) as src:
        aoi_proj = aoi_gdf.to_crs(src.crs)
        ventana = ventana_aoi(src, aoi_proj, config.MARGEN_VENTANA_PX)
        transform = src.window_transform(ventana)
        profile = src.profile.copy()
        ancho10, alto10 = src.width, src.height
    alto, ancho = int(ventana.height), int(ventana.width)
    profile.update(height=alto, width=ancho, transform=transform)

    aoi_mask = geometry_mask(list(aoi_proj.geometry), transform=transform, invert=True, out_shape=(alto, ancho))
    scl = leer_scl_ventana(zip_path, ruta_scl, ventana, ancho10, alto10)
    pct_nube, cobertura, invalido = analizar_nubes(scl, aoi_mask)
    resultado["porcentaje_nubosidad_aoi"] = round(pct_nube, 2)
    resultado["cobertura_aoi"] = round(cobertura, 3)

    if cobertura < config.COBERTURA_MIN_AOI:
        resultado["paso_filtro"] = False
        resultado["observaciones"] = f"Cobertura del AOI {cobertura:.0%} < {config.COBERTURA_MIN_AOI:.0%}"
        return
    resultado["paso_filtro"] = pct_nube <= config.UMBRAL_NUBOSIDAD_AOI
    if not resultado["paso_filtro"]:
        return

    offset, escala = leer_offset_y_escala_boa(zip_path)
    bandas = {}
    for b, ruta in rutas.items():
        dn = leer_banda_ventana(zip_path, ruta, ventana)
        invalido |= dn == 0
        bandas[b] = np.clip((dn + offset) / escala, 0, None)

    ndwi = calcular_ndwi(bandas["B03"], bandas["B08"])
    ndwi[invalido] = np.nan

    base = os.path.splitext(resultado["zip"])[0]
    os.makedirs(config.CARPETA_SALIDA, exist_ok=True)
    nombre = base + "_costa.jpg"
    generar_visual_costa(bandas, str(config.CARPETA_SALIDA / nombre))
    resultado["archivo_rgb"] = nombre
    nombre = base + "_ndwi.jpg"
    generar_visual_ndwi(ndwi, str(config.CARPETA_SALIDA / nombre))
    resultado["archivo_ndwi_jpg"] = nombre

    out_dir = utils.project_path(proyecto) / "output"
    (out_dir / "estimated_ndwi").mkdir(parents=True, exist_ok=True)
    (out_dir / "estimated_waterbodies_images").mkdir(parents=True, exist_ok=True)

    umbral, metodo = utils.compute_ndwi_threshold(ndwi[aoi_mask & np.isfinite(ndwi)])
    nombre = base + "_ndwi.tif"
    guardar_ndwi_tif(ndwi, out_dir / "estimated_ndwi" / nombre, profile, {
        "CLOUD_PCT_AOI": f"{pct_nube:.2f}", "AOI_COVERAGE": f"{cobertura:.3f}",
        "BOA_OFFSET": f"{offset:g}", "SOURCE_ZIP": resultado["zip"],
    })
    resultado["archivo_ndwi_tif"] = nombre
    nombre = base + "_waterbody.tif"
    guardar_waterbody_preview(ndwi, umbral, out_dir / "estimated_waterbodies_images" / nombre, profile)
    resultado["archivo_waterbody"] = nombre
    resultado["umbral_ndwi_preview"] = round(umbral, 4)
    en_aoi = ndwi[aoi_mask & np.isfinite(ndwi)]
    resultado["porcentaje_agua_aoi"] = round(float(np.mean(en_aoi > umbral)) * 100, 2) if en_aoi.size else None
    resultado["observaciones"] = f"umbral {metodo}; offset BOA {offset:g}"


def _procesar_l1c(zip_path, resultado):
    """L1C: solo registro (sin SCL por píxel ni corrección atmosférica no se generan imágenes)."""
    ruta_xml = buscar_archivo_en_zip(zip_path, "MTD_MSIL1C.xml")
    if ruta_xml is None:
        raise FileNotFoundError("No se encontró MTD_MSIL1C.xml en el zip")
    pct = leer_nubosidad_xml_l1c(zip_path, ruta_xml)
    resultado["porcentaje_nubosidad_aoi"] = round(pct, 2)
    resultado["paso_filtro"] = pct <= config.UMBRAL_NUBOSIDAD_AOI
    resultado["observaciones"] = "L1C: nubosidad global del XML; no se generan imágenes (sin SCL ni corrección atmosférica)"


# ---------------------------------------------------------------------------
# Registro y programa principal
# ---------------------------------------------------------------------------

def escribir_registro(csv_path, fila):
    """Añade una fila al CSV. Si el archivo existe con otro formato, lo aparta como .bak."""
    csv_path = str(csv_path)
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            if f.readline().strip().split(",") != CAMPOS_LOG:
                os.replace(csv_path, csv_path + ".bak")
    existe = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_LOG)
        if not existe:
            w.writeheader()
        w.writerow(fila)


def main(proyecto=config.NOMBRE_PROYECTO):
    for carpeta in (config.CARPETA_SALIDA, config.CARPETA_PROCESADOS, config.CARPETA_ERROR):
        os.makedirs(carpeta, exist_ok=True)
    if not config.CARPETA_ENTRADA.is_dir():
        print(f"No existe la carpeta de entrada: {config.CARPETA_ENTRADA}")
        return
    zips = sorted(f for f in os.listdir(config.CARPETA_ENTRADA) if f.lower().endswith(".zip"))
    if not zips:
        print(f"No se encontraron archivos .zip en {config.CARPETA_ENTRADA}")
        return

    aoi_gdf = cargar_aoi(proyecto)
    for nombre in zips:
        zip_path = str(config.CARPETA_ENTRADA / nombre)
        print(f"Procesando {nombre}...")
        r = procesar_zip(zip_path, proyecto, aoi_gdf)
        if r["error"]:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Nivel: {r['nivel_procesamiento']} - Nubes en AOI: {r['porcentaje_nubosidad_aoi']}% "
                  f"- Cobertura AOI: {r['cobertura_aoi']} - Pasa filtro: {r['paso_filtro']}")
        escribir_registro(config.CSV_LOG, r)
        destino = config.CARPETA_ERROR if r["error"] else config.CARPETA_PROCESADOS
        os.replace(zip_path, str(destino / nombre))

    print(f"\nProcesamiento completo. Registro en: {config.CSV_LOG}")
    print("Siguiente paso: python extract_shorelines.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Procesa los zips Sentinel-2 de la carpeta de entrada.")
    ap.add_argument("-p", "--project", default=config.NOMBRE_PROYECTO)
    main(ap.parse_args().project)
