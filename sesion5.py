"""
Procesamiento Sentinel-2 para costa y preparación de calibración.

Dependencias:
    pip install rasterio numpy pillow matplotlib

El modo costa necesita shoreline_utils.py.

Uso para costa:
    python sesion5.py --modo costa

Uso para calibración de Shadnagar:
    python sesion5.py --modo calibracion

Área predeterminada:
    0,1 km² = 100.000 m².

Para indicar otra área:
    python sesion5.py --modo calibracion --area-m2 100000

Notas:
- Los recortes se ajustan a píxeles completos, sin remuestreo.
- El área realmente exportada se guarda en metadatos.json.
- Se conservan los valores digitales originales en calibración.
- Los ZIP no se mueven por defecto.
- Este script no mide el objetivo ni calcula los RMSE.
- Este script no ejecuta Sen2Cor.
"""

import argparse
import csv
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform as transformar_coordenadas
from rasterio.windows import Window


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

CARPETA_ENTRADA = "Imagenes/zips_entrada"
CARPETA_PROCESADOS = "Imagenes/zips_procesados"
CARPETA_SALIDA = "Imagenes/resultados"

ENTRADA_CALIBRACION = "Imagenes/calibracion/shadnagar"
SALIDA_CALIBRACION = "Imagenes/calibracion/resultados"

NOMBRE_PROYECTO = "costa_principal"

UMBRAL_NUBOSIDAD = 10.0
UMBRAL_NDWI_AGUA = 0.2

# 0,1 km² = 100.000 m².
AREA_RECORTE_M2 = 100000.0

SHADNAGAR_LAT = 17.034261
SHADNAGAR_LON = 78.183078

CLASES_NUBE = {8, 9, 10}
CLASE_NODATA = 0

BANDAS = ["B02", "B03", "B04", "B08"]
BANDAS_RGB = ["B04", "B03", "B02"]

ID_BANDA_XML = {
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B08": 7,
}

CAMPOS_LOG = [
    "zip",
    "fecha_proceso",
    "modo",
    "nivel",
    "satelite",
    "fecha_adquisicion",
    "estado",
    "nubosidad",
    "ambito_nubosidad",
    "paso_filtro",
    "carpeta_salida",
    "archivo_waterbody",
    "porcentaje_agua",
    "observaciones",
    "error",
]


# ---------------------------------------------------------------------------
# ZIP y metadatos
# ---------------------------------------------------------------------------

def nombre_local(etiqueta):
    return etiqueta.split("}")[-1]


def listar_zip(zip_path):
    with zipfile.ZipFile(zip_path) as archivo:
        return [
            nombre
            for nombre in archivo.namelist()
            if not nombre.endswith("/")
        ]


def ruta_vsizip(zip_path, ruta_interna):
    absoluto = Path(zip_path).resolve().as_posix()
    return f"/vsizip/{absoluto}/{ruta_interna}"


def detectar_nivel(zip_path, miembros):
    nombre = Path(zip_path).name

    if "MSIL2A" in nombre:
        return "L2A"

    if "MSIL1C" in nombre:
        return "L1C"

    for nivel in ("L2A", "L1C"):
        esperado = f"MTD_MSI{nivel}.xml"

        if any(Path(p).name == esperado for p in miembros):
            return nivel

    raise ValueError("No se reconoce el nivel L1C/L2A.")


def localizar_xml(miembros, nivel):
    esperado = f"MTD_MSI{nivel}.xml"

    candidatos = [
        p for p in miembros
        if Path(p).name == esperado
    ]

    if len(candidatos) != 1:
        raise ValueError(
            f"Se esperaba un XML {esperado}; "
            f"se encontraron {len(candidatos)}."
        )

    return candidatos[0]


def leer_metadatos(zip_path, ruta_xml, nivel):
    with zipfile.ZipFile(zip_path) as archivo:
        raiz = ET.fromstring(archivo.read(ruta_xml))

    texto = {}

    for elemento in raiz.iter():
        etiqueta = nombre_local(elemento.tag)
        valor = (elemento.text or "").strip()

        if valor:
            texto.setdefault(etiqueta, valor)

    etiqueta_q = (
        "BOA_QUANTIFICATION_VALUE"
        if nivel == "L2A"
        else "QUANTIFICATION_VALUE"
    )

    valor_q = texto.get(etiqueta_q)

    if valor_q is None:
        valor_q = texto.get("QUANTIFICATION_VALUE")

    cuantificacion = (
        float(valor_q) if valor_q is not None else None
    )

    etiqueta_offset = (
        "BOA_ADD_OFFSET"
        if nivel == "L2A"
        else "RADIO_ADD_OFFSET"
    )

    offsets = {}

    for elemento in raiz.iter():
        if nombre_local(elemento.tag) != etiqueta_offset:
            continue

        atributos = {
            nombre_local(k): v
            for k, v in elemento.attrib.items()
        }

        id_banda = atributos.get(
            "band_id", atributos.get("bandId")
        )

        if id_banda is not None:
            offsets[int(id_banda)] = float(elemento.text)

    valores_especiales = set()

    for elemento in raiz.iter():
        if nombre_local(elemento.tag) != "Special_Values":
            continue

        hijos = {
            nombre_local(hijo.tag): (hijo.text or "").strip()
            for hijo in elemento
        }

        valor = hijos.get("SPECIAL_VALUE_INDEX")

        if valor:
            valores_especiales.add(int(valor))

    nubosidad = texto.get("Cloud_Coverage_Assessment")

    if nubosidad is None:
        nubosidad = texto.get("CLOUDY_PIXEL_PERCENTAGE")

    return {
        "nivel": nivel,
        "satelite": texto.get("SPACECRAFT_NAME", ""),
        "fecha_adquisicion": texto.get(
            "PRODUCT_START_TIME",
            texto.get("DATATAKE_SENSING_START", ""),
        ),
        "processing_baseline": texto.get(
            "PROCESSING_BASELINE", ""
        ),
        "cuantificacion": cuantificacion,
        "offsets": offsets,
        "valores_especiales": sorted(valores_especiales),
        "nubosidad_global": (
            float(nubosidad) if nubosidad is not None else None
        ),
        "xml_origen": ruta_xml,
    }


def localizar_banda(miembros, banda, nivel):
    sufijo = (
        rf"_{re.escape(banda)}_10m\.jp2$"
        if nivel == "L2A"
        else rf"_{re.escape(banda)}\.jp2$"
    )

    patron = re.compile(sufijo, re.IGNORECASE)

    candidatos = [
        p for p in miembros
        if "/IMG_DATA/" in p and patron.search(p)
    ]

    if len(candidatos) != 1:
        raise ValueError(
            f"{banda}: se esperaba una banda y se encontraron "
            f"{len(candidatos)}. Revisa los granulados."
        )

    return candidatos[0]


def localizar_scl(miembros):
    candidatos = [
        p for p in miembros
        if p.endswith("_SCL_20m.jp2")
    ]

    if len(candidatos) > 1:
        raise ValueError("Se encontraron varias bandas SCL.")

    return candidatos[0] if candidatos else None


# ---------------------------------------------------------------------------
# Ventanas y lectura
# ---------------------------------------------------------------------------

def comprobar_crs_metrico(src):
    if src.crs is None or not src.crs.is_projected:
        raise ValueError(
            "Se necesita un CRS proyectado para recortar en metros."
        )

    unidad = (src.crs.linear_units or "").lower()

    if unidad not in {"metre", "meter", "metres", "meters", "m"}:
        raise ValueError(
            f"El CRS no utiliza metros: {unidad}"
        )

    if (
        abs(src.transform.b) > 1e-9
        or abs(src.transform.d) > 1e-9
    ):
        raise ValueError(
            "Este recorte espera un raster sin rotación."
        )


def coordenadas_objetivo(src, latitud, longitud):
    xs, ys = transformar_coordenadas(
        "EPSG:4326",
        src.crs,
        [longitud],
        [latitud],
    )

    x, y = xs[0], ys[0]
    fila, columna = src.index(x, y)

    if not (
        0 <= fila < src.height
        and 0 <= columna < src.width
    ):
        raise ValueError(
            "El objetivo no está dentro de esta tesela."
        )

    return x, y, fila, columna


def ventana_objetivo(src, latitud, longitud, area_m2):
    """
    Aproxima un recorte cuadrado a píxeles completos.

    El recorte contiene la coordenada del objetivo, pero su
    centro no coincide necesariamente con ella.
    """
    comprobar_crs_metrico(src)

    _, _, fila, columna = coordenadas_objetivo(
        src, latitud, longitud
    )

    lado_m = math.sqrt(area_m2)

    resolucion_x = abs(src.transform.a)
    resolucion_y = abs(src.transform.e)

    columnas = max(
        1,
        int(math.floor(lado_m / resolucion_x + 0.5)),
    )
    filas = max(
        1,
        int(math.floor(lado_m / resolucion_y + 0.5)),
    )

    columnas = min(columnas, src.width)
    filas = min(filas, src.height)

    col_inicio = min(
        max(columna - columnas // 2, 0),
        src.width - columnas,
    )

    fila_inicio = min(
        max(fila - filas // 2, 0),
        src.height - filas,
    )

    return Window(
        col_inicio,
        fila_inicio,
        columnas,
        filas,
    )


def leer_raster(
    zip_path,
    ruta_interna,
    args,
    recortar=False,
    valores_especiales=None,
):
    with rasterio.open(
        ruta_vsizip(zip_path, ruta_interna)
    ) as src:
        ventana = (
            ventana_objetivo(
                src,
                args.latitud,
                args.longitud,
                args.area_m2,
            )
            if recortar else None
        )

        array = src.read(
            1,
            window=ventana,
            masked=True,
        )

        datos = np.asarray(array.data).copy()
        validos = ~np.ma.getmaskarray(array)

        if valores_especiales:
            validos &= ~np.isin(
                datos, valores_especiales
            )

        perfil = src.profile.copy()

        perfil.update(
            height=datos.shape[0],
            width=datos.shape[1],
            transform=(
                src.window_transform(ventana)
                if ventana is not None
                else src.transform
            ),
        )

    return datos, validos, perfil


def comprobar_rejillas(perfil_a, perfil_b):
    coinciden = (
        perfil_a["crs"] == perfil_b["crs"]
        and perfil_a["transform"] == perfil_b["transform"]
        and perfil_a["width"] == perfil_b["width"]
        and perfil_a["height"] == perfil_b["height"]
    )

    if not coinciden:
        raise ValueError(
            "Las bandas no comparten rejilla. "
            "Se necesita un alineado explícito."
        )


# ---------------------------------------------------------------------------
# Escritura y visualización
# ---------------------------------------------------------------------------

def guardar_tif(
    ruta,
    datos,
    perfil,
    validos=None,
    nodata=None,
):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    salida = {
        "driver": "GTiff",
        "height": datos.shape[0],
        "width": datos.shape[1],
        "count": 1,
        "dtype": datos.dtype,
        "crs": perfil["crs"],
        "transform": perfil["transform"],
        "compress": "deflate",
        "nodata": nodata,
    }

    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(ruta, "w", **salida) as dst:
            dst.write(datos, 1)

            if validos is not None:
                dst.write_mask(
                    validos.astype(np.uint8) * 255
                )


def normalizar(datos, validos):
    salida = np.zeros(
        datos.shape, dtype=np.uint8
    )

    utilizables = validos & np.isfinite(datos)
    valores = datos[utilizables]

    if valores.size == 0:
        return salida

    minimo, maximo = np.percentile(
        valores, [2, 98]
    )

    if maximo <= minimo:
        salida[utilizables] = 127
        return salida

    escalado = np.clip(
        (datos.astype(np.float32) - minimo)
        / (maximo - minimo),
        0,
        1,
    )

    salida[utilizables] = (
        escalado[utilizables] * 255
    ).astype(np.uint8)

    return salida


def guardar_rgb(bandas, validos, ruta):
    rgb = np.dstack([
        normalizar(bandas[banda], validos)
        for banda in BANDAS_RGB
    ])

    Image.fromarray(rgb).save(ruta)


def guardar_visual_ndwi(ndwi, ruta):
    figura, eje = plt.subplots()

    imagen = eje.imshow(
        np.ma.masked_invalid(ndwi),
        cmap="RdYlBu",
        vmin=-1,
        vmax=1,
    )

    figura.colorbar(
        imagen, ax=eje, label="NDWI"
    )

    eje.set_title("NDWI")
    eje.axis("off")

    figura.savefig(
        ruta,
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(figura)


# ---------------------------------------------------------------------------
# Nubosidad, reflectancia y NDWI
# ---------------------------------------------------------------------------

def nubosidad_scl(zip_path, ruta_scl, args, recortar):
    scl, validos, _ = leer_raster(
        zip_path,
        ruta_scl,
        args,
        recortar=recortar,
    )

    validos &= scl != CLASE_NODATA
    total = int(validos.sum())

    if total == 0:
        raise ValueError(
            "No hay píxeles válidos en la SCL analizada."
        )

    nubes = (
        np.isin(scl, list(CLASES_NUBE))
        & validos
    )

    return float(nubes.sum() / total * 100)


def convertir_reflectancia(datos, banda, meta):
    q = meta["cuantificacion"]

    if q is None or q <= 0:
        raise ValueError(
            "Falta un factor de cuantificación válido."
        )

    identificador = ID_BANDA_XML[banda]
    offsets = meta["offsets"]

    if offsets and identificador not in offsets:
        raise ValueError(
            f"Falta el offset de {banda}."
        )

    offset = offsets.get(identificador, 0.0)

    return (
        datos.astype(np.float32) + offset
    ) / np.float32(q)


def calcular_ndwi(green, nir, validos):
    denominador = green + nir

    utilizables = (
        validos
        & np.isfinite(green)
        & np.isfinite(nir)
        & (np.abs(denominador) > 1e-8)
    )

    ndwi = np.full(
        green.shape,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        green - nir,
        denominador,
        out=ndwi,
        where=utilizables,
    )

    return ndwi, utilizables


# ---------------------------------------------------------------------------
# Preparación de calibración
# ---------------------------------------------------------------------------

def preparar_calibracion(
    zip_path,
    carpeta,
    miembros,
    meta,
    args,
    resultado,
):
    carpeta.mkdir(parents=True, exist_ok=True)

    perfil_referencia = None
    mascara_comun = None
    bandas_visual = {}
    resumen_bandas = {}

    for banda in BANDAS:
        ruta = localizar_banda(
            miembros, banda, meta["nivel"]
        )

        datos, validos, perfil = leer_raster(
            zip_path,
            ruta,
            args,
            recortar=True,
            valores_especiales=meta["valores_especiales"],
        )

        if perfil_referencia is None:
            perfil_referencia = perfil
        else:
            comprobar_rejillas(
                perfil_referencia, perfil
            )

        guardar_tif(
            carpeta / f"{banda}.tif",
            datos,
            perfil,
            validos=validos,
            nodata=perfil.get("nodata"),
        )

        valores = datos[validos]

        resolucion_x = abs(
            perfil["transform"].a
        )
        resolucion_y = abs(
            perfil["transform"].e
        )

        area_real = (
            datos.shape[0]
            * datos.shape[1]
            * resolucion_x
            * resolucion_y
        )

        resumen_bandas[banda] = {
            "archivo": f"{banda}.tif",
            "ruta_original_zip": ruta,
            "dtype": str(datos.dtype),
            "filas": datos.shape[0],
            "columnas": datos.shape[1],
            "resolucion_x_m": resolucion_x,
            "resolucion_y_m": resolucion_y,
            "area_real_m2": area_real,
            "area_real_km2": area_real / 1_000_000,
            "pixeles_validos": int(validos.sum()),
            "dn_min": (
                float(valores.min())
                if valores.size else None
            ),
            "dn_max": (
                float(valores.max())
                if valores.size else None
            ),
            "dn_media": (
                float(valores.mean())
                if valores.size else None
            ),
        }

        bandas_visual[banda] = datos

        mascara_comun = (
            validos.copy()
            if mascara_comun is None
            else mascara_comun & validos
        )

    if (
        perfil_referencia["width"] > 1
        and perfil_referencia["height"] > 1
    ):
        guardar_rgb(
            bandas_visual,
            mascara_comun,
            carpeta / "rgb_objetivo.jpg",
        )

    documento = {
        **meta,
        "producto": Path(zip_path).stem,
        "tamano_zip_bytes": Path(zip_path).stat().st_size,
        "objetivo_referencia": {
            "latitud": args.latitud,
            "longitud": args.longitud,
            "crs": "EPSG:4326",
        },
        "area_solicitada_m2": args.area_m2,
        "area_solicitada_km2": args.area_m2 / 1_000_000,
        "lado_solicitado_m": math.sqrt(args.area_m2),
        "crs_recortes": str(perfil_referencia["crs"]),
        "bandas": resumen_bandas,
        "nubosidad": resultado["nubosidad"],
        "ambito_nubosidad": resultado["ambito_nubosidad"],
        "scl_disponible": localizar_scl(miembros) is not None,
        "nota": (
            "Valores digitales originales. Recorte ajustado "
            "a píxeles nativos, sin remuestreo. La coordenada "
            "de referencia está dentro del recorte, pero no "
            "necesariamente en su centro. No se ha medido "
            "todavía la posición observada del objetivo."
        ),
    }

    with (carpeta / "metadatos.json").open(
        "w", encoding="utf-8"
    ) as archivo:
        json.dump(
            documento,
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    resultado["observaciones"] = (
        f"Recortes preparados. Área solicitada: "
        f"{args.area_m2} m² "
        f"({args.area_m2 / 1_000_000} km²). "
        "Consultar el área real en metadatos.json."
    )


# ---------------------------------------------------------------------------
# Procesamiento de costa
# ---------------------------------------------------------------------------

def procesar_costa(
    zip_path,
    carpeta,
    miembros,
    meta,
    args,
    carpeta_waterbody,
    resultado,
):
    if meta["nivel"] != "L2A":
        resultado["estado"] = "solo_registro"
        resultado["observaciones"] = (
            "L1C registrado. Utiliza --modo calibracion "
            "para exportar sus bandas."
        )
        return

    carpeta.mkdir(parents=True, exist_ok=True)

    bandas = {}
    mascara_comun = None
    perfil_referencia = None

    for banda in BANDAS:
        ruta = localizar_banda(
            miembros, banda, meta["nivel"]
        )

        datos, validos, perfil = leer_raster(
            zip_path,
            ruta,
            args,
            valores_especiales=meta["valores_especiales"],
        )

        if perfil_referencia is None:
            perfil_referencia = perfil
        else:
            comprobar_rejillas(
                perfil_referencia, perfil
            )

        bandas[banda] = convertir_reflectancia(
            datos, banda, meta
        )

        mascara_comun = (
            validos.copy()
            if mascara_comun is None
            else mascara_comun & validos
        )

    base = Path(zip_path).stem

    guardar_rgb(
        bandas,
        mascara_comun,
        carpeta / f"{base}_costa.jpg",
    )

    ndwi, validos_ndwi = calcular_ndwi(
        bandas["B03"],
        bandas["B08"],
        mascara_comun,
    )

    guardar_tif(
        carpeta / f"{base}_ndwi.tif",
        ndwi,
        perfil_referencia,
        validos=validos_ndwi,
        nodata=np.nan,
    )

    guardar_visual_ndwi(
        ndwi,
        carpeta / f"{base}_ndwi.jpg",
    )

    waterbody = (
        validos_ndwi
        & (ndwi > args.umbral_ndwi)
    ).astype(np.uint8)

    ruta_waterbody = (
        carpeta_waterbody
        / f"{base}_waterbody.tif"
    )

    guardar_tif(
        ruta_waterbody,
        waterbody,
        perfil_referencia,
        validos=validos_ndwi,
        nodata=None,
    )

    total = int(validos_ndwi.sum())

    resultado["archivo_waterbody"] = str(
        ruta_waterbody
    )

    resultado["porcentaje_agua"] = (
        float(waterbody.sum() / total * 100)
        if total else None
    )

    resultado["observaciones"] = (
        "RGB, NDWI y máscara 0/1 generados. "
        "La máscara interna identifica píxeles sin datos. "
        "Las nubes no se eliminan individualmente del NDWI."
    )


# ---------------------------------------------------------------------------
# Orquestación y registro
# ---------------------------------------------------------------------------

def procesar_zip(
    zip_path,
    args,
    salida,
    carpeta_waterbody=None,
):
    resultado = {
        campo: None for campo in CAMPOS_LOG
    }

    resultado.update({
        "zip": str(zip_path),
        "fecha_proceso": datetime.now().isoformat(
            timespec="seconds"
        ),
        "modo": args.modo,
        "estado": "iniciado",
    })

    try:
        miembros = listar_zip(zip_path)
        nivel = detectar_nivel(zip_path, miembros)
        ruta_xml = localizar_xml(miembros, nivel)

        meta = leer_metadatos(
            zip_path, ruta_xml, nivel
        )

        resultado.update({
            "nivel": nivel,
            "satelite": meta["satelite"],
            "fecha_adquisicion": meta["fecha_adquisicion"],
        })

        ruta_scl = localizar_scl(miembros)

        if nivel == "L2A" and ruta_scl:
            recortar = args.modo == "calibracion"

            porcentaje = nubosidad_scl(
                zip_path,
                ruta_scl,
                args,
                recortar,
            )

            ambito = (
                "vecindad_objetivo_SCL_20m"
                if recortar
                else "tesela_SCL"
            )
        else:
            porcentaje = meta["nubosidad_global"]
            ambito = "producto_XML"

        resultado["nubosidad"] = porcentaje
        resultado["ambito_nubosidad"] = ambito

        resultado["paso_filtro"] = (
            porcentaje <= args.nubosidad
            if porcentaje is not None
            else None
        )

        aplicar_filtro = (
            args.modo == "costa"
            or args.filtrar_calibracion
        )

        if aplicar_filtro:
            if porcentaje is None:
                raise ValueError(
                    "No hay nubosidad disponible "
                    "para aplicar el filtro."
                )

            if porcentaje > args.nubosidad:
                resultado["estado"] = "descartado_nubosidad"
                resultado["observaciones"] = (
                    "No se generan salidas por nubosidad."
                )
                return resultado

        carpeta = salida / Path(zip_path).stem

        resultado["carpeta_salida"] = str(carpeta)

        if args.modo == "calibracion":
            preparar_calibracion(
                zip_path,
                carpeta,
                miembros,
                meta,
                args,
                resultado,
            )
        else:
            procesar_costa(
                zip_path,
                carpeta,
                miembros,
                meta,
                args,
                carpeta_waterbody,
                resultado,
            )

        if resultado["estado"] == "iniciado":
            resultado["estado"] = "correcto"

    except Exception as error:
        resultado["estado"] = "error"
        resultado["error"] = str(error)

    return resultado


def escribir_registro(ruta, fila):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    tiene_contenido = (
        ruta.exists()
        and ruta.stat().st_size > 0
    )

    if tiene_contenido:
        with ruta.open(
            newline="", encoding="utf-8"
        ) as archivo:
            cabecera = next(
                csv.reader(archivo), []
            )

        if cabecera != CAMPOS_LOG:
            raise ValueError(
                f"El CSV {ruta} tiene otro formato. "
                "Indica una ruta nueva con --csv."
            )

    with ruta.open(
        "a", newline="", encoding="utf-8"
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=CAMPOS_LOG,
        )

        if not tiene_contenido:
            escritor.writeheader()

        escritor.writerow(fila)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Procesamiento de costa y preparación "
            "de bandas para calibración."
        )
    )

    ap.add_argument(
        "--modo",
        choices=["costa", "calibracion"],
        default="costa",
    )
    ap.add_argument("--entrada")
    ap.add_argument("--salida")
    ap.add_argument("--csv")

    ap.add_argument(
        "--proyecto",
        default=NOMBRE_PROYECTO,
    )
    ap.add_argument(
        "--nubosidad",
        type=float,
        default=UMBRAL_NUBOSIDAD,
    )
    ap.add_argument(
        "--umbral-ndwi",
        type=float,
        default=UMBRAL_NDWI_AGUA,
    )
    ap.add_argument(
        "--latitud",
        type=float,
        default=SHADNAGAR_LAT,
    )
    ap.add_argument(
        "--longitud",
        type=float,
        default=SHADNAGAR_LON,
    )
    ap.add_argument(
        "--area-m2",
        type=float,
        default=AREA_RECORTE_M2,
        help=(
            "Área solicitada del recorte en m². "
            "Predeterminado: 100000 m² = 0,1 km²."
        ),
    )
    ap.add_argument(
        "--filtrar-calibracion",
        action="store_true",
    )
    ap.add_argument(
        "--mover-procesados",
        action="store_true",
    )
    ap.add_argument(
        "--procesados",
        help="Destino opcional de los ZIP procesados.",
    )

    args = ap.parse_args()

    if not 0 <= args.nubosidad <= 100:
        ap.error(
            "--nubosidad debe estar entre 0 y 100."
        )

    if (
        not math.isfinite(args.area_m2)
        or args.area_m2 <= 0
    ):
        ap.error(
            "--area-m2 debe ser un número positivo y finito."
        )

    if not -90 <= args.latitud <= 90:
        ap.error("Latitud no válida.")

    if not -180 <= args.longitud <= 180:
        ap.error("Longitud no válida.")

    entrada = Path(
        args.entrada
        or (
            ENTRADA_CALIBRACION
            if args.modo == "calibracion"
            else CARPETA_ENTRADA
        )
    )

    salida = Path(
        args.salida
        or (
            SALIDA_CALIBRACION
            if args.modo == "calibracion"
            else CARPETA_SALIDA
        )
    )

    csv_log = Path(
        args.csv
        or salida / f"registro_{args.modo}.csv"
    )

    procesados = Path(
        args.procesados
        or (
            "Imagenes/calibracion/zips_procesados"
            if args.modo == "calibracion"
            else CARPETA_PROCESADOS
        )
    )

    if not entrada.is_dir():
        raise SystemExit(
            f"No existe la entrada: {entrada}"
        )

    if args.mover_procesados:
        if (
            procesados.resolve() == entrada.resolve()
            or entrada.resolve()
            in procesados.resolve().parents
        ):
            ap.error(
                "--procesados debe quedar fuera "
                "de la carpeta de entrada."
            )

    carpeta_waterbody = None

    if args.modo == "costa":
        import shoreline_utils as utils

        carpeta_waterbody = (
            Path(utils.project_path(args.proyecto))
            / "output"
            / "estimated_waterbodies_images"
        )

        carpeta_waterbody.mkdir(
            parents=True, exist_ok=True
        )

    salida.mkdir(parents=True, exist_ok=True)

    candidatos = (
        entrada.rglob("*")
        if args.modo == "calibracion"
        else entrada.iterdir()
    )

    zips = sorted(
        p for p in candidatos
        if p.is_file()
        and p.suffix.lower() == ".zip"
    )

    if not zips:
        print(
            f"No se encontraron ZIP en {entrada}"
        )
        return

    if args.modo == "calibracion":
        print(
            f"Área solicitada: {args.area_m2:.2f} m² "
            f"({args.area_m2 / 1_000_000:.6f} km²)."
        )
        print(
            f"Lado equivalente: "
            f"{math.sqrt(args.area_m2):.2f} m."
        )
        print(
            "El área real se ajustará a píxeles nativos "
            "y quedará registrada en metadatos.json."
        )

    errores = 0

    for zip_path in zips:
        print(
            f"\nProcesando {zip_path.name}..."
        )

        resultado = procesar_zip(
            zip_path,
            args,
            salida,
            carpeta_waterbody,
        )

        if resultado["estado"] == "error":
            errores += 1
            print(
                f"  ERROR: {resultado['error']}"
            )
        else:
            print(
                f"  Estado: {resultado['estado']} | "
                f"Nivel: {resultado['nivel']} | "
                f"Nubosidad: {resultado['nubosidad']} | "
                f"Ámbito: {resultado['ambito_nubosidad']}"
            )

            if resultado["observaciones"]:
                print(
                    f"  {resultado['observaciones']}"
                )

        escribir_registro(
            csv_log, resultado
        )

        if (
            args.mover_procesados
            and resultado["estado"] == "correcto"
        ):
            procesados.mkdir(
                parents=True, exist_ok=True
            )

            destino = procesados / zip_path.name

            if destino.exists():
                print(
                    "  No se mueve: ya existe "
                    "un ZIP con ese nombre."
                )
            else:
                try:
                    shutil.move(
                        str(zip_path),
                        str(destino),
                    )
                except OSError as error:
                    print(
                        "  Procesamiento correcto, "
                        "pero no se pudo mover "
                        f"el ZIP: {error}"
                    )

    print(f"\nRegistro: {csv_log}")
    print(
        f"Productos con error: {errores}"
    )

    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
