"""
Descarga de productos Sentinel-2 desde Copernicus Data Space Ecosystem.

Requiere:
    pip install requests

Compatibilidad con el flujo anterior:
    python descarga_copernicus.py --inicio 2026-06-01 --fin 2026-09-30

Calibración en Shadnagar:
    python descarga_copernicus.py --zona shadnagar \
        --inicio 2024-01-01 --fin 2026-11-20 \
        --nivel AMBOS --num-fechas 10 --exigir-todos \
        --nubosidad 10 --solo-listar

Quitar --solo-listar para descargar.

Importante:
- El filtro de nubosidad corresponde a toda la escena.
- La selección automática no garantiza que el objetivo esté despejado.
- El área de búsqueda de Shadnagar NO es el recorte final de análisis.
- Este archivo descarga datos; no calcula RMSE ni aplica calibraciones.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Las variables de entorno tienen prioridad.
# No publicar este archivo con credenciales reales.
CDSE_USER = "antoni.ruiz.perez@estudiantat.upc.edu"
CDSE_PASSWORD = "00Copernicus*"

GEOJSON_PATH = "projects/costa_principal/input/polygon.geojson"
CARPETA_ENTRADA = "Imagenes/zips_entrada"
CARPETA_PROCESADOS = "Imagenes/zips_procesados"
CARPETA_CALIBRACION = "Imagenes/calibracion"

NUBOSIDAD_MAX_ESCENA = 40.0

SHADNAGAR_LAT = 17.034261
SHADNAGAR_LON = 78.183078

URL_TOKEN = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
URL_CATALOGO = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
URL_DESCARGA = (
    "https://download.dataspace.copernicus.eu/"
    "odata/v1/Products({id})/$value"
)

TIPOS = {"L1C": "S2MSI1C", "L2A": "S2MSI2A"}

PATRON_NOMBRE = re.compile(
    r"^(S2[ABC])_MSIL(1C|2A)_(\d{8}T\d{6})_"
    r"N(\d+)_R(\d+)_T([A-Z0-9]{5})_(\d{8}T\d{6})"
)


# ---------------------------------------------------------------------------
# Área de búsqueda
# ---------------------------------------------------------------------------

def geojson_a_wkt(ruta):
    """Lee un Polygon desde una geometría, Feature o FeatureCollection."""
    with open(ruta, encoding="utf-8") as f:
        contenido = json.load(f)

    tipo = contenido.get("type")

    if tipo == "FeatureCollection":
        features = contenido.get("features", [])
        if not features:
            raise ValueError("El GeoJSON no contiene elementos.")
        geometria = features[0]["geometry"]
    elif tipo == "Feature":
        geometria = contenido["geometry"]
    else:
        geometria = contenido

    if not geometria or geometria.get("type") != "Polygon":
        raise ValueError("El GeoJSON debe contener un Polygon.")

    anillos = []
    for anillo in geometria["coordinates"]:
        puntos = [tuple(p[:2]) for p in anillo]
        if len(puntos) < 3:
            raise ValueError("El polígono tiene un anillo no válido.")
        if puntos[0] != puntos[-1]:
            puntos.append(puntos[0])

        texto = ", ".join(
            f"{lon:.8f} {lat:.8f}" for lon, lat in puntos
        )
        anillos.append(f"({texto})")

    return f"POLYGON({', '.join(anillos)})"


def wkt_shadnagar(semilado_m=500):
    """
    Caja aproximada para buscar escenas alrededor del objetivo.
    No sustituye al recorte posterior en coordenadas proyectadas.
    """
    delta_lat = semilado_m / 111320.0
    delta_lon = semilado_m / (
        111320.0 * math.cos(math.radians(SHADNAGAR_LAT))
    )

    oeste = SHADNAGAR_LON - delta_lon
    este = SHADNAGAR_LON + delta_lon
    sur = SHADNAGAR_LAT - delta_lat
    norte = SHADNAGAR_LAT + delta_lat

    return (
        f"POLYGON(({oeste:.8f} {sur:.8f}, "
        f"{este:.8f} {sur:.8f}, "
        f"{este:.8f} {norte:.8f}, "
        f"{oeste:.8f} {norte:.8f}, "
        f"{oeste:.8f} {sur:.8f}))"
    )


# ---------------------------------------------------------------------------
# Catálogo y metadatos
# ---------------------------------------------------------------------------

def buscar_productos(wkt, inicio, fin, nubosidad_max, nivel="L2A"):
    tipo = TIPOS[nivel]

    filtro = (
        "Collection/Name eq 'SENTINEL-2' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') "
        f"and ContentDate/Start ge {inicio}T00:00:00.000Z "
        f"and ContentDate/Start le {fin}T23:59:59.999Z "
        "and Attributes/OData.CSC.StringAttribute/any(a:"
        "a/Name eq 'productType' "
        f"and a/OData.CSC.StringAttribute/Value eq '{tipo}') "
        "and Attributes/OData.CSC.DoubleAttribute/any(a:"
        "a/Name eq 'cloudCover' "
        f"and a/OData.CSC.DoubleAttribute/Value le {nubosidad_max})"
    )

    params = {
        "$filter": filtro,
        "$orderby": "ContentDate/Start asc",
        "$expand": "Attributes",
        "$top": 100,
    }

    productos = []
    url = URL_CATALOGO

    while url:
        respuesta = requests.get(url, params=params, timeout=60)
        respuesta.raise_for_status()
        datos = respuesta.json()
        productos.extend(datos["value"])
        url = datos.get("@odata.nextLink")
        params = None

    return productos


def metadatos(producto):
    nombre = producto["Name"]
    coincidencia = PATRON_NOMBRE.match(nombre)

    if not coincidencia:
        raise ValueError(
            f"Nombre Sentinel-2 no reconocido: {nombre}"
        )

    satelite, nivel, instante, baseline, orbita, tesela, generado = (
        coincidencia.groups()
    )

    nubes = None
    for atributo in producto.get("Attributes", []):
        if atributo.get("Name") == "cloudCover":
            nubes = float(atributo["Value"])
            break

    return {
        "satelite": satelite,
        "nivel": f"L{nivel}",
        "instante": instante,
        "fecha": (
            f"{instante[:4]}-{instante[4:6]}-{instante[6:8]}"
        ),
        "baseline": int(baseline),
        "orbita": orbita,
        "tesela": tesela,
        "generado": generado,
        "nubosidad": nubes,
    }


def clave_escena(producto):
    m = metadatos(producto)
    return (
        m["satelite"],
        m["instante"],
        m["orbita"],
        m["tesela"],
    )


def deduplicar(productos):
    """Conserva una versión por escena y nivel."""
    elegidos = {}

    for producto in productos:
        m = metadatos(producto)
        clave = (clave_escena(producto), m["nivel"])
        anterior = elegidos.get(clave)

        if anterior is None:
            elegidos[clave] = producto
            continue

        ma = metadatos(anterior)
        if (m["baseline"], m["generado"]) > (
            ma["baseline"], ma["generado"]
        ):
            elegidos[clave] = producto

    return list(elegidos.values())


def crear_grupos(productos, exigir_parejas=False):
    """Cada grupo contiene una escena y uno o dos niveles."""
    grupos = defaultdict(list)

    for producto in productos:
        grupos[clave_escena(producto)].append(producto)

    resultado = []
    for grupo in grupos.values():
        niveles = {metadatos(p)["nivel"] for p in grupo}
        if exigir_parejas and niveles != {"L1C", "L2A"}:
            continue
        resultado.append(grupo)

    return resultado


def calidad_grupo(grupo):
    nubes = [
        metadatos(p)["nubosidad"]
        for p in grupo
        if metadatos(p)["nubosidad"] is not None
    ]
    return (
        max(nubes) if nubes else float("inf"),
        metadatos(grupo[0])["instante"],
        metadatos(grupo[0])["tesela"],
    )


def seleccionar_fechas(grupos, numero, satelites, exigir_todos):
    """
    Selecciona una escena por día, priorizando baja nubosidad.
    Si se exige cobertura de satélites, la garantiza primero.
    """
    grupos = [
        g for g in grupos
        if metadatos(g[0])["satelite"] in satelites
    ]

    if numero is None:
        presentes = {metadatos(g[0])["satelite"] for g in grupos}
        if exigir_todos and not set(satelites).issubset(presentes):
            faltan = set(satelites) - presentes
            raise ValueError(f"Faltan satélites: {sorted(faltan)}")
        return sorted(
            grupos, key=lambda g: metadatos(g[0])["instante"]
        )

    if exigir_todos and numero < len(satelites):
        raise ValueError(
            "--num-fechas debe permitir al menos una fecha por satélite."
        )

    grupos = sorted(grupos, key=calidad_grupo)
    seleccion = []
    fechas_usadas = set()

    if exigir_todos:
        candidatos = {
            sat: [
                g for g in grupos
                if metadatos(g[0])["satelite"] == sat
            ]
            for sat in satelites
        }

        # Backtracking para garantizar días distintos entre satélites.
        orden = sorted(satelites, key=lambda s: len(candidatos[s]))

        def asignar(indice):
            if indice == len(orden):
                return True

            for grupo in candidatos[orden[indice]]:
                fecha = metadatos(grupo[0])["fecha"]
                if fecha in fechas_usadas:
                    continue

                seleccion.append(grupo)
                fechas_usadas.add(fecha)

                if asignar(indice + 1):
                    return True

                seleccion.pop()
                fechas_usadas.remove(fecha)

            return False

        if not asignar(0):
            raise ValueError(
                "No hay combinación de fechas distintas que incluya "
                "todos los satélites solicitados. Amplía el intervalo "
                "o revisa el filtro de nubosidad."
            )

    for grupo in grupos:
        if len(seleccion) >= numero:
            break

        fecha = metadatos(grupo[0])["fecha"]
        if fecha not in fechas_usadas:
            seleccion.append(grupo)
            fechas_usadas.add(fecha)

    if len(seleccion) < numero:
        raise ValueError(
            f"Se solicitaron {numero} fechas distintas, pero solo "
            f"se pueden seleccionar {len(seleccion)}."
        )

    return sorted(
        seleccion, key=lambda g: metadatos(g[0])["instante"]
    )


# ---------------------------------------------------------------------------
# Autenticación y descarga
# ---------------------------------------------------------------------------

def obtener_token(usuario, password):
    respuesta = requests.post(
        URL_TOKEN,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": usuario,
            "password": password,
        },
        timeout=60,
    )
    respuesta.raise_for_status()
    return respuesta.json()["access_token"]


def descargar_producto(producto, destino, usuario, password):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = Path(str(destino) + ".part")

    token = obtener_token(usuario, password)
    url = URL_DESCARGA.format(id=producto["Id"])

    with requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=(30, 180),
    ) as respuesta:
        respuesta.raise_for_status()
        recibidos = 0

        with parcial.open("wb") as archivo:
            for bloque in respuesta.iter_content(
                chunk_size=8 * 1024 * 1024
            ):
                if bloque:
                    archivo.write(bloque)
                    recibidos += len(bloque)

    if recibidos == 0:
        raise IOError("La descarga está vacía.")

    esperado = int(producto.get("ContentLength") or 0)
    if esperado and recibidos != esperado:
        raise IOError(
            f"Tamaño inesperado: {recibidos} bytes; "
            f"el catálogo indica {esperado}. Se conserva el .part."
        )

    os.replace(parcial, destino)


# ---------------------------------------------------------------------------
# Manifiesto
# ---------------------------------------------------------------------------

CAMPOS_CSV = [
    "id", "nombre", "fecha", "instante", "satelite", "nivel",
    "tesela", "orbita", "baseline", "nubosidad",
    "bytes_catalogo", "ruta", "estado", "error",
]


def guardar_manifiesto(filas, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS_CSV)
        escritor.writeheader()
        escritor.writerows(filas)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Descarga Sentinel-2 para costa y calibración."
    )
    ap.add_argument("--inicio", required=True, help="YYYY-MM-DD")
    ap.add_argument(
        "--fin", default=date.today().isoformat(), help="YYYY-MM-DD"
    )
    ap.add_argument(
        "--zona", choices=["costa", "shadnagar"], default="costa"
    )
    ap.add_argument(
        "--geojson",
        help="GeoJSON propio; tiene prioridad sobre --zona."
    )
    ap.add_argument(
        "--nivel", choices=["L1C", "L2A", "AMBOS"], default="L2A"
    )
    ap.add_argument(
        "--nubosidad", type=float, default=NUBOSIDAD_MAX_ESCENA
    )
    ap.add_argument(
        "--satelites", nargs="+",
        choices=["S2A", "S2B", "S2C"],
        default=["S2A", "S2B", "S2C"],
    )
    ap.add_argument(
        "--num-fechas", type=int,
        help="Seleccionar exactamente N días distintos."
    )
    ap.add_argument(
        "--exigir-todos", action="store_true",
        help="Exigir presencia de todos los satélites solicitados."
    )
    ap.add_argument("--destino", help="Carpeta de salida personalizada.")
    ap.add_argument("--manifiesto", help="Ruta del manifiesto CSV.")
    ap.add_argument("--solo-listar", action="store_true")
    args = ap.parse_args()

    try:
        inicio = date.fromisoformat(args.inicio)
        fin = date.fromisoformat(args.fin)
    except ValueError:
        ap.error("Las fechas deben tener formato YYYY-MM-DD.")

    if inicio > fin:
        ap.error("--inicio no puede ser posterior a --fin.")
    if not 0 <= args.nubosidad <= 100:
        ap.error("--nubosidad debe estar entre 0 y 100.")
    if args.num_fechas is not None and args.num_fechas <= 0:
        ap.error("--num-fechas debe ser mayor que cero.")

    args.satelites = list(dict.fromkeys(args.satelites))

    usuario = os.environ.get("CDSE_USER", CDSE_USER)
    password = os.environ.get("CDSE_PASSWORD", CDSE_PASSWORD)

    if not args.solo_listar and not (usuario and password):
        ap.error("Faltan las credenciales de CDSE.")

    if args.geojson:
        wkt = geojson_a_wkt(args.geojson)
    elif args.zona == "shadnagar":
        wkt = wkt_shadnagar()
    else:
        wkt = geojson_a_wkt(GEOJSON_PATH)

    # Solo el modo costa/L2A original conserva la salida antigua.
    modo_original = (
        args.zona == "costa"
        and args.geojson is None
        and args.nivel == "L2A"
        and args.destino is None
    )

    if args.destino:
        carpeta = Path(args.destino)
    elif modo_original:
        carpeta = Path(CARPETA_ENTRADA)
    else:
        carpeta = Path(CARPETA_CALIBRACION) / args.zona

    niveles = (
        ["L1C", "L2A"] if args.nivel == "AMBOS"
        else [args.nivel]
    )

    productos = []
    for nivel in niveles:
        encontrados = buscar_productos(
            wkt, args.inicio, args.fin, args.nubosidad, nivel
        )
        print(f"{nivel}: {len(encontrados)} productos encontrados.")
        productos.extend(encontrados)

    # No seleccionar productos que no están disponibles para descarga.
    productos = [
        p for p in productos if p.get("Online", True)
    ]
    productos = deduplicar(productos)

    grupos = crear_grupos(
        productos, exigir_parejas=args.nivel == "AMBOS"
    )

    try:
        grupos = seleccionar_fechas(
            grupos,
            args.num_fechas,
            args.satelites,
            args.exigir_todos,
        )
    except ValueError as error:
        raise SystemExit(str(error))

    seleccionados = [
        p
        for grupo in grupos
        for p in sorted(
            grupo, key=lambda p: metadatos(p)["nivel"]
        )
    ]

    if not seleccionados:
        raise SystemExit(
            "No hay productos que cumplan los criterios solicitados."
        )

    fechas = {metadatos(p)["fecha"] for p in seleccionados}
    sats = {metadatos(p)["satelite"] for p in seleccionados}

    print(
        f"\nSelección: {len(seleccionados)} productos, "
        f"{len(fechas)} fechas distintas."
    )
    print(f"Satélites: {', '.join(sorted(sats))}")
    print(
        "Aviso: revisar visualmente la nubosidad sobre el objetivo "
        "antes de dar por válidas las fechas."
    )

    ruta_manifiesto = (
        Path(args.manifiesto)
        if args.manifiesto
        else carpeta / (
            f"manifiesto_{args.nivel}_{args.inicio}_{args.fin}.csv"
        )
    )

    filas = []

    for producto in seleccionados:
        m = metadatos(producto)
        nombre_zip = (
            producto["Name"].removesuffix(".SAFE") + ".zip"
        )

        destino = (
            carpeta / nombre_zip
            if modo_original
            else carpeta / m["nivel"] / nombre_zip
        )

        fila = {
            "id": producto["Id"],
            "nombre": producto["Name"],
            **{
                campo: m[campo]
                for campo in [
                    "fecha", "instante", "satelite", "nivel",
                    "tesela", "orbita", "baseline", "nubosidad",
                ]
            },
            "bytes_catalogo": producto.get("ContentLength", ""),
            "ruta": str(destino),
            "estado": "seleccionado",
            "error": "",
        }
        filas.append(fila)

        print(
            f"\n{m['fecha']} | {m['satelite']} | "
            f"{m['nivel']} | T{m['tesela']} | "
            f"nubes: {m['nubosidad']}"
        )
        print(f"  {nombre_zip}")

        procesado = Path(CARPETA_PROCESADOS) / nombre_zip

        if destino.exists():
            fila["estado"] = "ya_existe"
            print("  Ya existe; se omite.")

        elif modo_original and procesado.exists():
            fila["ruta"] = str(procesado)
            fila["estado"] = "ya_procesado"
            print("  Ya procesado; se omite.")

        elif args.solo_listar:
            fila["estado"] = "solo_listado"
            print("  Solo listado; no se descarga.")

        else:
            try:
                descargar_producto(
                    producto, destino, usuario, password
                )
                fila["estado"] = "descargado"
                print("  Descarga completada.")
            except Exception as error:
                fila["estado"] = "error"
                fila["error"] = str(error)
                print(f"  ERROR: {error}")

        guardar_manifiesto(filas, ruta_manifiesto)

    errores = sum(f["estado"] == "error" for f in filas)

    print(f"\nManifiesto: {ruta_manifiesto}")
    print(f"Productos con error: {errores}")

    if modo_original and not args.solo_listar:
        print("Puedes continuar con sesion5.py.")
    else:
        print(
            "Las imágenes de calibración deben procesarse "
            "en su flujo específico."
        )

    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
