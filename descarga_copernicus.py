"""
Descarga de imágenes Sentinel-2 L2A desde Copernicus Data Space Ecosystem
(CDSE) para el área definida en un GeoJSON.

Los .zip se guardan en CARPETA_ENTRADA, listos para sesion5.py.

Requiere: requests
    pip install requests

Credenciales (cuenta gratuita en https://dataspace.copernicus.eu):
    export CDSE_USER="tu_email"
    export CDSE_PASSWORD="tu_password"

Uso:
    python descarga_copernicus.py --inicio 2026-06-01 --fin 2026-09-30
"""

import os
import json
import argparse
from datetime import date

import requests

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Credenciales de CDSE. Si existen las variables de entorno CDSE_USER /
# CDSE_PASSWORD, tienen prioridad sobre estos valores.
CDSE_USER = "antoni.ruiz.perez@estudiantat.upc.edu"
CDSE_PASSWORD = "00Copernicus*"

GEOJSON_PATH = "polygon.geojson"
CARPETA_ENTRADA = "Imagenes/zips-entrada"
CARPETA_PROCESADOS = "Imagenes/zips_procesados"

# Filtro previo por nubosidad de TODA la escena (100x100 km). Es más laxo
# que el de sesion5.py (que mide solo la SCL) para no descartar escenas
# cuya costa sí está despejada, pero evita bajar escenas totalmente nubladas.
NUBOSIDAD_MAX_ESCENA = 40.0

URL_TOKEN = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
URL_CATALOGO = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
URL_DESCARGA = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"


# ---------------------------------------------------------------------------
# GeoJSON -> WKT
# ---------------------------------------------------------------------------

def geojson_a_wkt(ruta):
    """Convierte el primer Polygon del GeoJSON a WKT (lon lat), que es lo que pide el catálogo."""
    with open(ruta, encoding="utf-8") as f:
        gj = json.load(f)

    geom = gj["features"][0]["geometry"] if gj["type"] == "FeatureCollection" else gj
    if geom["type"] != "Polygon":
        raise ValueError(f"Se esperaba un Polygon, no {geom['type']}")

    anillo = geom["coordinates"][0]
    puntos = ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat, *_ in anillo)
    return f"POLYGON(({puntos}))"


# ---------------------------------------------------------------------------
# Búsqueda en el catálogo
# ---------------------------------------------------------------------------

def buscar_productos(wkt, fecha_inicio, fecha_fin, nubosidad_max):
    """Devuelve la lista de productos S2 L2A que intersectan el polígono."""
    filtro = (
        "Collection/Name eq 'SENTINEL-2' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') "
        f"and ContentDate/Start ge {fecha_inicio}T00:00:00.000Z "
        f"and ContentDate/Start le {fecha_fin}T23:59:59.999Z "
        "and Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' "
        "and a/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        "and Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' "
        f"and a/OData.CSC.DoubleAttribute/Value le {nubosidad_max})"
    )
    params = {"$filter": filtro, "$orderby": "ContentDate/Start asc", "$top": 100}

    productos = []
    url = URL_CATALOGO
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        datos = r.json()
        productos.extend(datos["value"])
        url = datos.get("@odata.nextLink")  # paginación
        params = None  # nextLink ya trae los parámetros
    return productos


# ---------------------------------------------------------------------------
# Autenticación y descarga
# ---------------------------------------------------------------------------

def obtener_token(usuario, password):
    """Token de acceso (válido ~10 min, por eso se pide uno por descarga)."""
    r = requests.post(
        URL_TOKEN,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": usuario,
            "password": password,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def descargar_producto(producto, destino_zip, usuario, password):
    """Descarga un producto en streaming; escribe a .part y renombra al terminar."""
    token = obtener_token(usuario, password)
    url = URL_DESCARGA.format(id=producto["Id"])
    parcial = destino_zip + ".part"

    with requests.get(
        url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=120
    ) as r:
        r.raise_for_status()
        with open(parcial, "wb") as f:
            for trozo in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(trozo)

    os.rename(parcial, destino_zip)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", required=True, help="YYYY-MM-DD")
    ap.add_argument("--fin", default=date.today().isoformat(), help="YYYY-MM-DD")
    ap.add_argument("--nubosidad", type=float, default=NUBOSIDAD_MAX_ESCENA)
    ap.add_argument("--solo-listar", action="store_true", help="No descarga, solo muestra")
    args = ap.parse_args()

    usuario = os.environ.get("CDSE_USER", CDSE_USER)
    password = os.environ.get("CDSE_PASSWORD", CDSE_PASSWORD)
    if not args.solo_listar and not (usuario and password):
        raise SystemExit("Faltan las credenciales de CDSE")

    os.makedirs(CARPETA_ENTRADA, exist_ok=True)
    os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

    wkt = geojson_a_wkt(GEOJSON_PATH)
    productos = buscar_productos(wkt, args.inicio, args.fin, args.nubosidad)
    print(f"{len(productos)} productos encontrados.")

    for p in productos:
        nombre_zip = p["Name"].replace(".SAFE", "") + ".zip"
        ruta_entrada = os.path.join(CARPETA_ENTRADA, nombre_zip)
        ruta_procesado = os.path.join(CARPETA_PROCESADOS, nombre_zip)

        # Evita bajar de nuevo lo que ya está pendiente o ya procesado
        if os.path.exists(ruta_entrada) or os.path.exists(ruta_procesado):
            print(f"  Ya existe, se omite: {nombre_zip}")
            continue

        if not p.get("Online", True):
            print(f"  Archivado (offline), se omite: {nombre_zip}")
            continue

        tam_mb = p.get("ContentLength", 0) / 1e6
        print(f"  {nombre_zip}  ({tam_mb:.0f} MB)")
        if args.solo_listar:
            continue

        try:
            descargar_producto(p, ruta_entrada, usuario, password)
        except Exception as e:
            print(f"    ERROR descargando: {e}")

    print("Descarga terminada. Ahora puedes ejecutar sesion5.py")


if __name__ == "__main__":
    main()