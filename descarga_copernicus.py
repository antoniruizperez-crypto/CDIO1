"""
Descarga de imágenes Sentinel-2 L2A desde Copernicus Data Space Ecosystem (CDSE)
para el área definida en el GeoJSON del proyecto.

Los .zip se guardan en Imagenes/zips_entrada, listos para sesion5.py.

Credenciales (cuenta gratuita en https://dataspace.copernicus.eu). NUNCA van
en el código: se leen de variables de entorno.
    Linux / macOS:   export CDSE_USER="tu_email"; export CDSE_PASSWORD="tu_password"
    Windows (PS):    $env:CDSE_USER="tu_email"; $env:CDSE_PASSWORD="tu_password"

Uso:
    python descarga_copernicus.py --inicio 2026-06-01 --fin 2026-09-30
    python descarga_copernicus.py --inicio 2026-06-01 --solo-listar
"""

import argparse
import json
import os
from datetime import date

import requests

import config

URL_TOKEN = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
URL_CATALOGO = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
URL_DESCARGA = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"
REDIRECCIONES = (301, 302, 303, 307, 308)


def geojson_a_wkt(ruta):
    """Primer Polygon del GeoJSON -> WKT (lon lat), formato que pide el catálogo."""
    with open(ruta, encoding="utf-8") as f:
        gj = json.load(f)
    geom = gj["features"][0]["geometry"] if gj["type"] == "FeatureCollection" else gj
    if geom["type"] != "Polygon":
        raise ValueError(f"Se esperaba un Polygon, no {geom['type']}")
    puntos = ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat, *_ in geom["coordinates"][0])
    return f"POLYGON(({puntos}))"


def buscar_productos(wkt, fecha_inicio, fecha_fin, nubosidad_max):
    """Productos S2 L2A que intersectan el polígono, con nubosidad de escena <= umbral."""
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
    productos, url = [], URL_CATALOGO
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        datos = r.json()
        productos.extend(datos["value"])
        url = datos.get("@odata.nextLink")
        params = None  # nextLink ya incluye los parámetros
    return productos


def obtener_token(usuario, password):
    """Token de acceso (~10 min de validez, por eso se pide uno por descarga)."""
    r = requests.post(
        URL_TOKEN,
        data={"client_id": "cdse-public", "grant_type": "password",
              "username": usuario, "password": password},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def descargar_producto(producto, destino_zip, usuario, password):
    """
    Descarga en streaming a un .part y lo renombra al terminar. Las redirecciones
    se siguen a mano reenviando la cabecera Authorization: 'requests' la elimina
    si la redirección cambia de dominio y el servidor respondería 401.
    """
    token = obtener_token(usuario, password)
    parcial = str(destino_zip) + ".part"
    esperado = producto.get("ContentLength") or 0

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    url = URL_DESCARGA.format(id=producto["Id"])
    try:
        r = session.get(url, stream=True, allow_redirects=False, timeout=120)
        saltos = 0
        while r.status_code in REDIRECCIONES and saltos < 10:
            url = r.headers["Location"]
            r.close()
            r = session.get(url, stream=True, allow_redirects=False, timeout=120)
            saltos += 1
        with r:
            r.raise_for_status()
            with open(parcial, "wb") as f:
                for trozo in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(trozo)
        if esperado and os.path.getsize(parcial) != esperado:
            raise IOError(f"Descarga incompleta: {os.path.getsize(parcial)} de {esperado} bytes")
        os.replace(parcial, destino_zip)
    except Exception:
        if os.path.exists(parcial):
            os.remove(parcial)
        raise


def main():
    ap = argparse.ArgumentParser(description="Descarga Sentinel-2 L2A de CDSE para el AOI del proyecto.")
    ap.add_argument("-p", "--project", default=config.NOMBRE_PROYECTO)
    ap.add_argument("--inicio", required=True, help="YYYY-MM-DD")
    ap.add_argument("--fin", default=date.today().isoformat(), help="YYYY-MM-DD")
    ap.add_argument("--nubosidad", type=float, default=config.NUBOSIDAD_MAX_ESCENA)
    ap.add_argument("--solo-listar", action="store_true", help="No descarga, solo muestra")
    args = ap.parse_args()

    usuario, password = os.environ.get("CDSE_USER"), os.environ.get("CDSE_PASSWORD")
    if not args.solo_listar and not (usuario and password):
        raise SystemExit("Faltan las credenciales: define las variables de entorno CDSE_USER y CDSE_PASSWORD")

    for carpeta in (config.CARPETA_ENTRADA, config.CARPETA_PROCESADOS, config.CARPETA_ERROR):
        os.makedirs(carpeta, exist_ok=True)

    geojson = config.project_path(args.project) / "input" / "polygon.geojson"
    if not geojson.is_file():
        raise SystemExit(f"No existe el polígono del AOI: {geojson}")
    productos = buscar_productos(geojson_a_wkt(geojson), args.inicio, args.fin, args.nubosidad)
    print(f"{len(productos)} productos encontrados.")

    errores = 0
    for p in productos:
        nombre_zip = p["Name"].replace(".SAFE", "") + ".zip"
        destinos = [d / nombre_zip for d in (config.CARPETA_ENTRADA, config.CARPETA_PROCESADOS, config.CARPETA_ERROR)]
        if any(d.exists() for d in destinos):
            print(f"  Ya existe, se omite: {nombre_zip}")
            continue
        if not p.get("Online", True):
            print(f"  Archivado (offline), se omite: {nombre_zip}")
            continue

        print(f"  {nombre_zip}  ({p.get('ContentLength', 0) / 1e6:.0f} MB)")
        if args.solo_listar:
            continue
        try:
            descargar_producto(p, destinos[0], usuario, password)
        except Exception as e:  # noqa: BLE001
            errores += 1
            print(f"    ERROR descargando: {e}")

    print(f"Descarga terminada ({errores} errores). Ahora puedes ejecutar sesion5.py")


if __name__ == "__main__":
    main()
