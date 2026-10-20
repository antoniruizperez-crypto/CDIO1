"""
Configuración central del proyecto.

Todas las rutas se calculan a partir de la ubicación de este archivo, así que
los scripts funcionan desde cualquier directorio de trabajo.

Estructura esperada (dentro de esta carpeta):

    projects/<NOMBRE_PROYECTO>/input/polygon.geojson   <- AOI (lo pones tú)
    Imagenes/zips_entrada/                             <- zips Sentinel-2 L2A
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = BASE_DIR / "projects"
NOMBRE_PROYECTO = "costa_principal"


def project_path(project):
    return PROJECTS_DIR / project


# ---------------------------------------------------------------------------
# Carpetas de imágenes
# ---------------------------------------------------------------------------
IMAGES_DIR = BASE_DIR / "Imagenes"
CARPETA_ENTRADA = IMAGES_DIR / "zips_entrada"
CARPETA_PROCESADOS = IMAGES_DIR / "zips_procesados"
CARPETA_ERROR = IMAGES_DIR / "zips_error"      # zips que fallaron: se pueden reintentar
CARPETA_SALIDA = IMAGES_DIR / "resultados"     # JPG de inspección (RGB y NDWI)
CSV_LOG = IMAGES_DIR / "registro_procesamiento.csv"

# ---------------------------------------------------------------------------
# Descarga (descarga_copernicus.py)
# ---------------------------------------------------------------------------
# Filtro laxo sobre TODA la escena (100x100 km) solo para no bajar escenas
# totalmente nubladas. El filtro real se aplica después, sobre el AOI.
NUBOSIDAD_MAX_ESCENA = 40.0

# ---------------------------------------------------------------------------
# Procesamiento de imágenes (sesion5.py)
# ---------------------------------------------------------------------------
# % máximo de píxeles de nube/sombra DENTRO DEL AOI para aceptar la imagen
UMBRAL_NUBOSIDAD_AOI = 10.0
# Fracción mínima del AOI que debe tener datos válidos en la escena
COBERTURA_MIN_AOI = 0.8
# Clases SCL: 3 sombra de nube, 8/9 nube prob. media/alta, 10 cirros
CLASES_NUBE = {3, 8, 9, 10}
# Clases que se enmascaran (además de las nubes: 0 sin datos, 1 saturado/defectuoso)
CLASES_INVALIDAS = {0, 1} | CLASES_NUBE
# Se ensancha la máscara de nubes N píxeles (bordes de nube y halos)
DILATACION_NUBE_PX = 2
# Margen (px) alrededor del AOI al recortar la escena
MARGEN_VENTANA_PX = 20

# ---------------------------------------------------------------------------
# Umbral de NDWI (agua / tierra)
# ---------------------------------------------------------------------------
# "otsu"  -> umbral propio por imagen (más exacto, se adapta a turbidez/iluminación)
# "fixed" -> siempre UMBRAL_NDWI_FIJO (más consistente entre fechas)
MODO_UMBRAL = "otsu"
UMBRAL_NDWI_FIJO = 0.2
# Otsu se acota a este rango; si no hay bimodalidad se usa el fijo
OTSU_MIN = 0.0
OTSU_MAX = 0.3

# ---------------------------------------------------------------------------
# Extracción de la costa (extract_shorelines.py)
# ---------------------------------------------------------------------------
# Manchas (agua o tierra) menores que esto, en píxeles (1 px = 100 m²), se
# eliminan. 25 px = 2500 m². No borra espigones largos y estrechos porque
# están conectados a tierra y son grandes en conjunto.
MIN_AREA_PX = 25
# Closing morfológico opcional (0 = desactivado). Borra estructuras < 2*radio px.
CLOSING_RADIUS = 0
# Los segmentos de costa más cortos que esto (m) se descartan
MIN_SEGMENT_LENGTH_M = 100.0
# Valor que se asigna al NDWI de los píxeles corregidos (umbral +/- margen)
FLIP_MARGIN = 0.1
SAVE_CLEAN_MASKS = True

# ---------------------------------------------------------------------------
# Transectos (calculate_erosion.py / error_estimator.py)
# ---------------------------------------------------------------------------
SPATIAL_SPACING = 10          # m entre transectos
TRANSECT_HALF_LENGTH = 150    # m a cada lado de la referencia (antes ±50)
SPLINE_SMOOTH_M = 2.0         # desviación típica (m) tolerada por la spline de suavizado
REF_MIN_SEGMENT_M = 200.0     # segmentos de referencia más cortos se ignoran
GT_CRS = 32631                # CRS de los CSV de referencia externa (UTM 31N)
