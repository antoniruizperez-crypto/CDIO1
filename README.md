# 🌍 Monitorización de la línea de costa con Sentinel-2 (versión `prueba_extra`)

Proyecto CDIO1 · Ingeniería de Satélites · EETAC–UPC

Versión corregida y mejorada del pipeline original. Mide cómo se mueve la costa entre fechas a partir de imágenes Sentinel-2 L2A de Copernicus.

## 🚀 Puesta en marcha

```bash
pip install -r requirements.txt

# 1. Copia tu polígono del AOI aquí (no incluido en esta carpeta):
#    projects/costa_principal/input/polygon.geojson

# 2. Credenciales de Copernicus (variables de entorno, NUNCA en el código)
export CDSE_USER="tu_email"
export CDSE_PASSWORD="tu_password"       # Windows PowerShell: $env:CDSE_USER="..."

python test_sintetico.py                 # prueba sin datos reales: debe acabar en "TODO CORRECTO"
python test_sesion5_sintetico.py         # prueba de sesion5.py con un zip falso

python descarga_copernicus.py --inicio 2026-06-01 --fin 2026-09-30
python sesion5.py
python extract_shorelines.py
python calculate_erosion.py              # añade -g <carpeta_ground_truth> para usar una referencia externa
python error_estimator.py -g <carpeta_ground_truth> -d 2026-07-15
python analyze.py
```

Todas las rutas y parámetros están en `config.py`. Los scripts funcionan desde cualquier directorio.

## 🛰️ Flujo

```
descarga_copernicus.py → sesion5.py → extract_shorelines.py → calculate_erosion.py → analyze.py
   zips L2A                NDWI.tif        all_shorelines.csv       distancias            gráficas
                                                       ↘ error_estimator.py (validación)
```

| Archivo | Función |
| --- | --- |
| `config.py` | Rutas y todos los parámetros ajustables. |
| `descarga_copernicus.py` | Busca y descarga productos L2A del AOI. |
| `sesion5.py` | Recorta al AOI, filtra nubes en el AOI, calcula NDWI y guarda GeoTIFF + imágenes de inspección. |
| `extract_shorelines.py` | Limpia la máscara y extrae la costa como contornos subpíxel ordenados. |
| `calculate_erosion.py` | Transectos sobre una referencia y distancia de cada fecha. |
| `error_estimator.py` | Compara una fecha con una referencia externa (sesgo, MAE, RMSE…). |
| `analyze.py` | Gráficas por transecto y evolución temporal. |
| `shoreline_utils.py` | Funciones comunes (fechas, KMZ, geometría, transectos). |
| `test_sintetico.py`, `test_sesion5_sintetico.py` | Pruebas con datos sintéticos de resultado conocido. |

> `sesion8.py`, citado en el README original, no estaba entre los archivos recibidos; `sesion5.py` es ahora la única versión del procesador.

## ➕ / ➖ Convención de signo

La costa se orienta siempre con **el agua a la izquierda** del sentido de recorrido, y la normal de cada transecto apunta al mar:

- **distancia > 0** → la costa de esa fecha está **mar adentro** de la referencia (**acreción**)
- **distancia < 0** → está **tierra adentro** (**erosión**)

Con una referencia externa (ground truth) su sentido de recorrido se detecta automáticamente comparándola con la costa extraída, así que no importa en qué orden esté digitalizada.

## 🔧 Qué cambió respecto a la versión original

**Precisión de la costa**
- La costa se extrae como **contorno del NDWI continuo** (marching squares): posición subpíxel en lugar de los píxeles de agua junto a tierra (que daban un sesgo de ~5 m y puntos desordenados).
- Las polilíneas salen ya ordenadas: desaparecen la reconstrucción por vecino más cercano (O(n³), memoria enorme) y los saltos entre tramos separados (puertos, ríos).
- **Limpieza por área** (manchas < 25 px, tanto agua en tierra como tierra en agua) en lugar de un closing de radio 1. El closing borraba espigones y pantalanes < 20 m y no limpiaba el ruido de agua dentro de tierra. El closing sigue disponible con `--closing-radius`.
- **Umbral de NDWI por imagen (Otsu)**, acotado a [0, 0.3] con respaldo al fijo 0.2. Cambia con `MODO_UMBRAL` / `--threshold-mode fixed` si prefieres máxima consistencia entre fechas.
- **Reflectancia correcta**: se resta el `BOA_ADD_OFFSET` del XML (productos baseline ≥ 04.00 llevan +1000, que sesgaba el NDWI).
- Los píxeles sin datos, nubes y sombras quedan como **NaN** (antes se convertían en NDWI = 0, es decir, tierra).

**Filtro de nubes**
- Se mide **solo dentro del AOI** (antes, en los 100×100 km de la escena), incluyendo sombras (clase 3), con la máscara ensanchada 2 px. Se exige además una cobertura mínima del AOI.
- La escena se **recorta al AOI** al leerla: de ~2 GB a unos pocos MB por imagen.

**Transectos y medidas**
- La spline recorre los puntos **en el orden de la costa** (antes `np.unique` los ordenaba por x,y y la hacía zigzaguear) y **suaviza de verdad** (`s > 0`).
- Signo coherente también cuando hay varios cruces; se toma el cruce más cercano y se registra `n_crossings`.
- Transectos de ±150 m (antes ±50 m, que censuraba los cambios grandes) y la cobertura se informa siempre.
- El espaciado medio impreso ya no se acumula por fecha.
- Una sola referencia externa leída con una única función (antes dos parsers distintos).
- Se tratan los resultados `MultiLineString` tras recortar con el AOI (antes fallaba `.xy`).
- Se eliminan las cachés `.pkl` (ya no hacen falta y quedaban obsoletas).
- Si dos productos comparten fecha, se conserva el de menos nubes.

**Robustez y seguridad**
- **Credenciales fuera del código** (solo variables de entorno).
- Los zips que fallan van a `Imagenes/zips_error` (no se pierden); el log se escribe zip a zip; `os.replace` funciona también en Windows.
- Descarga: redirecciones con cabecera `Authorization`, comprobación de tamaño y limpieza de `.part`.
- `error_estimator.py` informa sesgo, MAE, RMSE, desviación, mediana, P90 y cobertura.
- `analyze.py` no une con líneas los tramos sin medida y guarda los PNG.

## 📊 Salidas principales

| Archivo | Contenido |
| --- | --- |
| `Imagenes/resultados/*_costa.jpg`, `*_ndwi.jpg` | RGB y NDWI para inspección visual. |
| `…/estimated_ndwi/*.tif` | NDWI continuo (NaN = inválido), con la nubosidad del AOI en las etiquetas. |
| `…/estimated_waterbodies_images_clean/*.tif` | Máscara limpia (0 tierra, 1 agua, 255 inválido). |
| `…/estimated_waterbodies_edges/all_shorelines.csv` | `x,y,date,segment,order,scene`. |
| `…/estimated_waterbodies_edges/extraction_params.csv` | Por fecha: umbral, píxeles corregidos por la limpieza, nº y longitud de segmentos, nubes. |
| `…/estimated_erosion_accretion/shoreline_distances.csv` | Fecha × transecto: `distance_m`, `n_crossings`. |
| `…/estimated_erosion_accretion/distances_summary.csv` | Media, mediana, desviación, rango, cobertura por fecha. |
| `…/estimated_error/error_metrics.csv` | Métricas de validación. |

## ⚠️ Limitaciones que siguen vigentes

- La resolución nativa es de 10 m. El método subpíxel mejora la posición, pero la precisión real depende de la mezcla agua/tierra del píxel, de las olas, de la marea y de la pendiente de la playa. **No se corrige la marea**: la línea de agua del día no es la línea de costa media.
- El NDWI con umbral no es un modelo entrenado; espigones, puertos y edificios húmedos pueden clasificarse mal.
- La costa extraída incluye cualquier masa de agua conectada al AOI (puertos, desembocaduras), no solo el mar abierto.
- Una referencia externa de otra fecha mezcla error y cambio real.
- Los transectos sin cruce no entran en las métricas: comprueba siempre `coverage_ratio`.
- Una serie corta no permite justificar conclusiones estacionales ni climáticas.

## 🔬 Próximos pasos

- Corrección por marea (altura de marea en cada adquisición) y regresiones por transecto con MAE/RMSE/R².
- Análisis de residuos y anomalías.

## 👥 Equipo

Antoni Ruiz · Joel Garcia · Jan Aleksander Fernández · Kaua Vieira
