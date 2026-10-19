# 🌍 Monitorización de la línea de costa con Sentinel-2

Proyecto CDIO1 · Ingeniería de Satélites · EETAC–UPC

Herramienta desarrollada en Python para estudiar la evolución de la línea de costa mediante imágenes Sentinel-2 obtenidas de Copernicus Data Space Ecosystem.

El proyecto transforma imágenes satelitales en medidas que permiten comparar la posición de la costa entre fechas, visualizar sus variaciones y evaluar los resultados frente a referencias externas.

La zona de estudio configurada actualmente corresponde a un área costera de Barcelona.

---

## 🎯 Objetivo

Desarrollar un sistema para estudiar los cambios costeros mediante la descarga de imágenes, su procesamiento y el análisis de las medidas obtenidas.

El sistema busca responder a preguntas como:

- ¿Cómo cambia la posición de la costa durante el período observado?
- ¿Qué tramos presentan mayores desplazamientos?
- ¿Cómo evolucionan las distancias respecto a una costa de referencia?
- ¿Qué discrepancia existe entre la costa detectada y una referencia externa?

La interpretación de los desplazamientos como erosión o acreción requiere comprobar la orientación de las medidas y la calidad de las observaciones.

## 🛰️ Cómo funciona

### 1. Descarga de imágenes

El usuario define la zona mediante un polígono GeoJSON y selecciona un intervalo de fechas.

El programa consulta el catálogo de Copernicus, filtra productos Sentinel-2 L2A según su nubosidad y descarga los archivos que todavía no están disponibles localmente.

### 2. Procesamiento y detección del agua

El sistema lee las bandas de los productos aceptados, genera una composición RGB y calcula el índice NDWI utilizando las bandas verde e infrarroja cercana.

Después aplica un umbral para generar una máscara binaria de agua y tierra.

La versión ampliada incorpora limpieza morfológica y guarda comparaciones para inspeccionar sus efectos.

### 3. Extracción de la costa

El programa detecta puntos del límite agua-tierra dentro del área de estudio.

Los puntos se exportan en distintos formatos y se reúnen en una tabla con coordenadas y fechas.

### 4. Medición de desplazamientos

Se construye una costa de referencia y se generan transectos perpendiculares.

Las intersecciones con las costas de otras fechas permiten calcular desplazamientos y estadísticas por observación.

### 5. Visualización y validación

El sistema representa las distancias por transecto y su evolución media en el tiempo.

También permite comparar una costa estimada con una referencia externa y calcular el RMSE de las separaciones medidas.

## ✨ Funcionalidades actuales

- 📥 Descarga de productos Sentinel-2 por zona, fechas y nubosidad.
- 🖼️ Generación de composiciones RGB e imágenes de NDWI.
- 🌊 Clasificación del agua mediante un umbral de NDWI.
- 🧹 Limpieza morfológica de máscaras.
- 📍 Extracción de puntos del borde agua-tierra.
- 🗺️ Exportación a CSV, GeoJSON y KMZ.
- 📏 Medición de desplazamientos mediante transectos.
- 📊 Estadísticas de distancias y cobertura por fecha.
- ✅ Comparación con referencias externas.
- 📈 Visualización de resultados espaciales y temporales.

## 📂 Archivos principales

| Archivo | Función |
| --- | --- |
| descarga_copernicus.py | Buscar y descargar productos de Copernicus. |
| sesion5.py / sesion8.py | Versiones del procesamiento por lotes: RGB, NDWI y máscaras. |
| extract_shorelines.py | Extraer puntos del límite agua-tierra. |
| calculate_erosion.py | Calcular desplazamientos respecto a una referencia. |
| error_estimator.py | Comparar la costa estimada con una referencia externa. |
| analyze.py | Generar gráficas a partir de las medidas. |
| shoreline_utils.py | Compartir funciones de rutas, fechas, coordenadas y geometría. |

Los procesadores por lotes representan versiones alternativas de una misma etapa, no dos pasos consecutivos.

## 🚀 Primeros pasos

1. Clonar el repositorio.
2. Preparar un entorno de Python e instalar las dependencias.
3. Configurar las credenciales de Copernicus mediante variables de entorno.
4. Comprobar el polígono del área de estudio y las rutas.
5. Descargar los productos del período seleccionado.
6. Ejecutar el procesador elegido para generar las máscaras.
7. Extraer los puntos de costa y calcular las distancias.
8. Visualizar los resultados y realizar la validación si se dispone de referencias externas.

Las credenciales deben mantenerse fuera del código y del repositorio.

## 🛠️ Tecnologías utilizadas

- Python.
- Requests.
- NumPy y Pandas.
- Rasterio y GeoPandas.
- Shapely y PyProj.
- SciPy y scikit-image.
- Matplotlib y Pillow.
- tqdm.

## 📊 Resultados generados

- Imágenes RGB para inspección visual.
- Mapas NDWI y máscaras binarias georreferenciadas.
- Comparaciones de máscaras antes y después de la limpieza.
- Coordenadas de los puntos de costa.
- Distancias por fecha y transecto.
- Resúmenes estadísticos y cobertura de medidas.
- Gráficas comparativas y de evolución temporal.
- Registro del procesamiento y de los errores encontrados.

## 🔬 Próximos pasos

- Unificar la configuración y las rutas.
- Consolidar los procesadores en una única versión configurable.
- Mejorar la gestión de errores.
- Incorporar análisis de series temporales.
- Ajustar regresiones lineales respecto al tiempo.
- Interpretar las pendientes como tasas de cambio.
- Evaluar los ajustes mediante MAE, RMSE y R².
- Analizar residuos y posibles anomalías.

Estas ampliaciones no se presentan como implementadas en los módulos actuales.

## ⚠️ Limitaciones

- La clasificación del agua utiliza un umbral fijo, no un modelo entrenado.
- Las nubes, los datos no válidos y la limpieza pueden influir en el borde detectado.
- El extractor identifica bordes de agua, no exclusivamente costa marina.
- La reconstrucción geométrica puede conectar grupos de puntos separados.
- El signo de las distancias depende de la orientación de la referencia.
- Las referencias externas de otras fechas pueden incluir cambios reales de la costa.
- Una serie corta no permite justificar conclusiones estacionales o climáticas.

## 👥 Equipo


- Antoni Ruiz.
- Joel Garcia.
- Jan .
- Kaua Vieira.

## 🛰️ Estado del proyecto

Proyecto académico en desarrollo, orientado a conectar la observación de la Tierra con el procesamiento geoespacial y el análisis de la evolución costera.
