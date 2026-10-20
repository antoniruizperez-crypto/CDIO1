<h1 align="center">Monitorización de la línea de costa</h1>

<p align="center">
  Sentinel-2 · Copernicus · Procesamiento geoespacial · Python
</p>

<p align="center">
  Proyecto CDIO1 — Ingeniería de Satélites<br>
  EETAC · Universitat Politècnica de Catalunya
</p>

---

## Presentación

Sistema desarrollado en Python para estudiar la evolución de la línea de costa a partir de imágenes Sentinel-2 obtenidas de Copernicus Data Space Ecosystem.

El proyecto conecta la descarga de productos satelitales con la detección del límite agua-tierra, la medición de desplazamientos y la visualización de los resultados.

La zona de estudio configurada actualmente corresponde a un área costera de Barcelona.

> Del píxel a la medida: transformar imágenes satelitales en información sobre los cambios de la costa.

## Contenido

- [Objetivo](#objetivo)
- [Flujo de trabajo](#flujo-de-trabajo)
- [Módulos del sistema](#módulos-del-sistema)
- [Productos y resultados](#productos-y-resultados)
- [Primeros pasos](#primeros-pasos)
- [Tecnologías](#tecnologías)
- [Próxima fase](#próxima-fase)
- [Alcance y limitaciones](#alcance-y-limitaciones)
- [Equipo](#equipo)

---

## Objetivo

Estudiar cómo cambia la posición de la línea de costa durante un período determinado y comparar su comportamiento entre distintos tramos de una misma zona.

| Pregunta | Resultado que permite estudiarla |
| :--- | :--- |
| ¿Cómo cambia la costa entre fechas? | Distancias respecto a una costa de referencia. |
| ¿Qué tramos presentan mayores desplazamientos? | Medidas individualizadas por transecto. |
| ¿Cómo evoluciona el comportamiento medio de la zona? | Estadísticas y gráficas por fecha. |
| ¿Qué discrepancia existe respecto a una referencia externa? | Distancias de validación y RMSE. |

La interpretación de los desplazamientos como erosión o acreción requiere comprobar la orientación de los transectos, la referencia utilizada y la calidad de las observaciones.

## Flujo de trabajo

### 01 · Adquisición

Se define el área de estudio mediante un polígono GeoJSON y se selecciona un intervalo de fechas.

El sistema consulta el catálogo de Copernicus, aplica un filtro de nubosidad y descarga los productos Sentinel-2 L2A que todavía no están disponibles localmente.

### 02 · Procesamiento de imágenes

Para los productos L2A aceptados, el programa:

- Lee las bandas necesarias directamente desde el ZIP.
- Genera una composición RGB.
- Calcula el índice NDWI.
- Aplica un umbral para obtener una máscara de agua y tierra.
- Guarda productos de inspección y un registro del procesamiento.

La versión ampliada incorpora una limpieza mediante apertura morfológica y genera comparaciones antes y después de su aplicación.

### 03 · Extracción del borde

El extractor aplica una limpieza opcional mediante cierre morfológico y localiza los píxeles de transición agua-tierra dentro del área de estudio.

Los puntos se convierten en coordenadas y se exportan a CSV, GeoJSON y KMZ.

### 04 · Medición

Se construye una costa de referencia y se generan transectos perpendiculares.

Las intersecciones de esos transectos con las costas de otras fechas permiten calcular distancias y resumir su distribución.

### 05 · Visualización

Las medidas se representan mediante:

- Curvas de distancia por transecto para cada fecha.
- Evolución temporal de la distancia media.
- Gráficas de geometrías e intersecciones.

### Validación independiente

Cuando se dispone de una referencia externa, el sistema compara con ella una costa estimada de una fecha concreta.

El resultado incluye distancias por transecto y un RMSE de las separaciones medidas.

---

## Módulos del sistema

| Etapa | Archivo | Responsabilidad |
| :--- | :--- | :--- |
| Descarga | `descarga_copernicus.py` | Consultar el catálogo y descargar productos. |
| Procesamiento | `sesion5.py` / `sesion8.py` | Leer bandas, calcular NDWI y generar máscaras. |
| Extracción | `extract_shorelines.py` | Limpiar máscaras y extraer puntos del borde agua-tierra. |
| Medición | `calculate_erosion.py` | Generar transectos, calcular distancias y resumirlas por fecha. |
| Validación | `error_estimator.py` | Comparar una costa estimada con una referencia externa. |
| Visualización | `analyze.py` | Representar distancias por transecto y su evolución media. |
| Utilidades | `shoreline_utils.py` | Gestionar rutas, fechas, coordenadas y construcción de líneas. |

> Los procesadores por lotes son versiones alternativas de una misma etapa. No deben ejecutarse como dos pasos consecutivos sobre los mismos ZIP.

## Productos y resultados

### Imágenes y máscaras

| Producto | Uso |
| :--- | :--- |
| Composición RGB | Inspección visual de la escena. |
| NDWI coloreado | Inspección de la distribución del índice. |
| NDWI GeoTIFF | Conservación del índice continuo georreferenciado. |
| Máscara binaria | Clasificación de píxeles como agua o no agua. |
| Comparación antes/después | Inspección de los cambios introducidos por la limpieza. |

### Resultados geoespaciales y medidas

Las siguientes rutas son relativas a la carpeta `output` del proyecto:

| Carpeta o archivo | Contenido |
| :--- | :--- |
| `estimated_waterbodies_images/` | Máscaras de entrada para el extractor. |
| `estimated_waterbodies_images_closed/` | Máscaras tras el cierre morfológico, cuando se activa su guardado. |
| `estimated_waterbodies_edges/` | Puntos de costa exportados. |
| `estimated_waterbodies_edges/all_shorelines.csv` | Coordenadas de los puntos y sus fechas. |
| `estimated_erosion_accretion/shoreline_distances.csv` | Distancias por fecha y transecto. |
| `estimated_erosion_accretion/distances_summary.csv` | Estadísticas y cobertura de medidas por fecha. |
| `estimated_error/shoreline_distances.csv` | Separaciones respecto a una referencia externa. |

Los productos visuales del procesamiento y su registro se guardan también en las rutas `Imagenes/resultados/` e `Imagenes/registro_procesamiento.csv`.

## Primeros pasos

1. Clonar el repositorio.
2. Preparar un entorno de Python.
3. Instalar las dependencias utilizadas por los módulos.
4. Configurar las credenciales de Copernicus mediante variables de entorno.
5. Comprobar las rutas y el polígono del área de estudio.
6. Descargar los productos del intervalo seleccionado.
7. Ejecutar el procesador elegido para generar las máscaras.
8. Extraer los puntos de costa.
9. Calcular y visualizar los desplazamientos.
10. Realizar la validación si se dispone de referencias externas.

> Las credenciales deben mantenerse fuera del código y del repositorio.

## Tecnologías

| Área | Herramientas |
| :--- | :--- |
| Lenguaje | Python |
| Consultas y descargas | Requests |
| Tratamiento de datos | NumPy · Pandas |
| Imágenes georreferenciadas | Rasterio |
| Datos vectoriales y geometría | GeoPandas · Shapely |
| Transformación de coordenadas | PyProj |
| Interpolación y cálculo | SciPy |
| Morfología de imágenes | scikit-image |
| Visualización | Matplotlib · Pillow |
| Progreso del procesamiento | tqdm |

---

## Próxima fase 

La siguiente ampliación se centra en el análisis temporal y la introducción al aprendizaje automático mediante regresión lineal.

### Series temporales y análisis exploratorio

- [ ] Aplicar los fundamentos de las series temporales a las medidas de posición de la costa.
- [ ] Ordenar las observaciones por fecha y revisar los valores ausentes.
- [ ] Calcular estadísticas descriptivas.
- [ ] Visualizar la evolución temporal mediante herramientas de Python.
- [ ] Identificar patrones, tendencias y posibles anomalías.

### Aprendizaje automático y regresión

- [ ] Introducir el aprendizaje supervisado mediante regresión lineal.
- [ ] Ajustar modelos que relacionen el tiempo con la posición de la costa.
- [ ] Interpretar físicamente sus parámetros.
- [ ] Interpretar la pendiente como una tasa de cambio.
- [ ] Evaluar los ajustes mediante MAE, RMSE y R².

Estas tareas están pendientes de implementación en los módulos actuales.

## Alcance y limitaciones

### Clasificación y extracción

- La clasificación del agua utiliza un umbral fijo, no un modelo entrenado.
- La nubosidad se calcula sobre la SCL seleccionada, no específicamente sobre el polígono de estudio.
- El procesamiento no aplica la SCL como máscara de exclusión al NDWI.
- La limpieza morfológica puede modificar detalles del contorno.
- El extractor identifica bordes de agua, no exclusivamente costa marina.

### Medición e interpretación

- La reconstrucción de líneas puede conectar grupos de puntos separados.
- Las coordenadas deben ser compatibles y estar expresadas en unidades adecuadas para medir distancias.
- El significado del signo depende de la orientación de la referencia.
- Las medias por fecha pueden incluir distintos transectos si faltan medidas.
- Una referencia externa de otra fecha puede introducir cambios reales en la discrepancia observada.
- Una serie corta no permite justificar conclusiones estacionales o climáticas.

## Equipo

Proyecto desarrollado por cuatro estudiantes:

| Integrante |
| :--- |
| Antoni Ruiz |
| Joel Garcia |
| Jan Aleksander Fernández |
| Kaua Vieira |

CDIO1 · Grado en Ingeniería de Satélites  
EETAC · Universitat Politècnica de Catalunya

---

<p align="center">
  Proyecto académico en desarrollo<br>
  Observación de la Tierra · Análisis geoespacial · Evolución costera
</p>
