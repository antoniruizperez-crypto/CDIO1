<h1 align="center">🌍 Monitorización de la línea de costa</h1>

<p align="center">
  Sentinel-2 · Copernicus · Procesamiento geoespacial · Python
</p>

<p align="center">
  Proyecto CDIO1 — Grado en Ingeniería de Satélites<br>
  EETAC · Universitat Politècnica de Catalunya
</p>

<p align="center">
  <a href="#presentacion">Presentación</a> ·
  <a href="#flujo-de-trabajo">Funcionamiento</a> ·
  <a href="#primeros-pasos">Primeros pasos</a> ·
  <a href="#equipo">Equipo</a>
</p>

---

<a id="presentacion"></a>
## 🛰️ Presentación

Herramienta desarrollada en Python para estudiar la evolución de la línea de costa mediante imágenes Sentinel-2 obtenidas de Copernicus Data Space Ecosystem.

El sistema conecta la descarga de productos satelitales con la detección del límite agua-tierra, la medición de desplazamientos y la visualización de resultados. También permite contrastar la costa estimada con referencias externas.

La zona de estudio configurada actualmente corresponde a un área costera de Barcelona.

> Del píxel a la medida: transformar imágenes satelitales en información sobre los cambios de la costa.

| Contexto | Descripción |
| :--- | :--- |
| Ámbito | Observación de la Tierra y análisis costero |
| Fuente de datos | Productos Sentinel-2 de Copernicus |
| Zona de estudio | Área costera de Barcelona |
| Lenguaje | Python |
| Estado | Proyecto académico en desarrollo |

<a id="contenido"></a>
## 📑 Contenido

1. [Objetivo](#objetivo)
2. [Flujo de trabajo](#flujo-de-trabajo)
3. [Funcionalidades actuales](#funcionalidades)
4. [Módulos del sistema](#modulos)
5. [Productos y resultados](#resultados)
6. [Primeros pasos](#primeros-pasos)
7. [Tecnologías](#tecnologias)
8. [Próxima fase · Sesión 9](#sesion-9)
9. [Alcance y limitaciones](#limitaciones)
10. [Equipo](#equipo)

---

<a id="objetivo"></a>
## 🎯 Objetivo

Estudiar cómo cambia la posición de la línea de costa durante un período determinado y comparar su comportamiento entre distintos tramos de una misma zona.

| Pregunta | Cómo se estudia |
| :--- | :--- |
| ¿Cómo cambia la costa entre fechas? | Midiendo distancias respecto a una costa de referencia. |
| ¿Qué tramos presentan mayores desplazamientos? | Comparando las medidas de los distintos transectos. |
| ¿Cómo evoluciona el comportamiento medio de la zona? | Calculando estadísticas y representándolas por fecha. |
| ¿Qué discrepancia existe respecto a una referencia externa? | Calculando separaciones y su RMSE. |

> La interpretación de los desplazamientos como erosión o acreción requiere comprobar la orientación de los transectos, la referencia utilizada y la calidad de las observaciones.

<a id="flujo-de-trabajo"></a>
## ⚙️ Flujo de trabajo

<p align="center">
  Zona y fechas → Descarga → Procesamiento → Extracción → Medición → Visualización
</p>

### 01 · 📥 Adquisición de imágenes

El área de estudio se define mediante un polígono GeoJSON y se selecciona un intervalo de fechas.

El programa consulta el catálogo de Copernicus, filtra productos Sentinel-2 L2A según su nubosidad y descarga los que todavía no están disponibles localmente.

### 02 · 🌊 Procesamiento y detección del agua

Para los productos L2A que superan el filtro, el sistema:

- Lee las bandas necesarias directamente desde el ZIP.
- Genera una composición RGB.
- Calcula el índice NDWI a partir de las bandas verde e infrarroja cercana.
- Aplica un umbral para obtener una máscara binaria de agua y tierra.
- Guarda productos de inspección y un registro del procesamiento.

La versión ampliada incorpora una apertura morfológica (`opening`) para reducir pequeñas regiones de agua y guarda comparaciones antes y después de su aplicación.

### 03 · 📍 Extracción del borde

El extractor puede aplicar un cierre morfológico (`closing`) y detecta los píxeles de transición agua-tierra dentro del área de estudio.

Estos puntos se convierten en coordenadas, se exportan a CSV, GeoJSON y KMZ y se reúnen en una tabla con sus fechas.

### 04 · 📏 Medición de desplazamientos

Se construye una costa de referencia y se generan transectos perpendiculares.

Las intersecciones de los transectos con las costas de otras fechas permiten calcular distancias y obtener estadísticas por observación.

### 05 · 📈 Visualización

Los resultados se representan mediante:

- Curvas de distancia por transecto para cada fecha.
- Evolución temporal de la distancia media.
- Gráficas de las geometrías y sus intersecciones.

### ✅ Validación con referencias externas

Como comprobación independiente, el sistema puede comparar la costa estimada de una fecha con una referencia externa.

Se selecciona el archivo de referencia de fecha más próxima y se calculan las separaciones por transecto y su RMSE.

---

<a id="funcionalidades"></a>
## ✨ Funcionalidades actuales

| Área | Funcionalidades |
| :--- | :--- |
| Adquisición | Búsqueda por zona y fechas, filtro de nubosidad y descarga de productos. |
| Procesamiento | Generación de RGB, NDWI y máscaras binarias georreferenciadas. |
| Limpieza | Operaciones morfológicas e inspección de sus efectos. |
| Extracción | Detección y exportación de puntos del borde agua-tierra. |
| Medición | Distancias por transecto y estadísticas por fecha. |
| Validación | Comparación con referencias externas mediante RMSE. |
| Visualización | Gráficas de geometrías, distancias y evolución media. |
| Registro | Información del procesamiento, productos generados y errores. |

<a id="modulos"></a>
## 📂 Módulos del sistema

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

<a id="resultados"></a>
## 📊 Productos y resultados

### Imágenes y máscaras

| Producto | Finalidad |
| :--- | :--- |
| Composición RGB | Inspección visual de la escena. |
| NDWI coloreado | Visualización de la distribución del índice. |
| NDWI GeoTIFF | Conservación del índice continuo georreferenciado. |
| Máscara binaria | Clasificación de píxeles como agua o no agua. |
| Comparación antes/después | Inspección de los cambios introducidos por la limpieza. |

### Coordenadas y medidas

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

Los productos visuales del procesamiento se guardan también en `Imagenes/resultados/`, y el registro del lote se conserva en `Imagenes/registro_procesamiento.csv`.

[↑ Volver al índice](#contenido)

---

<a id="primeros-pasos"></a>
## 🚀 Primeros pasos

### Preparar el proyecto

1. Clonar el repositorio.
2. Preparar un entorno de Python.
3. Instalar las dependencias utilizadas por los módulos.
4. Configurar las credenciales de Copernicus mediante variables de entorno.
5. Comprobar las rutas y el polígono del área de estudio.

### Ejecutar el flujo de trabajo

1. Descargar los productos del intervalo seleccionado.
2. Ejecutar el procesador elegido para generar las máscaras.
3. Extraer los puntos de costa.
4. Calcular los desplazamientos mediante transectos.
5. Visualizar las medidas.
6. Realizar la validación si se dispone de referencias externas.

> 🔐 Las credenciales deben mantenerse fuera del código y del repositorio.

<a id="tecnologias"></a>
## 🛠️ Tecnologías

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

<a id="sesion-9"></a>
## 🔬 Próxima fase · Sesión 9

La siguiente ampliación se centra en el análisis de series temporales y la introducción al aprendizaje automático mediante regresión lineal.

### Series temporales y análisis exploratorio

- [ ] Aplicar los fundamentos de las series temporales a las medidas de posición de la costa.
- [ ] Ordenar las observaciones por fecha y revisar los valores ausentes.
- [ ] Calcular estadísticas descriptivas.
- [ ] Visualizar la evolución temporal mediante herramientas de Python.
- [ ] Identificar patrones, tendencias y posibles anomalías.

### Aprendizaje automático y regresión

- [ ] Introducir el aprendizaje supervisado mediante regresión lineal.
- [ ] Ajustar modelos que relacionen el tiempo con la posición de la costa.
- [ ] Interpretar físicamente los parámetros del modelo.
- [ ] Interpretar la pendiente como una tasa de cambio.
- [ ] Evaluar los ajustes mediante MAE, RMSE y R².

> Estas tareas están pendientes de implementación en los módulos actuales.

<a id="limitaciones"></a>
## ⚠️ Alcance y limitaciones

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

[↑ Volver al índice](#contenido)

---

<a id="equipo"></a>
## 👥 Equipo

Proyecto desarrollado por cuatro estudiantes del Grado en Ingeniería de Satélites:

| Integrantes |
| :--- |
| Antoni Ruiz |
| Joel Garcia |
| Jan Aleksander Fernández |
| Kaua Vieira |

<p align="center">
  CDIO1 · Grado en Ingeniería de Satélites<br>
  EETAC · Universitat Politècnica de Catalunya
</p>

---

<p align="center">
  🌍 Observación de la Tierra · Análisis geoespacial · Evolución costera
</p>

<p align="center">
  Proyecto académico en desarrollo
</p>
