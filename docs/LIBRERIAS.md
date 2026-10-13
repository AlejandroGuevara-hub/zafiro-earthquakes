# Librerias y funciones

Proposito de **cada libreria** y de **cada funcion** del proyecto, archivo por
archivo. Util para la sustentacion: aqui esta "que hace y por que" cada pieza.

## Librerias usadas

| Libreria | Version (equipo) | Para que se usa |
|---|---|---|
| `pandas` | 3.0.1 | Cargar/limpiar el CSV, agrupar por municipio, tablas. |
| `numpy` | 2.4.2 | Operaciones numericas, `np.select`, arreglos. |
| `scikit-learn` | 1.8.0 | **Corazon del ML**: `StandardScaler`, `KMeans`, `PCA`, `silhouette_score`. |
| `joblib` | (incluida) | Guardar/cargar el modelo entrenado (`.pkl`). |
| `streamlit` | 1.65.0 | Interfaz del dashboard (pestanas, metricas, graficas, tablas). |
| `streamlit-folium` | (instalada) | Incrustar el mapa de folium en Streamlit (`st_folium`). |
| `folium` | 0.20.0 | Mapas interactivos: coropleta por municipio y epicentros. |
| `matplotlib` | (incluida) | Base de las figuras. |
| `seaborn` | 0.13.2 | Graficas estadisticas (barras, boxplot, dispersion). |
| `html` | stdlib | Decodificar entidades HTML (`&aacute;`). |
| `re` | stdlib | Expresiones regulares para limpiar texto y extraer municipio. |
| `io.StringIO` | stdlib | Leer el CSV desde texto en memoria. |
| `pathlib.Path` | stdlib | Rutas de archivos independientes del sistema operativo. |
| `urllib.request` | stdlib | Descargar el GeoJSON desde el servicio del DANE. |
| `json` | stdlib | Leer/escribir el GeoJSON. |
| `unicodedata` | stdlib | Quitar tildes para unir nombres de municipios. |

## `src/data_prep.py`

| Funcion | Que hace |
|---|---|
| `_clean_text(valor)` | Decodifica entidades HTML y colapsa espacios en textos. |
| `_parse_intensidad(valor)` | Convierte `"6"` o `"6/7"` a numero (promedio). |
| `cargar_catalogo(path)` | Localiza el encabezado (linea con `"No."`) y lee el CSV. |
| `limpiar(df)` | Renombra, decodifica, castea tipos, valida coordenadas, deduplica. |
| `_municipio_base(valor)` | Extrae el municipio real desde `Centro poblado`. |
| `agregar_features(df)` | Crea `anio`, `mes`, `municipio_base`, `clase_profundidad`. |
| `etiquetar_riesgo(df)` | Columna descriptiva `riesgo_sismico` por umbrales de intensidad. |
| `preparar()` | Ejecuta todo y exporta `datos/sismos_clean.csv`. |

## `src/geo_prep.py`

| Funcion | Que hace |
|---|---|
| `descargar()` | Consulta el servicio ArcGIS del DANE y baja el GeoJSON de Santander (`COD_DPTO=68`). |
| `cargar()` | Usa el GeoJSON local si existe; si no, descarga. |
| `_redondear(valor, decimales)` | Redondea coordenadas recursivamente para aligerar el archivo. |
| `simplificar(geojson, decimales)` | Conserva solo `codigo` y `municipio` y reduce precision. |

## `src/segmentar.py`

| Funcion | Que hace |
|---|---|
| `agregar_por_municipio(df)` | Resume cada municipio: `n_eventos`, `intensidad_media/max`, `magnitud_max`, `profundidad_media`. |
| `evaluar_k(Xs, ks)` | Prueba varios k y calcula inercia (codo) y silhouette. |
| `nombrar_clusters(municipios, k)` | Ordena los grupos por intensidad media -> Alto / Medio / Bajo. |
| `segmentar(k)` | Pipeline: escala -> KMeans -> PCA -> exporta resultados y modelo. |

**Clases de scikit-learn empleadas:**

| Objeto | Rol |
|---|---|
| `StandardScaler` | Estandariza (media 0, desv 1) para que las distancias sean justas. |
| `KMeans` | Algoritmo que agrupa municipios por similitud (no supervisado). |
| `PCA` | Reduce a 2 dimensiones para visualizar los grupos. |
| `silhouette_score` | Metrica de calidad del agrupamiento (separacion de grupos). |

## `src/app.py`

| Funcion | Que hace |
|---|---|
| `normalizar(texto)` | Mayusculas sin tildes para unir dataset y GeoJSON. |
| `cargar()` | Carga con cache el clustering, los sismos, las metricas y el GeoJSON. |
| `construir_mapa(...)` | Arma el mapa folium: coropleta por nivel + tooltip + epicentros. |
| `pestana_mapa(...)` | Renderiza la pestana "Mapa" del dashboard. |
| `pestana_analisis(...)` | Renderiza la pestana "Analisis" (graficas y tablas). |
| `cargar_modelo()` | Carga con cache `models/segmentacion.pkl` (scaler + kmeans + pca). |
| `niveles_por_cluster(municipios)` | Relaciona cada cluster de KMeans con su nivel (Alto/Medio/Bajo). |
| `clasificar(modelo, niveles, media, max)` | Estandariza un caso, predice su nivel y calcula la distancia a cada centroide. |
| `pestana_municipio(...)` | Renderiza la pestana "Municipio": ficha, explicacion del nivel e historial. |
| `pestana_simulador(...)` | Renderiza la pestana "Simulador": clasifica un caso nuevo con el modelo. |
| `main()` | Configura la pagina y organiza las pestanas. |

**Elementos de folium/streamlit usados:** `folium.Map`, `folium.Choropleth`
(colorea municipios por valor), `folium.GeoJson` + `GeoJsonTooltip` (ficha al
pasar el mouse), `folium.CircleMarker` (epicentros), `st.tabs`, `st.metric`,
`st.pyplot`, `st.dataframe`, `st_folium`.
