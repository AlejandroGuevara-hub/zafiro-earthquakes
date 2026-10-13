# Metodologia

Documento que explica **que** hace el proyecto, **como** y **por que** en cada
etapa. Es la base tecnica para la sustentacion.

## 1. Objetivo y enfoque

Clasificar e **interpretar territorialmente** el riesgo sismico de los municipios
de Santander a partir de datos oficiales del SGC.

El enfoque es **aprendizaje no supervisado**: no existe una etiqueta "verdadera"
de riesgo ni se pretende predecir el futuro. El modelo **agrupa** municipios con
comportamiento sismico similar y esos grupos se **interpretan** como niveles de
riesgo (Alto / Medio / Bajo). Esto cumple el requisito de un sistema de ML
funcional (no es un sistema de reglas fijas) sin caer en prediccion temporal.

## 2. Datos

- **Origen:** Servicio Geologico Colombiano (SGC). Archivo `datos/sismos_raw.csv`,
  ya filtrado por el SGC al departamento de Santander.
- **Estructura:** la unidad es el par **(sismo, municipio afectado)**. Un mismo
  sismo aparece varias veces, una por cada centro poblado donde se reporto
  intensidad.

| Metrica | Valor |
|---|---|
| Filas (sismo-municipio) | 242 |
| Sismos unicos (fecha+coordenadas+magnitud) | 24 |
| Municipios con registros | 73 (de los 87 del departamento) |
| Rango temporal | 1826 - 2023 |

**Columnas originales clave:** `Fecha`, `Latitud epicentro`, `Longitud epicentro`,
`Magnitud`, `Profundidad (km)`, `Centro poblado`, `Intensidad sitio (EMS-98)`,
`Intensidad maxima (EMS-98)`.

**Sobre la escala EMS-98:** la intensidad mide el **efecto del sismo en un lugar**
(no la energia liberada). Es el indicador mas directo del impacto sentido por la
poblacion, por eso es la variable central del analisis.

## 3. Limpieza y variables (`src/data_prep.py`)

| Paso | Funcion | Que hace y por que |
|---|---|---|
| 1 | `cargar_catalogo()` | El CSV trae filas vacias y un titulo antes del encabezado. Busca la linea que empieza con `"No."` y la usa como cabecera. |
| 2 | `_clean_text()` | Decodifica entidades HTML (`G&aacute;mbita` -> `Gambita`) y normaliza espacios. |
| 3 | `_parse_intensidad()` | Convierte intensidades tipo rango (`6/7`) a un numero (promedio). |
| 4 | `limpiar()` | Renombra columnas, castea a numero, parsea fechas, valida coordenadas en Colombia y **deduplica por (fecha, coordenadas, magnitud, municipio)** sin perder los distintos municipios de un mismo sismo. |
| 5 | `_municipio_base()` | Extrae el municipio real desde `Centro poblado` (`"El Paujil, Matanza-Santander"` -> `Matanza`). Reduce 83 textos a 73 municipios. |
| 6 | `agregar_features()` | Crea `anio`, `mes`, `municipio_base` y `clase_profundidad`. |
| 7 | `etiquetar_riesgo()` | Columna **descriptiva auxiliar** `riesgo_sismico` por umbrales de intensidad (Bajo <=3, Medio 4-5, Alto >=6). **No** es la salida del modelo; sirve de referencia rapida. |

Salida: `datos/sismos_clean.csv`.

## 4. Modelo de clasificacion no supervisada (`src/segmentar.py`)

### 4.1 Agregacion por municipio
`agregar_por_municipio()` resume cada municipio con:

- `n_eventos`: cuantos sismos sintio (frecuencia).
- `intensidad_media`: promedio de intensidad en el municipio.
- `intensidad_max`: la mayor intensidad registrada.
- `magnitud_max`, `profundidad_media`: contexto del evento mas fuerte.

### 4.2 Variables usadas para agrupar
El clustering usa **`intensidad_media` + `intensidad_max`**.

- Representan la **severidad del sacudimiento** que ha vivido cada municipio.
- **No** se usa `n_eventos` para agrupar porque la frecuencia de reportes esta
  confundida con el tamano/poblacion del municipio (las ciudades grandes
  reportan mas eventos aunque el peligro sea similar). Se conserva como columna
  descriptiva.

### 4.3 Estandarizacion
`StandardScaler` lleva cada variable a media 0 y desviacion 1. **Es obligatorio**
en K-Means porque el algoritmo mide **distancias**: sin escalar, una variable con
mayor rango dominaria el resultado.

### 4.4 Numero de grupos (k)
`evaluar_k()` prueba k = 2..6 y calcula:

- **Inercia** (metodo del codo): suma de distancias al centroide; baja al subir k.
- **Silhouette**: que tan bien separados estan los grupos (-1 a 1, mas alto mejor).

| k | Inercia | Silhouette |
|---|---|---|
| 2 | 59.87 | 0.581 |
| **3** | **33.49** | **0.452** |
| 4 | 23.43 | 0.456 |
| 5 | 15.68 | 0.484 |
| 6 | 12.97 | 0.456 |

Se elige **k = 3** para obtener tres niveles interpretables (Alto/Medio/Bajo).
El codo aparece en k = 3 (la inercia cae fuerte de k=2 a k=3 y luego se suaviza).

**Nota honesta:** k = 3 tiene el silhouette **mas bajo** del barrido (0.452); el
mejor es k = 2 (0.581). La eleccion de k = 3 es por **interpretabilidad**, no
porque sea el optimo segun esa metrica: dos grupos solo separarian "afectado /
poco afectado" y no permitirian un nivel intermedio.

### 4.5 K-Means y PCA
- `KMeans(k=3, n_init=10, random_state=42)` asigna cada municipio a un grupo.
  - `n_init=10`: repite la inicializacion 10 veces y conserva la mejor.
  - `random_state=42`: resultado **reproducible**.
- `PCA(2)` rota los municipios a 2 ejes para **visualizar** los grupos
  (91.3% + 8.7% de la varianza). Como el modelo solo tiene 2 variables, 2
  componentes conservan el 100% por construccion: no es una reduccion de
  dimensionalidad ni una medida de calidad del modelo.

### 4.6 Nombrar los grupos
`nombrar_clusters()` ordena los grupos por su `intensidad_media` promedio: el mas
alto se llama **Alto**, el intermedio **Medio** y el mas bajo **Bajo**. Asi la
etiqueta es automatica y objetiva.

Salidas: `datos/municipios_segmentados.csv`, `datos/metricas_clustering.csv` y
`models/segmentacion.pkl`.

## 5. Interpretacion territorial

Resultado (k=3):

| Nivel | Municipios | Zona tipica |
|---|---|---|
| **Alto (25)** | Betulia, Zapatoca, San Joaquin, Matanza, Surata, Velez... | Centro-montanoso de Santander / area del **Nido Sismico** |
| **Medio (37)** | Bucaramanga, Floridablanca, Los Santos, Barrancabermeja, San Gil, Malaga... | Ciudades grandes y piedemonte |
| **Bajo (11)** | Sabana de Torres, Puerto Wilches, Cimitarra, Barbosa, Puerto Parra... | **Magdalena Medio** (tierras bajas, lejos de los epicentros andinos) |

**Caso Los Santos:** aunque esta sobre el Nido Sismico, queda en **Medio**. Tiene
4 registros con intensidades 7, 7, 4 y 3 (media 5.25, maxima 7): los sismos
lejanos de 2020 y 2023 se sintieron debil alli y bajan su promedio. Es un efecto
de usar la intensidad media con pocos eventos (ver Limitaciones).

La lectura es coherente con la geologia: los epicentros historicos fuertes estan
en la cordillera (Nido de Bucaramanga) y las tierras bajas del Magdalena Medio
reciben menor sacudimiento.

## 6. Limitaciones

- El catalogo del SGC es de **eventos historicos significativos**, no el listado
  denso mensual del Nido: solo 24 sismos y 73 municipios con datos.
- La intensidad EMS-98 historica puede tener sesgos de reporte.
- **Pocos eventos por municipio:** 16 municipios tienen un solo registro y 36
  tienen dos o menos. Su nivel depende de uno o dos reportes; la pestana
  "Municipio" del dashboard los marca como de baja confianza.
- 14 municipios de Santander no tienen registros y aparecen como "sin datos".
- El clustering es **descriptivo**: no indica cuando ocurrira un sismo.

## 7. Trabajo futuro

- Incorporar el catalogo denso del Nido (miles de eventos) para mayor robustez.
- Anadir exposicion (poblacion, infraestructura) para pasar de *peligro* a *riesgo* real.
- Comparar K-Means con DBSCAN o clustering jerarquico.
