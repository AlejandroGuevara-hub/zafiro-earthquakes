# Zafiro Earthquakes - Clasificacion del riesgo sismico de Santander

Proyecto para la **Hackathon UTS**. Convierte los datos sismicos oficiales del
Servicio Geologico Colombiano (SGC) sobre **Santander** en una **clasificacion
territorial del riesgo sismico**, usando **Machine Learning no supervisado**
(K-Means) y un dashboard interactivo en Streamlit.

- **Zona de estudio:** departamento de Santander (enfoque en el area del Nido
  Sismico de Bucaramanga / Mesa de Los Santos).
- **Nivel:** intermedio. Se realiza **clasificacion + interpretacion territorial**
  (no se pronostica el futuro).
- **Componente obligatorio:** un sistema de ML funcional entrenado con datos
  reales de fuentes oficiales (SGC y DANE).

## Fuentes de datos (oficiales)

| Dato | Fuente | Uso |
|---|---|---|
| Historia sismica de Santander | Servicio Geologico Colombiano (SGC) | `datos/sismos_raw.csv` |
| Limites municipales | DANE - Marco Geoestadistico Nacional | `datos/municipios_santander.geojson` |

`sismos_raw.csv` describe la historia sismica del departamento: cada fila es un
par **(sismo, municipio afectado)** con su intensidad EMS-98, magnitud y
profundidad del evento.

## Estructura del repositorio

```
zafiro-earthquakes/
├── datos/
│   ├── sismos_raw.csv               # dato oficial SGC (entrada, ya filtrado a Santander)
│   ├── sismos_clean.csv             # generado por data_prep.py
│   ├── municipios_santander.geojson # limites municipales (DANE, simplificado)
│   ├── municipios_segmentados.csv   # generado por segmentar.py (resultado del clustering)
│   └── metricas_clustering.csv      # barrido de k (codo / silhouette)
├── src/
│   ├── data_prep.py                 # limpieza y features (Paso 1)
│   ├── geo_prep.py                  # descarga/simplifica el GeoJSON del DANE (Paso 2)
│   ├── segmentar.py                 # clustering KMeans + PCA (Paso 3)
│   └── app.py                       # dashboard Streamlit (Paso 4)
├── models/
│   └── segmentacion.pkl             # scaler + kmeans + pca entrenados
├── docs/
│   ├── METODOLOGIA.md               # que se hizo y por que
│   ├── LIBRERIAS.md                 # proposito de cada libreria y funcion
│   └── SUSTENTACION.md              # guion de defensa y preguntas
├── requirements.txt
└── README.md
```

## Instalacion y ejecucion

```bash
pip install -r requirements.txt

# 1) Limpieza y variables derivadas
python src/data_prep.py

# 2) (Opcional) regenerar el GeoJSON de municipios desde el DANE
python src/geo_prep.py

# 3) Modelo de clustering
python src/segmentar.py

# 4) Dashboard
streamlit run src/app.py
```

## Resultados

- **242** filas (sismo-municipio), **24** sismos unicos, **73** municipios con registros.
- Rango temporal: **1826 - 2023**.
- Niveles de riesgo (KMeans, k=3): **Alto 25 / Medio 37 / Bajo 11** municipios.
- Silhouette (k=3): **0.452**. PCA(2) explica **~100%** de la varianza
  (91.3% + 8.7%).
- Lectura territorial: los municipios de **riesgo Alto** se concentran en el
  centro-montanoso de Santander (area del Nido: Betulia, Zapatoca, Los Santos,
  San Joaquin, etc.); los de **riesgo Bajo** estan en el Magdalena Medio
  (Sabana de Torres, Puerto Wilches, Cimitarra...).

## Documentacion

- [docs/METODOLOGIA.md](docs/METODOLOGIA.md)
- [docs/LIBRERIAS.md](docs/LIBRERIAS.md)
- [docs/SUSTENTACION.md](docs/SUSTENTACION.md)
