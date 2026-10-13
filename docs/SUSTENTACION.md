# Guia de sustentacion

Guion para defender el proyecto: pitch, demo, reparto por persona y preguntas
probables con sus respuestas.

## Pitch (1 frase)

> "Convertimos la historia sismica oficial del SGC sobre Santander en un mapa de
> riesgo territorial usando aprendizaje no supervisado: agrupamos los 73
> municipios con registros en niveles Alto, Medio y Bajo y explicamos por que."

## Mensajes clave (lo que NO se debe olvidar)

1. **Es Machine Learning real, no reglas fijas:** el modelo es `KMeans`, entrenado
   con datos oficiales.
2. **Es clasificacion, no prediccion:** agrupamos el presente/pasado; **no**
   pronosticamos cuando temblara.
3. **Dos fuentes oficiales:** SGC (sismos) y DANE (limites municipales).
4. **La interpretacion territorial es el valor:** el mapa muestra donde se
   concentra el riesgo y por que (Nido Sismico vs Magdalena Medio).

## Guion del demo (3-4 min)

1. `python src/data_prep.py` -> mostrar las 242 filas, 24 sismos, 73 municipios.
2. `python src/segmentar.py` -> mostrar el barrido de k (codo + silhouette) y los
   tamano de cada nivel.
3. `streamlit run src/app.py`:
   - Pestana **Mapa**: coropleta coloreada por nivel, tooltip por municipio,
     epicentros historicos encima.
   - Pestana **Analisis**: municipios por nivel, boxplot de intensidad, scatter PCA,
     tabla de k y ranking de municipios.

## Reparto sugerido (3 personas)

| Persona | Parte | Que dice |
|---|---|---|
| A | Datos (`data_prep.py`) | Fuente SGC, limpieza, features, el problema sismo-municipio. |
| B | Modelo (`segmentar.py`) | Por que no supervisado, StandardScaler, KMeans, k=3, silhouette, PCA. |
| C | Dashboard (`app.py`) | Mapa folium, join con GeoJSON del DANE, interpretacion territorial. |

## Numeros para citar

- 242 filas (sismo-municipio), 24 sismos unicos, 73 municipios, 1826-2023.
- Clustering: KMeans, k=3, `random_state=42` (reproducible).
- Silhouette (k=3) = 0.452; codo en k=3.
- PCA(2): 91.3% + 8.7% de varianza.
- Niveles: Alto 25 / Medio 37 / Bajo 11.

## Preguntas probables y respuestas

**¿Esto predice sismos?**
No. Es clasificacion no supervisada: agrupa municipios por su comportamiento
sismico historico. No estima cuando ni donde ocurrira el proximo sismo.

**¿Por que no usaron un modelo supervisado?**
No existe una etiqueta "verdadera" de riesgo; ademas el nivel pide clasificacion
e interpretacion. K-Means descubre los grupos a partir de los propios datos.

**¿Por que k=3 y no otro numero?**
Para obtener tres niveles interpretables (Alto/Medio/Bajo). El metodo del codo
cae fuerte en k=3 y el silhouette ahi es razonable (0.452).

**¿Que es el silhouette?**
Una metrica (-1 a 1) que mide que tan separados y compactos quedan los grupos.
Mas alto = mejor separacion.

**¿Por que estandarizan las variables?**
K-Means mide distancias; sin estandarizar, una variable con mayor rango
dominaria el agrupamiento. `StandardScaler` pone todo a la misma escala.

**¿Por que no usaron la frecuencia de eventos para agrupar?**
Porque el numero de reportes esta confundido con el tamano del municipio
(las ciudades reportan mas). Usamos severidad (`intensidad_media/max`), que es el
peligro real; la frecuencia se conserva como dato descriptivo.

**¿Que es la intensidad EMS-98?**
Una escala que mide el **efecto del sismo en un lugar** (no la energia). Va de I
a XII; cuanto mas alta, mas sacudida y posible dano.

**¿De donde salen los datos?**
Sismos: Servicio Geologico Colombiano (SGC). Limites: DANE (Marco Geoestadistico
Nacional). Ambas fuentes oficiales.

**¿Por que hay municipios sin color en el mapa?**
14 municipios de Santander no tienen registros en el catalogo; se muestran como
"sin datos".

**¿La validacion como se hizo?**
Con silhouette y con coherencia geologica: los municipios de riesgo Alto caen en
la zona del Nido Sismico de Bucaramanga y los Bajo en el Magdalena Medio.

**¿Como garantizan reproducibilidad?**
`random_state=42` y `n_init=10` en KMeans; el pipeline es determinista.

**¿Esto sirve para gestion del riesgo?**
Si: prioriza los municipios que requieren estudios detallados, mitigacion y
planes de respuesta.

## Cierre

El proyecto entrega un clasificador funcional (K-Means) con datos oficiales y una
**lectura territorial accionable** del riesgo sismico de Santander, sin depender
de reglas fijas ni de pronosticos.
