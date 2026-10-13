"""Dashboard Streamlit: mapa territorial + interpretacion del clustering.

Pestanas:
    Mapa    -> coropleta de municipios de Santander coloreada por nivel de riesgo,
               con capa opcional de epicentros historicos.
    Analisis-> tamanos de cluster, boxplot de intensidad, scatter PCA,
               barrido de k (codo/silhouette) y ranking de municipios.
    Municipio-> ficha de un municipio: nivel, por que quedo ahi e historial de sismos.
    Simulador-> clasifica un caso nuevo con el modelo entrenado (segmentacion.pkl).

Ejecutar: streamlit run src/app.py
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import folium
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_folium import st_folium

BASE = Path(__file__).resolve().parents[1]
RUTA_MUN = BASE / "datos" / "municipios_segmentados.csv"
RUTA_SISMOS = BASE / "datos" / "sismos_clean.csv"
RUTA_GEO = BASE / "datos" / "municipios_santander.geojson"
RUTA_METRICAS = BASE / "datos" / "metricas_clustering.csv"
RUTA_MODELO = BASE / "models" / "segmentacion.pkl"

ORDEN = ["Alto", "Medio", "Bajo"]
COLORES = {"Alto": "#d73027", "Medio": "#fee08b", "Bajo": "#1a9850"}
RIESGO_NUM = {"Bajo": 0, "Medio": 1, "Alto": 2}
CENTRO_SANTANDER = [6.6, -73.2]
MIN_EVENTOS_CONFIABLE = 3


def normalizar(texto: str) -> str:
    """Mayusculas sin tildes para unir nombres del dataset con el GeoJSON del DANE."""
    limpio = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return limpio.upper().strip()


@st.cache_data
def cargar():
    """Carga el resultado del clustering, los sismos y el GeoJSON (con cache)."""
    municipios = pd.read_csv(RUTA_MUN)
    sismos = pd.read_csv(RUTA_SISMOS)
    metricas = pd.read_csv(RUTA_METRICAS)
    geo = json.loads(RUTA_GEO.read_text(encoding="utf-8"))
    return municipios, sismos, metricas, geo


@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado (scaler + kmeans + pca) generado por segmentar.py."""
    return joblib.load(RUTA_MODELO)


def niveles_por_cluster(municipios) -> dict:
    """Relaciona cada cluster de KMeans con su nivel (Alto / Medio / Bajo)."""
    pares = municipios.drop_duplicates("cluster")
    return {int(c): n for c, n in zip(pares["cluster"], pares["nivel_riesgo"])}


def clasificar(modelo, niveles, intensidad_media, intensidad_max):
    """Aplica scaler + KMeans a un caso y devuelve (nivel, tabla de centroides y distancias)."""
    Xs = modelo["scaler"].transform([[intensidad_media, intensidad_max]])
    cluster = int(modelo["kmeans"].predict(Xs)[0])
    distancias = modelo["kmeans"].transform(Xs)[0]
    centroides = modelo["scaler"].inverse_transform(modelo["kmeans"].cluster_centers_)
    tabla = pd.DataFrame(
        {
            "nivel": [niveles[c] for c in range(len(distancias))],
            "centroide_intensidad_media": centroides[:, 0].round(2),
            "centroide_intensidad_max": centroides[:, 1].round(2),
            "distancia": distancias.round(3),
        }
    )
    return niveles[cluster], tabla.set_index("nivel").reindex(ORDEN).reset_index()


def construir_mapa(municipios, sismos, geo, ver_epicentros):
    """Construye el mapa folium con la coropleta por nivel y epicentros opcionales."""
    municipios = municipios.copy()
    municipios["clave"] = municipios["municipio"].map(normalizar)
    municipios["riesgo_num"] = municipios["nivel_riesgo"].map(RIESGO_NUM)

    detalle = municipios.set_index("clave")
    for feature in geo["features"]:
        props = feature["properties"]
        clave = normalizar(props.get("municipio", ""))
        props["clave"] = clave
        fila = detalle.loc[clave] if clave in detalle.index else None
        if fila is not None:
            props["nivel_riesgo"] = fila["nivel_riesgo"]
            props["intensidad_media"] = round(float(fila["intensidad_media"]), 1)
            props["intensidad_max"] = int(fila["intensidad_max"])
            props["n_eventos"] = int(fila["n_eventos"])
        else:
            props["nivel_riesgo"] = "Sin datos"
            props["intensidad_media"] = None
            props["intensidad_max"] = None
            props["n_eventos"] = None

    mapa = folium.Map(location=CENTRO_SANTANDER, zoom_start=8, tiles="OpenStreetMap")

    folium.Choropleth(
        geo_data=geo,
        data=municipios,
        columns=["clave", "riesgo_num"],
        key_on="feature.properties.clave",
        fill_color="RdYlGn_r",
        fill_opacity=0.7,
        nan_fill_color="#f0f0f0",
        nan_fill_opacity=0.4,
        line_opacity=0.4,
        bins=[-0.5, 0.5, 1.5, 2.5],
        legend_name="Nivel de riesgo sismico (0=Bajo, 2=Alto)",
    ).add_to(mapa)

    folium.GeoJson(
        geo,
        style_function=lambda _: {"fillOpacity": 0, "weight": 0.4, "color": "#555"},
        tooltip=folium.GeoJsonTooltip(
            fields=["municipio", "nivel_riesgo", "intensidad_media", "intensidad_max", "n_eventos"],
            aliases=["Municipio:", "Nivel:", "Intensidad media:", "Intensidad max.:", "Eventos:"],
            localize=True,
        ),
    ).add_to(mapa)

    if ver_epicentros:
        epicentros = sismos.drop_duplicates(["fecha", "latitud", "longitud", "magnitud"])
        for _, ev in epicentros.iterrows():
            folium.CircleMarker(
                location=[ev["latitud"], ev["longitud"]],
                radius=3,
                color="#2c3e50",
                fill=True,
                fill_opacity=0.8,
                popup=f"M {ev['magnitud']} - {ev['fecha']}",
            ).add_to(mapa)

    return mapa


def contar_coincidencias(municipios, geo) -> int:
    """Cuenta municipios del dataset que se pudieron cruzar con el GeoJSON."""
    claves_dataset = set(municipios["municipio"].map(normalizar))
    claves_geo = {normalizar(f["properties"].get("municipio", "")) for f in geo["features"]}
    return len(claves_dataset & claves_geo)


def pestana_mapa(municipios, sismos, geo):
    st.subheader("Mapa de riesgo sismico territorial")
    if contar_coincidencias(municipios, geo) == 0:
        st.error(
            "No se pudieron cruzar los municipios con el GeoJSON, por eso el mapa "
            "no colorea. Regenera el archivo con: python src/geo_prep.py"
        )
    ver_epicentros = st.checkbox("Mostrar epicentros historicos", value=True)
    st_folium(construir_mapa(municipios, sismos, geo, ver_epicentros), height=560, use_container_width=True)


def pestana_analisis(municipios, metricas, n_sismos, silhouette):
    col1, col2, col3 = st.columns(3)
    col1.metric("Municipios analizados", len(municipios))
    col2.metric("Sismos unicos", n_sismos)
    col3.metric("Silhouette (k=3)", f"{silhouette:.3f}")

    izq, der = st.columns(2)
    with izq:
        st.markdown("**Municipios por nivel de riesgo**")
        conteo = municipios["nivel_riesgo"].value_counts().reindex(ORDEN)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=conteo.index, y=conteo.values, hue=conteo.index, palette=COLORES, legend=False, ax=ax)
        ax.set_ylabel("Numero de municipios")
        ax.set_xlabel("")
        st.pyplot(fig)
    with der:
        st.markdown("**Intensidad media segun nivel**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=municipios, x="nivel_riesgo", y="intensidad_media", order=ORDEN, hue="nivel_riesgo", palette=COLORES, legend=False, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Intensidad media (EMS-98)")
        st.pyplot(fig)

    st.markdown("**Grupos en el plano PCA (2 componentes)**")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=municipios, x="pca1", y="pca2", hue="nivel_riesgo", palette=COLORES, hue_order=ORDEN, s=70, ax=ax)
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    st.pyplot(fig)

    st.markdown("**Barrido de k (metodo del codo y silhouette)**")
    st.dataframe(metricas, width="stretch")

    st.markdown("**Ranking de municipios por nivel**")
    tabla = municipios.sort_values(["nivel_riesgo", "intensidad_max"], ascending=[True, False])
    st.dataframe(
        tabla[["municipio", "nivel_riesgo", "intensidad_media", "intensidad_max", "n_eventos", "magnitud_max"]],
        width="stretch",
        hide_index=True,
    )


def pestana_municipio(municipios, sismos, modelo, niveles):
    st.subheader("Ficha por municipio")
    nombre = st.selectbox("Municipio", sorted(municipios["municipio"]))
    fila = municipios.loc[municipios["municipio"] == nombre].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nivel de riesgo", fila["nivel_riesgo"])
    col2.metric("Eventos registrados", int(fila["n_eventos"]))
    col3.metric("Intensidad media (EMS-98)", f"{fila['intensidad_media']:.1f}")
    col4.metric("Intensidad maxima (EMS-98)", f"{fila['intensidad_max']:.0f}")

    if fila["n_eventos"] < MIN_EVENTOS_CONFIABLE:
        st.warning(
            f"Baja confianza: solo {int(fila['n_eventos'])} evento(s) registrado(s). "
            "El nivel puede cambiar mucho con un solo reporte adicional."
        )

    _, distancias = clasificar(modelo, niveles, fila["intensidad_media"], fila["intensidad_max"])
    st.markdown("**Por que quedo en este nivel**")
    st.caption(
        "KMeans asigna el municipio al centroide mas cercano (menor distancia) "
        "en el espacio estandarizado de intensidad media y maxima."
    )
    st.dataframe(distancias, width="stretch", hide_index=True)

    st.markdown("**Historial de sismos sentidos en el municipio**")
    historial = sismos.loc[sismos["municipio_base"] == nombre].sort_values("fecha", ascending=False)
    st.dataframe(
        historial[["fecha", "municipio", "magnitud", "profundidad_km", "intensidad_sitio", "intensidad_max"]].rename(
            columns={
                "municipio": "centro_poblado",
                "intensidad_sitio": "intensidad_en_el_sitio",
                "intensidad_max": "intensidad_max_del_sismo",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def pestana_simulador(municipios, modelo, niveles):
    st.subheader("Simulador: clasifica un caso con el modelo entrenado")
    st.caption(
        "Usa el KMeans guardado en models/segmentacion.pkl: estandariza el caso "
        "y le asigna el nivel del centroide mas cercano."
    )

    col1, col2 = st.columns(2)
    intensidad_max = col1.slider("Intensidad maxima registrada (EMS-98)", 1.0, 12.0, 6.0, 0.5)
    intensidad_media = col2.slider("Intensidad media (EMS-98)", 1.0, 12.0, 5.0, 0.1)
    if intensidad_media > intensidad_max:
        st.warning("La intensidad media no puede superar la maxima; se usa el valor de la maxima.")
        intensidad_media = intensidad_max
    if not municipios["intensidad_max"].min() <= intensidad_max <= municipios["intensidad_max"].max():
        st.info(
            "Este caso esta fuera del rango con el que se entreno el modelo "
            f"(intensidad maxima entre {municipios['intensidad_max'].min():.0f} y "
            f"{municipios['intensidad_max'].max():.0f}); el resultado es una extrapolacion."
        )

    nivel, distancias = clasificar(modelo, niveles, intensidad_media, intensidad_max)
    st.metric("Nivel asignado por el modelo", nivel)

    izq, der = st.columns(2)
    with izq:
        st.markdown("**Distancia a cada centroide**")
        st.dataframe(distancias, width="stretch", hide_index=True)
    with der:
        st.markdown("**El caso frente a los municipios reales**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(data=municipios, x="intensidad_media", y="intensidad_max", hue="nivel_riesgo", palette=COLORES, hue_order=ORDEN, s=50, ax=ax)
        ax.scatter([intensidad_media], [intensidad_max], marker="*", s=260, color="black", label="Caso simulado")
        ax.set_xlabel("Intensidad media (EMS-98)")
        ax.set_ylabel("Intensidad maxima (EMS-98)")
        ax.legend(title="")
        st.pyplot(fig)


def main():
    st.set_page_config(page_title="Riesgo sismico de Santander", layout="wide")
    st.title("Clasificacion del riesgo sismico de Santander")
    st.caption("Datos oficiales: Servicio Geologico Colombiano | Limites: DANE (MGN) | Modelo: KMeans no supervisado")

    municipios, sismos, metricas, geo = cargar()
    n_sismos = sismos.drop_duplicates(["fecha", "latitud", "longitud", "magnitud"]).shape[0]
    fila_k3 = metricas.loc[metricas["k"] == 3, "silhouette"]
    silhouette = float(fila_k3.iloc[0]) if not fila_k3.empty else float("nan")

    modelo = cargar_modelo()
    niveles = niveles_por_cluster(municipios)

    tab_mapa, tab_analisis, tab_municipio, tab_simulador = st.tabs(["Mapa", "Analisis", "Municipio", "Simulador"])
    with tab_mapa:
        pestana_mapa(municipios, sismos, geo)
    with tab_analisis:
        pestana_analisis(municipios, metricas, n_sismos, silhouette)
    with tab_municipio:
        pestana_municipio(municipios, sismos, modelo, niveles)
    with tab_simulador:
        pestana_simulador(municipios, modelo, niveles)


if __name__ == "__main__":
    main()
