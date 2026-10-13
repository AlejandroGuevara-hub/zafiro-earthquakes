"""Segmentacion no supervisada de municipios de Santander por riesgo sismico.

Modelo NO predictivo (aprendizaje no supervisado):
    1. agregar_por_municipio() -> resume cada municipio con variables de frecuencia y severidad
    2. evaluar_k()             -> barrido de k con inercia (codo) y silhouette
    3. segmentar()             -> StandardScaler + KMeans(k=3) + PCA(2)
    4. Nombrar grupos por su intensidad media: Alto / Medio / Bajo

Entrada : datos/sismos_clean.csv
Salidas : datos/municipios_segmentados.csv  (municipio, features, cluster, nivel, PCA)
          datos/metricas_clustering.csv     (k, inercia, silhouette)
          models/segmentacion.pkl           (scaler + kmeans + pca)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
ENTRADA = BASE / "datos" / "sismos_clean.csv"
SAL_MUNICIPIOS = BASE / "datos" / "municipios_segmentados.csv"
SAL_METRICAS = BASE / "datos" / "metricas_clustering.csv"
SAL_MODELO = BASE / "models" / "segmentacion.pkl"

FEATURES = ["intensidad_media", "intensidad_max"]
NIVELES = {0: "Alto", 1: "Medio", 2: "Bajo"}
SEMILLA = 42


def agregar_por_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """Resume cada municipio con frecuencia (n_eventos) y severidad (intensidad/magnitud)."""
    resumen = (
        df.groupby("municipio_base")
        .agg(
            n_eventos=("riesgo_sismico", "size"),
            intensidad_media=("intensidad_sitio", "mean"),
            intensidad_max=("intensidad_sitio", "max"),
            magnitud_max=("magnitud", "max"),
            profundidad_media=("profundidad_km", "mean"),
        )
        .reset_index()
        .rename(columns={"municipio_base": "municipio"})
    )
    return resumen


def evaluar_k(Xs: np.ndarray, ks=range(2, 7)) -> pd.DataFrame:
    """Calcula inercia (metodo del codo) y silhouette para varios k."""
    filas = []
    for k in ks:
        modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(Xs)
        filas.append(
            {
                "k": k,
                "inercia": round(float(modelo.inertia_), 3),
                "silhouette": round(float(silhouette_score(Xs, modelo.labels_)), 4),
            }
        )
    return pd.DataFrame(filas)


def nombrar_clusters(municipios: pd.DataFrame, k: int) -> dict:
    """Ordena los clusters por intensidad media: el mas alto = Alto, el mas bajo = Bajo."""
    orden = (
        municipios.groupby("cluster")["intensidad_media"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    etiquetas = ["Alto", "Medio", "Bajo"][:k]
    return {cluster: etiqueta for cluster, etiqueta in zip(orden, etiquetas)}


def segmentar(k: int = 3) -> pd.DataFrame:
    """Ejecuta el pipeline de clustering y exporta resultados y modelo."""
    crudo = pd.read_csv(ENTRADA)
    municipios = agregar_por_municipio(crudo)

    X = municipios[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    metricas = evaluar_k(Xs)
    metricas.to_csv(SAL_METRICAS, index=False)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(Xs)
    municipios["cluster"] = kmeans.labels_
    municipios["nivel_riesgo"] = municipios["cluster"].map(nombrar_clusters(municipios, k))

    pca = PCA(n_components=2, random_state=SEMILLA).fit(Xs)
    componentes = pca.transform(Xs)
    municipios["pca1"] = componentes[:, 0].round(4)
    municipios["pca2"] = componentes[:, 1].round(4)

    municipios = municipios.sort_values(
        ["nivel_riesgo", "intensidad_max"], ascending=[True, False]
    ).reset_index(drop=True)
    municipios.to_csv(SAL_MUNICIPIOS, index=False)

    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "pca": pca,
            "features": FEATURES,
            "silhouette": float(silhouette_score(Xs, kmeans.labels_)),
            "varianza_explicada": pca.explained_variance_ratio_.round(4).tolist(),
        },
        SAL_MODELO,
    )
    return municipios


if __name__ == "__main__":
    resultado = segmentar(k=3)
    print("=== Barrido de k ===")
    print(pd.read_csv(SAL_METRICAS).to_string(index=False))
    print("\n=== Tamano de cada nivel ===")
    print(resultado["nivel_riesgo"].value_counts().reindex(["Alto", "Medio", "Bajo"]).to_string())
    modelo = joblib.load(SAL_MODELO)
    print(f"\nSilhouette (k=3): {modelo['silhouette']:.4f}")
    print(f"Varianza explicada por PCA(2): {modelo['varianza_explicada']}")
    print("\n=== Municipios por nivel ===")
    for nivel in ["Alto", "Medio", "Bajo"]:
        lista = resultado.loc[resultado["nivel_riesgo"] == nivel, "municipio"].tolist()
        print(f"\n{nivel} ({len(lista)}): {', '.join(lista)}")
    print(f"\nGuardado en: {SAL_MUNICIPIOS}")
