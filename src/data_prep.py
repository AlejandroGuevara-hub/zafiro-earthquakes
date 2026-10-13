"""Limpieza y preparacion de la historia sismica de Santander (SGC).

Entrada : datos/sismos_raw.csv   (ya viene filtrado a Santander por el SGC)
Salida  : datos/sismos_clean.csv (dataset limpio, con features y etiqueta)

Cada fila es un par (sismo, municipio afectado): el catalogo repite un mismo
sismo una vez por cada centro poblado donde se reporto intensidad.

Flujo:
    1. cargar_catalogo()   -> localiza el encabezado y lee el CSV crudo
    2. limpiar()           -> renombra columnas, decodifica HTML, castea tipos, depura
    3. agregar_features()  -> variables derivadas (tiempo, municipio, clase de profundidad)
    4. etiquetar_riesgo()  -> variable objetivo riesgo_sismico (Bajo / Medio / Alto)
"""

from __future__ import annotations

import html
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "datos" / "sismos_raw.csv"
OUT = BASE / "datos" / "sismos_clean.csv"

COLUMNAS = {
    "Fecha": "fecha",
    "Hora local": "hora",
    "Latitud epicentro": "latitud",
    "Longitud epicentro": "longitud",
    "Magnitud": "magnitud",
    "Tipo magnitud": "magnitud_tipo",
    "Profundidad (km)": "profundidad_km",
    "Centro poblado": "municipio",
    "Intensidad sitio (EMS-98)": "intensidad_sitio",
    "Intensidad maxima (EMS-98)": "intensidad_max",
}

NUMERICAS = ["latitud", "longitud", "magnitud", "profundidad_km"]

FEATURES = ["latitud", "longitud", "profundidad_km", "magnitud", "anio", "mes"]


def _clean_text(valor):
    """Normaliza texto: decodifica entidades HTML y colapsa espacios."""
    if pd.isna(valor):
        return np.nan
    texto = html.unescape(str(valor)).replace("\xa0", " ")
    texto = texto.strip().strip('"')
    return re.sub(r"\s+", " ", texto)


def _parse_intensidad(valor):
    """Convierte intensidades como '6' o '6/7' (rango) a un unico numero (media)."""
    if pd.isna(valor):
        return np.nan
    numeros = re.findall(r"\d+(?:\.\d+)?", str(valor))
    if not numeros:
        return np.nan
    return float(np.mean([float(n) for n in numeros]))


def cargar_catalogo(path: Path = RAW) -> pd.DataFrame:
    """Lee el CSV localizando la fila de encabezado (la que empieza con 'No.')."""
    lineas = path.read_text(encoding="utf-8").splitlines()
    encabezado = next(
        (i for i, linea in enumerate(lineas) if linea.lstrip('"').startswith("No.")),
        None,
    )
    if encabezado is None:
        raise ValueError(f"No se encontro el encabezado en {path}")
    return pd.read_csv(StringIO("\n".join(lineas[encabezado:])), dtype=str)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra, decodifica, castea y depura el catalogo crudo."""
    df = df.rename(columns=COLUMNAS)
    df = df[[c for c in COLUMNAS.values() if c in df.columns]].copy()

    for col in df.columns:
        df[col] = df[col].map(_clean_text)

    for col in NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["intensidad_sitio", "intensidad_max"]:
        df[col] = df[col].map(_parse_intensidad)

    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y/%m/%d", errors="coerce")

    df = df.dropna(subset=["latitud", "longitud", "magnitud", "fecha"])
    df = df[df["latitud"].between(-5, 14) & df["longitud"].between(-82, -66)]
    df = df.drop_duplicates(
        subset=["fecha", "latitud", "longitud", "magnitud", "municipio"]
    )
    return df.sort_values("fecha").reset_index(drop=True)


def _municipio_base(valor):
    """Extrae el municipio desde 'Centro poblado' con formato 'Vereda, Municipio-Santander'."""
    if pd.isna(valor):
        return np.nan
    partes = [p.strip() for p in str(valor).split(",")]
    ultimo = re.sub(r"\s*-\s*Santander$", "", partes[-1]).strip()
    if ultimo and ultimo.lower() != "santander":
        return ultimo
    return partes[0]


def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas para el modelo y la interpretacion territorial."""
    df = df.copy()
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["municipio_base"] = df["municipio"].map(_municipio_base)
    df["clase_profundidad"] = pd.cut(
        df["profundidad_km"],
        bins=[-1, 30, 70, 150, 1000],
        labels=["somero", "intermedio_superior", "intermedio", "profundo"],
    )
    return df


def etiquetar_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    """Define riesgo_sismico a partir de la intensidad EMS-98 en el sitio.

        Bajo  : intensidad <= 3   (sacudida debil)
        Medio : intensidad 4 - 5  (sacudida moderada)
        Alto  : intensidad >= 6   (sacudida fuerte, posible dano)

    La intensidad EMS-98 es el indicador estandar del efecto del sismo en un
    lugar y por eso es la medida mas directa del riesgo percibido en el municipio.
    """
    df = df.copy()
    condiciones = [df["intensidad_sitio"] <= 3, df["intensidad_sitio"] <= 5]
    df["riesgo_sismico"] = np.select(condiciones, ["Bajo", "Medio"], default="Alto")
    df.loc[df["intensidad_sitio"].isna(), "riesgo_sismico"] = np.nan
    return df


def preparar() -> pd.DataFrame:
    """Ejecuta el pipeline completo y exporta datos/sismos_clean.csv."""
    crudo = cargar_catalogo()
    limpio = limpiar(crudo)
    con_features = agregar_features(limpio)
    final = etiquetar_riesgo(con_features)
    final.to_csv(OUT, index=False)
    return final


if __name__ == "__main__":
    datos = preparar()
    eventos = datos.drop_duplicates(["fecha", "latitud", "longitud", "magnitud"])
    print(f"Filas (sismo-municipio): {len(datos)}")
    print(f"Sismos unicos: {len(eventos)}  |  municipios: {datos['municipio_base'].nunique()}")
    print(f"Rango temporal: {datos.anio.min()} - {datos.anio.max()}")
    print("\nDistribucion de riesgo_sismico:")
    print(datos["riesgo_sismico"].value_counts().sort_index().to_string())
    print("\nClase de profundidad:")
    print(datos["clase_profundidad"].value_counts().to_string())
    print("\nTop 5 municipios con mas registros:")
    print(datos["municipio_base"].value_counts().head(5).to_string())
    print("\nVista previa:")
    print(
        datos[
            [
                "fecha",
                "municipio_base",
                "latitud",
                "longitud",
                "profundidad_km",
                "magnitud",
                "intensidad_sitio",
                "intensidad_max",
                "riesgo_sismico",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )
    print(f"\nGuardado en: {OUT}")
