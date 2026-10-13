"""Descarga y prepara los limites municipales de Santander (DANE).

Fuente oficial : DANE - Marco Geoestadistico Nacional, publicada en el portal
                 geografico del DANE (cdge.dane.gov.co). Se consulta el servicio
                 ArcGIS del DANE filtrando el departamento de Santander
                 (COD_DPTO = 68). Norte de Santander es 54, por lo que no se mezcla.

Entrada : servicio ArcGIS del DANE (red) o el GeoJSON ya descargado
Salida  : datos/municipios_santander.geojson (municipios de Santander, liviano)
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SALIDA = BASE / "datos" / "municipios_santander.geojson"

COD_DEPTO = "68"
SERVICIO = (
    "https://portalgis.dane.gov.co/mparcgis/rest/services/"
    "INDICADORES_CTERRITORIO/Cache_MpiosCTFechaCreacion_2012/MapServer/3/query"
)
PARAMS = (
    f"where=COD_DPTO%3D%27{COD_DEPTO}%27"
    "&outFields=COD_MPIO,NOM_MPIO,COD_DPTO,NOM_DPTO"
    "&returnGeometry=true&outSR=4326&f=geojson"
)


def descargar() -> dict:
    """Consulta el servicio del DANE y devuelve el GeoJSON de Santander."""
    with urllib.request.urlopen(f"{SERVICIO}?{PARAMS}", timeout=120) as respuesta:
        return json.loads(respuesta.read().decode("utf-8"))


def cargar() -> dict:
    """Usa el GeoJSON local si existe; si no, lo descarga."""
    if SALIDA.exists():
        return json.loads(SALIDA.read_text(encoding="utf-8"))
    return descargar()


def _redondear(valor, decimales: int = 4):
    """Redondea recursivamente las coordenadas para reducir el peso del archivo."""
    if isinstance(valor, list):
        return [_redondear(v, decimales) for v in valor]
    if isinstance(valor, float):
        return round(valor, decimales)
    return valor


def simplificar(geojson: dict, decimales: int = 4) -> dict:
    """Conserva solo las propiedades utiles y reduce la precision de las coordenadas."""
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        feature["properties"] = {
            "codigo": props.get("COD_MPIO"),
            "municipio": props.get("NOM_MPIO"),
        }
        geometria = feature.get("geometry")
        if geometria and "coordinates" in geometria:
            geometria["coordinates"] = _redondear(geometria["coordinates"], decimales)
    return geojson


if __name__ == "__main__":
    geojson = simplificar(cargar())
    SALIDA.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    peso_mb = SALIDA.stat().st_size / 1_048_576
    print(f"Municipios de Santander: {len(geojson['features'])}")
    print(f"Guardado en: {SALIDA} ({peso_mb:.2f} MB)")
