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


def _tiene_municipios(geojson: dict) -> bool:
    """True si las features traen un nombre de municipio util (no None)."""
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        if props.get("NOM_MPIO") or props.get("municipio"):
            return True
    return False


def cargar() -> dict:
    """Usa el GeoJSON local si es valido; si esta vacio/corrupto, lo descarga."""
    if SALIDA.exists():
        local = json.loads(SALIDA.read_text(encoding="utf-8"))
        if _tiene_municipios(local):
            return local
    return descargar()


def _redondear(valor, decimales: int = 4):
    """Redondea recursivamente las coordenadas para reducir el peso del archivo."""
    if isinstance(valor, list):
        return [_redondear(v, decimales) for v in valor]
    if isinstance(valor, float):
        return round(valor, decimales)
    return valor


def simplificar(geojson: dict, decimales: int = 4) -> dict:
    """Conserva solo las propiedades utiles y reduce la precision de las coordenadas.

    Es idempotente: acepta tanto las claves originales del DANE (COD_MPIO/NOM_MPIO)
    como las ya simplificadas (codigo/municipio), sin sobrescribir con None.
    """
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        municipio = props.get("NOM_MPIO") or props.get("municipio")
        codigo = props.get("COD_MPIO") or props.get("codigo")
        feature["properties"] = {"codigo": codigo, "municipio": municipio}
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
