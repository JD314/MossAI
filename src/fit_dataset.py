"""
Persistencia y lectura del dataset de ajustes Mössbauer (modo básico vs combinaciones).

Cada espectro genera dos registros:
  - allow_combinations=False  → solo (1,0,0), (0,1,0), (0,2,0), (0,0,1)
  - allow_combinations=True   → básicos + mezclas
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FIT_DATASET = Path("outputs/mossbauer_fits_dual.parquet")

_JSON_COLUMNS = ("hyperfine_params", "report_rows", "areas", "best_fit")

__all__ = [
  "DEFAULT_FIT_DATASET",
  "save_fit_dataset",
  "load_fit_dataset",
  "get_spectrum_fits",
  "get_best_fit",
  "compare_fit_modes",
  "hyperfine_table",
  "summary_by_mode",
]


def _serialize_json_columns(df: pd.DataFrame) -> pd.DataFrame:
  out = df.copy()
  for col in _JSON_COLUMNS:
    if col not in out.columns:
      continue
    out[col] = out[col].apply(
      lambda v: json.dumps(v, ensure_ascii=False) if v is not None and v == v else None
    )
  return out


def _deserialize_json_columns(df: pd.DataFrame) -> pd.DataFrame:
  out = df.copy()
  for col in _JSON_COLUMNS:
    if col not in out.columns:
      continue

    def _loads(v: Any) -> Any:
      if v is None or (isinstance(v, float) and np.isnan(v)):
        return None if col == "best_fit" else ([] if col == "report_rows" else {})
      if isinstance(v, (list, dict)):
        return v
      return json.loads(v)

    out[col] = out[col].apply(_loads)
  return out


def save_fit_dataset(records: list[dict], path: str | Path = DEFAULT_FIT_DATASET) -> Path:
  """Guarda el dataset de ajustes (2 filas por espectro) en parquet."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  df = _serialize_json_columns(pd.DataFrame(records))
  df.to_parquet(path, index=False)
  return path


def load_fit_dataset(path: str | Path = DEFAULT_FIT_DATASET) -> pd.DataFrame:
  """Carga el dataset de ajustes y restaura columnas JSON."""
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"No se encontró el dataset de ajustes: {path}")
  return _deserialize_json_columns(pd.read_parquet(path))


def get_spectrum_fits(
  pkey: str,
  path: str | Path = DEFAULT_FIT_DATASET,
) -> pd.DataFrame:
  """Devuelve los dos ajustes (básico y con combinaciones) de un pkey."""
  df = load_fit_dataset(path)
  key = str(pkey)
  out = df[df["pkey"].astype(str) == key].copy()
  if out.empty:
    raise KeyError(f"pkey={key!r} no encontrado en {path}")
  return out.sort_values("allow_combinations").reset_index(drop=True)


def get_best_fit(
  pkey: str,
  allow_combinations: bool = False,
  path: str | Path = DEFAULT_FIT_DATASET,
) -> pd.Series:
  """Un registro de ajuste para un pkey y modo."""
  fits = get_spectrum_fits(pkey, path=path)
  row = fits[fits["allow_combinations"] == allow_combinations]
  if row.empty:
    raise KeyError(
      f"No hay ajuste para pkey={pkey!r} con allow_combinations={allow_combinations}"
    )
  return row.iloc[0]


def compare_fit_modes(
  pkey: str,
  path: str | Path = DEFAULT_FIT_DATASET,
) -> pd.DataFrame:
  """Compara lado a lado básico vs combinaciones para un espectro."""
  fits = get_spectrum_fits(pkey, path=path)
  cols = [
    "allow_combinations",
    "fit_ok",
    "n_s",
    "n_d",
    "n_x",
    "fit_rmse",
    "fit_bic",
    "fit_method",
  ]
  return fits[cols]


def hyperfine_table(path: str | Path = DEFAULT_FIT_DATASET) -> pd.DataFrame:
  """
  Tabla larga: una fila por fase/subspectro con IS, QS, Bhf y área.

  Columnas: pkey, allow_combinations, phase, type, IS, QS, Bhf, area_pct, ...
  """
  df = load_fit_dataset(path)
  rows: list[dict] = []
  for _, rec in df.iterrows():
    if not rec.get("fit_ok"):
      continue
    for phase in rec.get("report_rows") or []:
      rows.append(
        {
          "pkey": rec["pkey"],
          "allow_combinations": rec["allow_combinations"],
          "topology": (rec["n_s"], rec["n_d"], rec["n_x"]),
          "fit_rmse": rec["fit_rmse"],
          "phase": phase.get("Phase"),
          "type": phase.get("Type"),
          "IS_mm_s": phase.get("IS (mm/s)"),
          "quad_splitting_mm_s": phase.get("Quad Splitting (mm/s)"),
          "bhf_T": phase.get("Bhf (T)"),
          "linewidth_mm_s": phase.get("Width (mm/s)"),
          "amplitude": phase.get("Amplitude"),
          "area_pct": phase.get("Area (%)"),
        }
      )
  return pd.DataFrame(rows)


def summary_by_mode(path: str | Path = DEFAULT_FIT_DATASET) -> pd.DataFrame:
  """Resumen agregado por modo (allow_combinations)."""
  df = load_fit_dataset(path)
  ok = df[df["fit_ok"] == True]
  rows = []
  for allow, grp in ok.groupby("allow_combinations"):
    rows.append(
      {
        "allow_combinations": bool(allow),
        "n_spectra": int(len(grp)),
        "rmse_mean": float(grp["fit_rmse"].mean()),
        "rmse_median": float(grp["fit_rmse"].median()),
        "topologies": grp[["n_s", "n_d", "n_x"]]
        .astype(int)
        .apply(tuple, axis=1)
        .value_counts()
        .head(10)
        .to_dict(),
      }
    )
  return pd.DataFrame(rows)
