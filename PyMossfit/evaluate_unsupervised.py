#!/usr/bin/env python3
"""Benchmark unsupervised topology + fitting methods on Mössbauer spectra."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyMossfit.pymossfit import (  # noqa: E402
  compare_unsupervised_methods,
  fit_spectrum,
  identify_phases,
  load_calibrated_csv,
)


def _load_parquet_samples(parquet_path: Path, n_samples: int, seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
  df = pd.read_parquet(parquet_path)
  rng = np.random.default_rng(seed)
  idx = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)
  samples = []
  for i in idx:
    row = df.iloc[i]
    v = np.array(row["velocity_uniform"], dtype=float)
    y = np.array(row["intensity_uniform"], dtype=float)
    label = str(row.get("pkey", i))
    samples.append((label, v, y))
  return samples


def main():
  parser = argparse.ArgumentParser(description="Compare unsupervised PyMossFit methods")
  parser.add_argument("--calib", default=str(Path(__file__).parent / "Calib-Fe.csv"))
  parser.add_argument("--parquet", default=str(ROOT / "outputs" / "mossbauer_processed.parquet"))
  parser.add_argument("--n-samples", type=int, default=5, help="Random spectra from parquet")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--n-starts", type=int, default=6, help="Multi-start lmfit attempts per topology")
  parser.add_argument("--out", default=str(ROOT / "outputs" / "results" / "pymossfit_unsupervised_benchmark.json"))
  args = parser.parse_args()

  all_results = []

  # 1) Calib-Fe reference spectrum (known sextet from manual PyMossFit fit)
  if Path(args.calib).exists():
    v, y = load_calibrated_csv(args.calib)
    print("=== Calib-Fe.csv ===")
    df_calib = compare_unsupervised_methods(v, y, n_random_starts=args.n_starts)
    print(df_calib.to_string(index=False))
    best = fit_spectrum(v, y, method="bic_grid", n_random_starts=args.n_starts)
    phases = identify_phases(best.report_rows)
    print(f"\nBest bic_grid topology: {best.topology.as_tuple()}, RMSE={best.rmse:.6f}")
    if phases:
      print("Top KNN match:", phases[0]["matches"][0]["compound"])
    all_results.append({"spectrum": "Calib-Fe", "comparison": df_calib.to_dict(orient="records")})

  # 2) Random samples from processed dataset
  if Path(args.parquet).exists() and args.n_samples > 0:
    samples = _load_parquet_samples(Path(args.parquet), args.n_samples, args.seed)
    summary_rows = []
    for name, v, y in samples:
      df_cmp = compare_unsupervised_methods(v, y, n_random_starts=args.n_starts)
      best_method = df_cmp.iloc[0]["method"] if len(df_cmp) else "bic_grid"
      summary_rows.append(
        {
          "pkey": name,
          "best_method": best_method,
          "best_rmse": float(df_cmp.iloc[0]["rmse"]) if len(df_cmp) else None,
          "methods": df_cmp.to_dict(orient="records"),
        }
      )
      print(f"\n=== {name} | best={best_method} rmse={df_cmp.iloc[0]['rmse']:.5f} ===")

    agg = []
    for row in summary_rows:
      for m in row["methods"]:
        agg.append({"pkey": row["pkey"], **m})
    if agg:
      df_agg = pd.DataFrame(agg)
      print("\n=== Mean RMSE by method (parquet sample) ===")
      print(df_agg.groupby("method")["rmse"].mean().sort_values().to_string())
      all_results.append({"parquet_summary": summary_rows, "mean_rmse_by_method": df_agg.groupby("method")["rmse"].mean().to_dict()})

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
  print(f"\nSaved benchmark to {out_path}")


if __name__ == "__main__":
  main()
