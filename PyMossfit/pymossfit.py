"""
Automated unsupervised Mössbauer spectrum fitting (PyMossFit conventions).

Input:  calibrated velocity and transmission intensity arrays.
Output: fitted spectrum, subspectra, hyperfine parameters — no P_vec, no labels.

Several unsupervised topology predictors estimate (n_s, n_d, n_x); lmfit refines
hyperfine parameters via multi-start minimization.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import lmfit
import numpy as np
import pandas as pd
from lmfit import Model, Parameters, minimize
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

# --- PyMossFit line-shape constants -------------------------------------------------

POSITIONS_D = np.array([-1.0, 1.0]) / 2.0
INTENSITIES_D = np.array([1.0, 1.0])
INTENSITIES_X = np.array([3.0, 2.0, 1.0, 1.0, 2.0, 3.0])


def lorentzian(x, amplitude, center, width):
    """Singlet Lorentzian (absorption component, positive upward)."""
    return amplitude * (2.0 * width / (np.pi * ((x - center) ** 2 + width**2)))


def doublet_lorentzian(x, delta, quad, gamma, scale):
  """Doublet for quadrupolar splitting."""
  y = np.zeros_like(x, dtype=float)
  for pos, inten in zip(POSITIONS_D, INTENSITIES_D):
    center = delta + pos * quad
    y += lorentzian(x, scale * inten, center, gamma)
  return y


def sextet_lorentzian(x, delta, q_shift, B_hf, gamma, scale):
  """Sextet for magnetic hyperfine splitting."""
  y = np.zeros_like(x, dtype=float)
  d = q_shift / B_hf if B_hf != 0 else 0.0
  positions_x = np.array([-1.0, -3 / 5 + d, -1 / 5, 1 / 5, 3 / 5 - d, 1.0])
  for pos, inten in zip(positions_x, INTENSITIES_X):
    center = delta + pos * B_hf
    y += lorentzian(x, scale * inten, center, gamma)
  return y


# --- Data structures ----------------------------------------------------------------

@dataclass(frozen=True)
class Topology:
  n_s: int
  n_d: int
  n_x: int

  @property
  def n_components(self) -> int:
    return self.n_s + self.n_d + self.n_x

  def as_tuple(self) -> tuple[int, int, int]:
    return (self.n_s, self.n_d, self.n_x)


@dataclass
class FitResult:
  topology: Topology
  velocity: np.ndarray
  intensity: np.ndarray
  best_fit: np.ndarray
  component_fits: dict[str, np.ndarray]
  params: lmfit.Parameters
  rmse: float
  mean_rel_error: float
  bic: float
  method: str
  areas: dict[str, float] = field(default_factory=dict)
  report_rows: list[dict] = field(default_factory=list)


# --- Topology catalogue -------------------------------------------------------------

MAX_S, MAX_D, MAX_X = 2, 2, 2

# Four basic model groups: singlet, doublet, two doublets, sextet.
BASIC_TOPOLOGIES = [
  Topology(1, 0, 0),
  Topology(0, 1, 0),
  Topology(0, 2, 0),
  Topology(0, 0, 1),
]


def combination_topologies() -> list[Topology]:
  """Mixtures of the four basic groups (at most 1 singlet, 2 doublets, 1 sextet)."""
  basic = {t.as_tuple() for t in BASIC_TOPOLOGIES}
  combos: list[Topology] = []
  for ns in range(0, 2):
    for nd in range(0, 3):
      for nx in range(0, 2):
        if ns + nd + nx == 0:
          continue
        topo = Topology(ns, nd, nx)
        if topo.as_tuple() not in basic:
          combos.append(topo)
  return combos


def topology_candidates(allow_combinations: bool = False) -> list[Topology]:
  if allow_combinations:
    return BASIC_TOPOLOGIES + combination_topologies()
  return list(BASIC_TOPOLOGIES)


# Backward-compatible aliases
SINGLE_TOPOLOGIES = BASIC_TOPOLOGIES[:2]
COMPETING_TOPOLOGIES = BASIC_TOPOLOGIES[2:]
SIMPLE_TOPOLOGIES = BASIC_TOPOLOGIES

PHASE1_TOPOLOGIES = SIMPLE_TOPOLOGIES


def _all_topologies() -> list[Topology]:
  tops = []
  for ns in range(MAX_S + 1):
    for nd in range(MAX_D + 1):
      for nx in range(MAX_X + 1):
        if ns + nd + nx == 0:
          continue
        tops.append(Topology(ns, nd, nx))
  return tops


PHASE2_TOPOLOGIES = [t for t in _all_topologies() if t not in PHASE1_TOPOLOGIES]


# --- Model construction & fitting ---------------------------------------------------

def _param_bounds(velocity: np.ndarray) -> dict[str, tuple[float, float]]:
  """Physical bounds scaled to the velocity window of the spectrum."""
  v = np.asarray(velocity, dtype=float)
  span = float(v.max() - v.min()) if v.size else 24.0
  v_abs = max(abs(float(v.min())), abs(float(v.max())), span * 0.5)
  return {
    "singlet_center": (-2.5, 2.5),
    "singlet_width": (0.03, 0.9),
    "doublet_delta": (-1.5, 1.5),
    "doublet_quad": (0.1, min(4.0, span * 0.3)),
    "gamma": (0.03, 0.75),
    "sextet_delta": (-1.5, 1.5),
    "q_shift": (-0.8, 0.8),
    "B_hf": (0.5, max(12.0, v_abs * 0.95)),
    "scale": (0.02, 1.2),
    "amplitude": (0.02, 1.2),
  }


def _clip(value: float, lo: float, hi: float) -> float:
  return float(np.clip(value, lo, hi))


def _default_param_block(
  comp_type: str,
  prefix: str,
  rng: np.random.Generator,
  bounds: dict[str, tuple[float, float]],
) -> list[tuple]:
  """Return lmfit add_many tuples for one component."""
  if comp_type == "singlet":
    lo_c, hi_c = bounds["singlet_center"]
    lo_w, hi_w = bounds["singlet_width"]
    lo_a, hi_a = bounds["amplitude"]
    return [
      (f"{prefix}amplitude", float(rng.uniform(lo_a, hi_a * 0.5)), True, lo_a, hi_a),
      (f"{prefix}center", float(rng.uniform(lo_c * 0.5, hi_c * 0.5)), True, lo_c, hi_c),
      (f"{prefix}width", float(rng.uniform(lo_w, hi_w * 0.6)), True, lo_w, hi_w),
    ]
  if comp_type == "doublet":
    lo_d, hi_d = bounds["doublet_delta"]
    lo_q, hi_q = bounds["doublet_quad"]
    lo_g, hi_g = bounds["gamma"]
    lo_s, hi_s = bounds["scale"]
    return [
      (f"{prefix}delta", float(rng.uniform(lo_d * 0.5, hi_d * 0.8)), True, lo_d, hi_d),
      (f"{prefix}quad", float(rng.uniform(lo_q * 0.4, hi_q * 0.9)), True, lo_q, hi_q),
      (f"{prefix}gamma", float(rng.uniform(lo_g, hi_g * 0.6)), True, lo_g, hi_g),
      (f"{prefix}scale", float(rng.uniform(lo_s, hi_s * 0.5)), True, lo_s, hi_s),
    ]
  lo_d, hi_d = bounds["sextet_delta"]
  lo_qs, hi_qs = bounds["q_shift"]
  lo_b, hi_b = bounds["B_hf"]
  lo_g, hi_g = bounds["gamma"]
  lo_s, hi_s = bounds["scale"]
  return [
    (f"{prefix}delta", float(rng.uniform(lo_d * 0.5, hi_d * 0.8)), True, lo_d, hi_d),
    (f"{prefix}q_shift", float(rng.uniform(lo_qs * 0.5, hi_qs * 0.5)), True, lo_qs, hi_qs),
    (f"{prefix}B_hf", float(rng.uniform(lo_b * 0.2, hi_b * 0.95)), True, lo_b, hi_b),
    (f"{prefix}gamma", float(rng.uniform(lo_g, hi_g * 0.7)), True, lo_g, hi_g),
    (f"{prefix}scale", float(rng.uniform(lo_s, hi_s * 0.5)), True, lo_s, hi_s),
  ]


def _estimate_peak_gamma(velocity: np.ndarray, absorption: np.ndarray, peak_idx: np.ndarray) -> float:
  if peak_idx.size == 0:
    return 0.2
  try:
    widths = peak_widths(absorption, peak_idx, rel_height=0.5)[0]
    step = float(np.median(np.diff(velocity))) if len(velocity) > 1 else 0.1
    return _clip(float(np.median(widths) * abs(step)), 0.05, 0.75)
  except Exception:
    return 0.2


def _build_params(
  topology: Topology,
  bounds: dict[str, tuple[float, float]],
  *,
  singlet: tuple[float, float, float] | None = None,
  doublet: tuple[float, float, float, float] | None = None,
  doublets: list[tuple[float, float, float, float]] | None = None,
  sextet: tuple[float, float, float, float, float] | None = None,
) -> Parameters:
  """Build lmfit Parameters for a topology from optional per-type seeds."""
  params = Parameters()
  lo_c, hi_c = bounds["singlet_center"]
  lo_w, hi_w = bounds["singlet_width"]
  lo_a, hi_a = bounds["amplitude"]
  lo_d, hi_d = bounds["doublet_delta"]
  lo_q, hi_q = bounds["doublet_quad"]
  lo_g, hi_g = bounds["gamma"]
  lo_s, hi_s = bounds["scale"]
  lo_sd, hi_sd = bounds["sextet_delta"]
  lo_qs, hi_qs = bounds["q_shift"]
  lo_b, hi_b = bounds["B_hf"]

  for i in range(topology.n_s):
    prefix = f"l{i + 1}_"
    amp, center, width = singlet if singlet else (0.15, 0.0, 0.15)
    params.add(f"{prefix}amplitude", _clip(amp, lo_a, hi_a), True, lo_a, hi_a)
    params.add(f"{prefix}center", _clip(center, lo_c, hi_c), True, lo_c, hi_c)
    params.add(f"{prefix}width", _clip(width, lo_w, hi_w), True, lo_w, hi_w)

  for i in range(topology.n_d):
    prefix = f"d{i + 1}_"
    if doublets and i < len(doublets):
      delta, quad, gamma, scale = doublets[i]
    elif doublet:
      delta, quad, gamma, scale = doublet
    else:
      delta, quad, gamma, scale = (0.3 + 0.1 * i, 1.0 - 0.2 * i, 0.15, 0.2 - 0.05 * i)
    params.add(f"{prefix}delta", _clip(delta, lo_d, hi_d), True, lo_d, hi_d)
    params.add(f"{prefix}quad", _clip(quad, lo_q, hi_q), True, lo_q, hi_q)
    params.add(f"{prefix}gamma", _clip(gamma, lo_g, hi_g), True, lo_g, hi_g)
    params.add(f"{prefix}scale", _clip(scale, lo_s, hi_s), True, lo_s, hi_s)

  for i in range(topology.n_x):
    prefix = f"x{i + 1}_"
    delta, q_shift, b_hf, gamma, scale = sextet if sextet else (0.3, 0.0, 2.5 + 0.5 * i, 0.2, 0.2)
    params.add(f"{prefix}delta", _clip(delta, lo_sd, hi_sd), True, lo_sd, hi_sd)
    params.add(f"{prefix}q_shift", _clip(q_shift, lo_qs, hi_qs), True, lo_qs, hi_qs)
    params.add(f"{prefix}B_hf", _clip(b_hf, lo_b, hi_b), True, lo_b, hi_b)
    params.add(f"{prefix}gamma", _clip(gamma, lo_g, hi_g), True, lo_g, hi_g)
    params.add(f"{prefix}scale", _clip(scale, lo_s, hi_s), True, lo_s, hi_s)

  return params


def _heuristic_starts(
  velocity: np.ndarray,
  intensity: np.ndarray,
  topology: Topology,
  bounds: dict[str, tuple[float, float]],
) -> list[Parameters]:
  """Data-driven starting points from absorption peaks."""
  v = np.asarray(velocity, dtype=float)
  absorp = absorption_profile(intensity)
  _, peak_idx = _count_absorption_peaks(absorp, prominence_frac=0.012)
  if peak_idx.size == 0:
    return []

  peak_pos = v[peak_idx]
  peak_heights = absorp[peak_idx]
  gamma_est = _estimate_peak_gamma(v, absorp, peak_idx)
  scale_est = _clip(float(peak_heights.max()) * 0.6, bounds["scale"][0], bounds["scale"][1])
  starts: list[Parameters] = []

  if topology.n_s > 0 and peak_idx.size >= 1:
    strongest = int(peak_idx[np.argmax(peak_heights)])
    center = float(v[strongest])
    starts.append(
      _build_params(
        topology,
        bounds,
        singlet=(scale_est, center, gamma_est),
      )
    )

  if topology.n_d > 0 and peak_idx.size >= 2:
    order = np.argsort(peak_heights)[::-1][:2]
    p1, p2 = float(peak_pos[order[0]]), float(peak_pos[order[1]])
    delta = (p1 + p2) / 2.0
    quad = abs(p2 - p1)
    starts.append(_build_params(topology, bounds, doublet=(delta, quad, gamma_est, scale_est)))

    if topology.n_d >= 2:
      sorted_pos = np.sort(peak_pos)
      if peak_idx.size >= 4:
        d1 = (
          (sorted_pos[0] + sorted_pos[1]) / 2.0,
          abs(sorted_pos[1] - sorted_pos[0]),
          gamma_est,
          scale_est * 0.5,
        )
        d2 = (
          (sorted_pos[-2] + sorted_pos[-1]) / 2.0,
          abs(sorted_pos[-1] - sorted_pos[-2]),
          gamma_est,
          scale_est * 0.5,
        )
        starts.append(_build_params(topology, bounds, doublets=[d1, d2]))
      else:
        p_lo, p_hi = float(sorted_pos[0]), float(sorted_pos[-1])
        d1 = (p_lo, max(0.4, quad * 0.8), gamma_est, scale_est * 0.55)
        d2 = (p_hi, max(0.4, quad * 0.8), gamma_est, scale_est * 0.45)
        starts.append(_build_params(topology, bounds, doublets=[d1, d2]))

  if topology.n_x > 0:
    lo_b, hi_b = bounds["B_hf"]
    if peak_idx.size >= 2:
      outer = (float(peak_pos.min()), float(peak_pos.max()))
      delta_est = sum(outer) / 2.0
      b_hf_est = (outer[1] - outer[0]) / 2.0
      starts.append(
        _build_params(
          topology,
          bounds,
          sextet=(delta_est, 0.0, _clip(b_hf_est, lo_b, hi_b), gamma_est, scale_est),
        )
      )
    v_abs = max(abs(float(v.min())), abs(float(v.max())))
    for frac in (0.35, 0.5, 0.7, 0.85):
      b_hf = _clip(v_abs * frac, lo_b, hi_b)
      starts.append(_build_params(topology, bounds, sextet=(0.0, 0.0, b_hf, gamma_est, scale_est)))

  return starts


def build_combined_model(topology: Topology) -> Model:
  combined = Model(lambda x: np.zeros_like(x, dtype=float))
  for i in range(topology.n_s):
    combined += Model(lorentzian, prefix=f"l{i + 1}_")
  for i in range(topology.n_d):
    combined += Model(doublet_lorentzian, prefix=f"d{i + 1}_")
  for i in range(topology.n_x):
    combined += Model(sextet_lorentzian, prefix=f"x{i + 1}_")
  return combined


def _notebook_defaults(topology: Topology, bounds: dict[str, tuple[float, float]]) -> Parameters:
  """Notebook-style defaults within widened bounds."""
  kwargs: dict = {}
  if topology.n_s > 0:
    kwargs["singlet"] = (0.15, 0.0, 0.15)
  if topology.n_d == 1:
    kwargs["doublet"] = (0.3, 1.0, 0.15, 0.2)
  elif topology.n_d >= 2:
    kwargs["doublets"] = [(0.3, 1.0, 0.15, 0.15), (0.5, 0.8, 0.15, 0.12)]
  if topology.n_x > 0:
    lo_b, hi_b = bounds["B_hf"]
    kwargs["sextet"] = (0.3, 0.0, _clip(4.0, lo_b, hi_b), 0.2, 0.15)
  return _build_params(topology, bounds, **kwargs)


def _fit_starts(
  velocity: np.ndarray,
  intensity: np.ndarray,
  topology: Topology,
  n_random: int,
  seed: int,
) -> list[Parameters]:
  bounds = _param_bounds(velocity)
  rng = np.random.default_rng(seed)
  starts: list[Parameters] = []
  seen: set[tuple] = set()

  def _add(params: Parameters) -> None:
    key = tuple(round(params[name].value, 5) for name in params)
    if key not in seen:
      seen.add(key)
      starts.append(params)

  for params in _heuristic_starts(velocity, intensity, topology, bounds):
    _add(params)
  _add(_notebook_defaults(topology, bounds))
  for _ in range(n_random):
    params = Parameters()
    for i in range(topology.n_s):
      for item in _default_param_block("singlet", f"l{i + 1}_", rng, bounds):
        params.add(*item)
    for i in range(topology.n_d):
      for item in _default_param_block("doublet", f"d{i + 1}_", rng, bounds):
        params.add(*item)
    for i in range(topology.n_x):
      for item in _default_param_block("sextet", f"x{i + 1}_", rng, bounds):
        params.add(*item)
    _add(params)
  return starts


def _component_fits_from_result(
  x: np.ndarray, topology: Topology, result: lmfit.minimizer.MinimizerResult
) -> dict[str, np.ndarray]:
  fits: dict[str, np.ndarray] = {}
  p = result.params
  for i in range(topology.n_s):
    prefix = f"l{i + 1}_"
    fits[prefix] = lorentzian(
      x,
      p[f"{prefix}amplitude"].value,
      p[f"{prefix}center"].value,
      p[f"{prefix}width"].value,
    )
  for i in range(topology.n_d):
    prefix = f"d{i + 1}_"
    fits[prefix] = doublet_lorentzian(
      x,
      p[f"{prefix}delta"].value,
      p[f"{prefix}quad"].value,
      p[f"{prefix}gamma"].value,
      p[f"{prefix}scale"].value,
    )
  for i in range(topology.n_x):
    prefix = f"x{i + 1}_"
    fits[prefix] = sextet_lorentzian(
      x,
      p[f"{prefix}delta"].value,
      p[f"{prefix}q_shift"].value,
      p[f"{prefix}B_hf"].value,
      p[f"{prefix}gamma"].value,
      p[f"{prefix}scale"].value,
    )
  return fits


def calculate_areas(x: np.ndarray, component_fits: dict[str, np.ndarray]) -> dict[str, float]:
  total_fit = np.zeros_like(x, dtype=float)
  for fit in component_fits.values():
    total_fit += fit
  total_area = trapezoid(total_fit, x)
  areas = {}
  for prefix, fit in component_fits.items():
    comp_area = trapezoid(fit, x)
    areas[prefix] = (comp_area / total_area) * 100.0 if total_area != 0 else 0.0
  return areas


def _metrics(y: np.ndarray, y_fit: np.ndarray, result: lmfit.minimizer.MinimizerResult) -> tuple[float, float, float]:
  residual = y_fit - y
  rmse = float(np.sqrt(np.mean(residual**2)))
  denom = np.where(np.abs(y) < 1e-6, 1.0, y)
  mean_rel = float(np.mean(np.abs((y - y_fit) / denom)) * 100.0)
  bic = float(result.bic) if result.bic is not None else float("inf")
  return rmse, mean_rel, bic


def fit_topology(
  velocity: np.ndarray,
  intensity: np.ndarray,
  topology: Topology,
  n_random_starts: int = 10,
  seed: int = 42,
) -> FitResult | None:
  """Fit a fixed topology with multi-start lmfit (least_squares)."""
  x = np.asarray(velocity, dtype=float)
  y = np.asarray(intensity, dtype=float)
  if topology.n_components == 0:
    return None

  offset = _uses_offset_convention(y)
  y_trans = _to_transmission_scale(y)
  combined = build_combined_model(topology)

  def residual(params, xv, yv):
    y_fit = 1.0 - combined.eval(params=params, x=xv)
    return y_fit - yv

  best_result = None
  best_rmse = float("inf")

  for start_params in _fit_starts(x, y, topology, n_random_starts, seed):
    try:
      result = minimize(
        residual,
        start_params,
        args=(x, y_trans),
        method="least_squares",
        max_nfev=1200,
      )
      y_fit_trans = 1.0 - combined.eval(params=result.params, x=x)
      y_fit = _from_transmission_scale(y_fit_trans, offset)
      rmse, _, _ = _metrics(y, y_fit, result)
      if rmse < best_rmse and np.isfinite(rmse):
        best_rmse = rmse
        best_result = result
    except Exception:
      continue

  if best_result is None:
    return None

  y_fit_trans = 1.0 - combined.eval(params=best_result.params, x=x)
  y_fit = _from_transmission_scale(y_fit_trans, offset)
  component_fits = _component_fits_from_result(x, topology, best_result)
  rmse, mean_rel, bic = _metrics(y, y_fit, best_result)
  areas = calculate_areas(x, component_fits)

  return FitResult(
    topology=topology,
    velocity=x,
    intensity=y,
    best_fit=y_fit,
    component_fits=component_fits,
    params=best_result.params,
    rmse=rmse,
    mean_rel_error=mean_rel,
    bic=bic,
    method="fit_topology",
    areas=areas,
    report_rows=_build_report_rows(topology, best_result.params, areas),
  )


def _build_report_rows(
  topology: Topology, params: lmfit.Parameters, areas: dict[str, float]
) -> list[dict]:
  rows = []
  for i in range(topology.n_s):
    prefix = f"l{i + 1}_"
    rows.append(
      {
        "Phase": f"Singlet {i + 1}",
        "Type": "Singlet",
        "Amplitude": params[f"{prefix}amplitude"].value,
        "Center (mm/s)": params[f"{prefix}center"].value,
        "Width (mm/s)": params[f"{prefix}width"].value,
        "IS (mm/s)": params[f"{prefix}center"].value,
        "Quad Splitting (mm/s)": np.nan,
        "Bhf (T)": np.nan,
        "Area (%)": areas.get(prefix, 0.0),
      }
    )
  for i in range(topology.n_d):
    prefix = f"d{i + 1}_"
    rows.append(
      {
        "Phase": f"Doublet {i + 1}",
        "Type": "Doublet",
        "Amplitude": np.nan,
        "Center (mm/s)": np.nan,
        "Width (mm/s)": params[f"{prefix}gamma"].value,
        "IS (mm/s)": params[f"{prefix}delta"].value,
        "Quad Splitting (mm/s)": params[f"{prefix}quad"].value,
        "Bhf (T)": np.nan,
        "Area (%)": areas.get(prefix, 0.0),
      }
    )
  for i in range(topology.n_x):
    prefix = f"x{i + 1}_"
    rows.append(
      {
        "Phase": f"Sextet {i + 1}",
        "Type": "Sextet",
        "Amplitude": np.nan,
        "Center (mm/s)": np.nan,
        "Width (mm/s)": params[f"{prefix}gamma"].value,
        "IS (mm/s)": params[f"{prefix}delta"].value,
        "Quad Splitting (mm/s)": params[f"{prefix}q_shift"].value,
        "Bhf (T)": params[f"{prefix}B_hf"].value,
        "Area (%)": areas.get(prefix, 0.0),
      }
    )
  return rows


# --- Intensity conventions ----------------------------------------------------------

def _uses_offset_convention(intensity: np.ndarray) -> bool:
  """
  MossAI preprocessed spectra use transmission - 1 (baseline ~0, dips negative).
  PyMossFit notebook CSVs use transmission (baseline ~1, dips below 1).
  """
  y = np.asarray(intensity, dtype=float)
  if y.size == 0:
    return True
  return float(np.median(y)) < 0.5


def _to_transmission_scale(intensity: np.ndarray) -> np.ndarray:
  y = np.asarray(intensity, dtype=float)
  if _uses_offset_convention(y):
    return y + 1.0
  return y


def _from_transmission_scale(intensity: np.ndarray, offset: bool) -> np.ndarray:
  y = np.asarray(intensity, dtype=float)
  if offset:
    return y - 1.0
  return y


# --- Unsupervised topology predictors -----------------------------------------------

def absorption_profile(intensity: np.ndarray) -> np.ndarray:
  y = np.asarray(intensity, dtype=float)
  if _uses_offset_convention(y):
    return np.clip(-y, 0.0, None)
  return np.clip(1.0 - y, 0.0, None)


def _count_absorption_peaks(absorption: np.ndarray, prominence_frac: float = 0.02) -> tuple[int, np.ndarray]:
  prom = max(prominence_frac * absorption.max(), 1e-5)
  peaks, _ = find_peaks(absorption, prominence=prom, distance=3)
  return len(peaks), peaks


def _map_peak_count_to_topology(n_peaks: int) -> Topology:
  """Heuristic mapping from detected absorption peaks to a simple topology."""
  if n_peaks <= 1:
    return Topology(1, 0, 0)
  if n_peaks == 2:
    return Topology(0, 1, 0)
  if n_peaks in (3, 4):
    return Topology(0, 2, 0)
  return Topology(0, 0, 1)


def predict_topology_peak_heuristic(velocity: np.ndarray, intensity: np.ndarray) -> Topology:
  """Peak detection on absorption profile (unsupervised)."""
  absorp = absorption_profile(intensity)
  n_peaks, _ = _count_absorption_peaks(absorp)
  return _map_peak_count_to_topology(n_peaks)


def predict_topology_gmm(
  velocity: np.ndarray,
  intensity: np.ndarray,
  max_components: int = 8,
) -> Topology:
  """
  Weighted 1-D GMM on velocity axis; component count chosen by BIC (unsupervised).
  """
  x = np.asarray(velocity, dtype=float).reshape(-1, 1)
  w = absorption_profile(intensity)
  if w.sum() <= 0:
    return Topology(1, 0, 0)

  best_k, best_bic = 1, float("inf")
  for k in range(1, min(max_components, len(x)) + 1):
    try:
      gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, max_iter=200)
      gmm.fit(x, sample_weight=w)
      bic = gmm.bic(x)
      if bic < best_bic:
        best_bic = bic
        best_k = k
    except Exception:
      continue
  return _map_peak_count_to_topology(best_k)


def predict_topology_nmf(
  velocity: np.ndarray,
  intensity: np.ndarray,
  max_rank: int = 6,
) -> Topology:
  """
  NMF rank on absorption envelope; rank chosen by reconstruction error elbow (unsupervised).
  """
  _ = velocity
  absorp = absorption_profile(intensity).reshape(1, -1)
  if absorp.max() <= 0:
    return Topology(1, 0, 0)

  errors = []
  ranks = range(1, min(max_rank, absorp.shape[1]) + 1)
  for r in ranks:
    try:
      nmf = NMF(n_components=r, init="nndsvda", random_state=42, max_iter=400)
      nmf.fit(absorp)
      errors.append(nmf.reconstruction_err_)
    except Exception:
      errors.append(float("inf"))

  if not errors or all(np.isinf(errors)):
    return Topology(1, 0, 0)

  # Elbow: largest drop in relative error
  rel = np.diff(errors) / (np.array(errors[:-1]) + 1e-12)
  if len(rel) == 0:
    rank = 1
  else:
    rank = int(np.argmin(rel)) + 1
  return _map_peak_count_to_topology(rank)


def predict_topology_dbscan_peaks(velocity: np.ndarray, intensity: np.ndarray) -> Topology:
  """
  Cluster peak positions with distance-based grouping (unsupervised).
  Pairs -> doublets, hexads -> sextets, singletons -> singlets.
  """
  from sklearn.cluster import DBSCAN

  absorp = absorption_profile(intensity)
  n_peaks, peak_idx = _count_absorption_peaks(absorp)
  if n_peaks == 0:
    return Topology(1, 0, 0)

  peak_pos = np.asarray(velocity, dtype=float)[peak_idx].reshape(-1, 1)
  if len(peak_pos) == 1:
    return Topology(1, 0, 0)

  clustering = DBSCAN(eps=0.35, min_samples=1).fit(peak_pos)
  labels = clustering.labels_
  clusters: dict[int, list[float]] = {}
  for label, pos in zip(labels, peak_pos.ravel()):
    clusters.setdefault(int(label), []).append(float(pos))

  n_s = sum(1 for pts in clusters.values() if len(pts) == 1)
  n_d = sum(1 for pts in clusters.values() if len(pts) == 2)
  n_x = sum(1 for pts in clusters.values() if len(pts) >= 5)

  if n_s + n_d + n_x == 0:
    return _map_peak_count_to_topology(n_peaks)

  if n_x > 0:
    return Topology(0, 0, 1)
  if n_d >= 2:
    return Topology(0, 2, 0)
  if n_d > 0:
    return Topology(0, 1, 0)
  return Topology(1, 0, 0)


def _best_topology_fit(
  velocity: np.ndarray,
  intensity: np.ndarray,
  topologies: list[Topology],
  n_random_starts: int,
) -> tuple[Topology, FitResult | None]:
  best_fit = None
  best_rmse = float("inf")
  best_topo = topologies[0]

  for topo in topologies:
    result = fit_topology(velocity, intensity, topo, n_random_starts=n_random_starts)
    if result is None:
      continue
    if result.rmse < best_rmse:
      best_rmse = result.rmse
      best_fit = result
      best_topo = topo

  return best_topo, best_fit


def predict_topology_bic_grid(
  velocity: np.ndarray,
  intensity: np.ndarray,
  topologies: list[Topology] | None = None,
  n_random_starts: int = 6,
) -> tuple[Topology, FitResult | None]:
  """
  Exhaustive unsupervised model selection: try topologies, pick lowest BIC.
  Reference baseline (no labels).
  """
  candidates = topologies or SIMPLE_TOPOLOGIES
  best_fit = None
  best_bic = float("inf")
  best_topo = Topology(1, 0, 0)

  for topo in candidates:
    result = fit_topology(velocity, intensity, topo, n_random_starts=n_random_starts)
    if result is None:
      continue
    score = result.bic if np.isfinite(result.bic) else result.rmse
    if score < best_bic:
      best_bic = score
      best_fit = result
      best_topo = topo

  return best_topo, best_fit


def predict_topology_two_phase(
  velocity: np.ndarray,
  intensity: np.ndarray,
  rmse_threshold: float = 0.02,
  n_random_starts: int = 8,
  allow_combinations: bool = False,
) -> tuple[Topology, FitResult | None]:
  """
  Try basic topologies — (1,0,0), (0,1,0), (0,2,0), (0,0,1) — and keep the best RMSE.

  If allow_combinations is True, also tries mixtures such as (1,1,0), (0,1,1), etc.
  """
  _ = rmse_threshold  # kept for API compatibility
  candidates = topology_candidates(allow_combinations)
  return _best_topology_fit(velocity, intensity, candidates, n_random_starts)


TOPOLOGY_PREDICTORS: dict[str, Callable[..., Topology]] = {
  "peak_heuristic": predict_topology_peak_heuristic,
  "gmm": predict_topology_gmm,
  "nmf": predict_topology_nmf,
  "dbscan_peaks": predict_topology_dbscan_peaks,
}


def fit_spectrum(
  velocity: np.ndarray,
  intensity: np.ndarray,
  method: str = "two_phase",
  n_random_starts: int = 10,
  rmse_threshold: float = 0.02,
  allow_combinations: bool = False,
) -> FitResult:
  """
  End-to-end unsupervised fit.

  Methods
  -------
  peak_heuristic, gmm, nmf, dbscan_peaks:
      predict topology then multi-start lmfit.
  two_phase:
      try (1,0,0), (0,1,0), (0,2,0), (0,0,1) [+ mixtures if allow_combinations];
      pick best RMSE (default).
  bic_grid:
      search candidate topologies, pick best BIC.
  """
  if method == "bic_grid":
    topo, cached = predict_topology_bic_grid(
      velocity,
      intensity,
      topologies=topology_candidates(allow_combinations),
      n_random_starts=n_random_starts,
    )
    if cached is not None:
      cached.method = "bic_grid"
      return cached
    result = fit_topology(velocity, intensity, topo, n_random_starts=n_random_starts)
    if result is None:
      raise RuntimeError("bic_grid fit failed for all topologies")
    result.method = "bic_grid"
    return result

  if method == "two_phase":
    topo, cached = predict_topology_two_phase(
      velocity,
      intensity,
      rmse_threshold=rmse_threshold,
      n_random_starts=n_random_starts,
      allow_combinations=allow_combinations,
    )
    if cached is not None:
      cached.method = "two_phase"
      return cached
    result = fit_topology(velocity, intensity, topo, n_random_starts=n_random_starts)
    if result is None:
      raise RuntimeError("two_phase fit failed")
    result.method = "two_phase"
    return result

  if method not in TOPOLOGY_PREDICTORS:
    raise ValueError(f"Unknown method: {method}. Choose from {list(TOPOLOGY_PREDICTORS) + ['two_phase', 'bic_grid']}")

  topo = TOPOLOGY_PREDICTORS[method](velocity, intensity)
  if topo.as_tuple() in {t.as_tuple() for t in BASIC_TOPOLOGIES}:
    topo, result = _best_topology_fit(
      velocity,
      intensity,
      topology_candidates(allow_combinations),
      n_random_starts,
    )
  elif allow_combinations:
    topo, result = _best_topology_fit(
      velocity,
      intensity,
      topology_candidates(True),
      n_random_starts,
    )
  else:
    result = fit_topology(velocity, intensity, topo, n_random_starts=n_random_starts)
  if result is None:
    # fallback to single singlet
    result = fit_topology(velocity, intensity, Topology(1, 0, 0), n_random_starts=n_random_starts)
  if result is None:
    raise RuntimeError(f"fit failed for method={method}")
  result.method = method
  return result


def compare_unsupervised_methods(
  velocity: np.ndarray,
  intensity: np.ndarray,
  methods: list[str] | None = None,
  n_random_starts: int = 8,
) -> pd.DataFrame:
  """Benchmark all unsupervised methods on one spectrum (no labels)."""
  methods = methods or ["peak_heuristic", "gmm", "nmf", "dbscan_peaks", "two_phase", "bic_grid"]
  rows = []
  for name in methods:
    try:
      fit = fit_spectrum(velocity, intensity, method=name, n_random_starts=n_random_starts)
      ns, nd, nx = fit.topology.as_tuple()
      rows.append(
        {
          "method": name,
          "n_s": ns,
          "n_d": nd,
          "n_x": nx,
          "rmse": fit.rmse,
          "mean_rel_error_%": fit.mean_rel_error,
          "bic": fit.bic,
        }
      )
    except Exception as exc:
      rows.append(
        {
          "method": name,
          "n_s": np.nan,
          "n_d": np.nan,
          "n_x": np.nan,
          "rmse": np.nan,
          "mean_rel_error_%": np.nan,
          "bic": np.nan,
          "error": str(exc),
        }
      )
  return pd.DataFrame(rows).sort_values("rmse", na_position="last")


# --- Post-hoc phase identification (not part of the fit model) ----------------------

def _parse_ref_value(value) -> float:
  if isinstance(value, str) and "-" in value:
    lo, hi = map(float, value.split("-"))
    return (lo + hi) / 2.0
  return float(value)


def identify_phases(
  report_rows: list[dict],
  reference_csv: str | Path | None = None,
  n_neighbors: int = 3,
) -> list[dict]:
  """
  KNN matching against reference_data.csv (PyMossFit notebook validation step).
  Not used during fitting — only interprets fitted hyperfine parameters.
  """
  ref_path = Path(reference_csv) if reference_csv else Path(__file__).resolve().parent / "reference_data.csv"
  df_ref = pd.read_csv(ref_path)
  cols = ["IS (mm/s)", "Quad Splitting (mm/s)", "Bhf (T)"]
  for col in cols:
    df_ref[col] = df_ref[col].apply(_parse_ref_value)

  X_ref = df_ref[cols].fillna(0.0).values
  knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
  knn.fit(X_ref)

  out = []
  for i, row in enumerate(report_rows):
    x_exp = np.array(
      [
        row.get("IS (mm/s)", 0.0) or 0.0,
        row.get("Quad Splitting (mm/s)", 0.0) or 0.0,
        row.get("Bhf (T)", 0.0) or 0.0,
      ],
      dtype=float,
    ).reshape(1, -1)
    x_exp = np.nan_to_num(x_exp, nan=0.0)
    dists, idxs = knn.kneighbors(x_exp)
    matches = []
    for d, j in zip(dists[0], idxs[0]):
      ref = df_ref.iloc[j]
      matches.append(
        {
          "compound": ref["Compound Name"],
          "formula": ref["Chemical Formula"],
          "IS_mm_s": ref["IS (mm/s)"],
          "QS_mm_s": ref["Quad Splitting (mm/s)"],
          "Bhf_T": ref["Bhf (T)"],
          "distance": float(d),
        }
      )
    out.append({"phase_index": i, "phase": row.get("Phase"), "matches": matches})
  return out


def load_calibrated_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
  """Load velocity,intensity CSV (Calib-Fe.csv format)."""
  data = np.loadtxt(path, delimiter=",")
  return data[:, 0], data[:, 1]


def fit_result_to_record(
  idx: int,
  pkey: str,
  allow_combinations: bool,
  result: FitResult | None = None,
  fit_error: str | None = None,
) -> dict:
  """Serialize a FitResult (or failure) to a tabular record."""
  record: dict = {
    "idx": int(idx),
    "pkey": str(pkey),
    "allow_combinations": bool(allow_combinations),
    "fit_ok": result is not None,
    "fit_error": fit_error,
    "fit_method": None,
    "n_s": None,
    "n_d": None,
    "n_x": None,
    "fit_rmse": None,
    "fit_bic": None,
    "fit_mean_rel_error": None,
    "hyperfine_params": {},
    "report_rows": [],
    "areas": {},
    "best_fit": None,
  }
  if result is None:
    return record

  record.update(
    {
      "fit_method": result.method,
      "n_s": int(result.topology.n_s),
      "n_d": int(result.topology.n_d),
      "n_x": int(result.topology.n_x),
      "fit_rmse": float(result.rmse),
      "fit_bic": float(result.bic) if np.isfinite(result.bic) else None,
      "fit_mean_rel_error": float(result.mean_rel_error),
      "hyperfine_params": {
        name: float(param.value)
        for name, param in result.params.items()
        if param.value is not None and np.isfinite(param.value)
      },
      "report_rows": result.report_rows,
      "areas": result.areas,
      "best_fit": result.best_fit.tolist(),
    }
  )
  return record


def fit_spectrum_dual_worker(args: tuple) -> list[dict]:
  """
  Picklable worker: fits one spectrum in both modes (allow_combinations False and True).

  args: (idx, pkey, velocity, intensity, method, n_random_starts)
  """
  idx, pkey, velocity, intensity, method, n_starts = args
  v = np.asarray(velocity, dtype=float)
  y = np.asarray(intensity, dtype=float)
  records: list[dict] = []

  for allow_combinations in (False, True):
    try:
      result = fit_spectrum(
        v,
        y,
        method=method,
        n_random_starts=n_starts,
        allow_combinations=allow_combinations,
      )
      records.append(
        fit_result_to_record(idx, pkey, allow_combinations, result=result)
      )
    except Exception as exc:
      records.append(
        fit_result_to_record(
          idx, pkey, allow_combinations, result=None, fit_error=str(exc)
        )
      )
  return records


def fit_spectrum_worker(args: tuple) -> dict:
  """
  Picklable worker for multiprocessing batch fits.

  args: (idx, pkey, velocity, intensity, method, n_random_starts[, allow_combinations])
  """
  if len(args) == 6:
    idx, pkey, velocity, intensity, method, n_starts = args
    allow_combinations = False
  else:
    idx, pkey, velocity, intensity, method, n_starts, allow_combinations = args
  try:
    v = np.asarray(velocity, dtype=float)
    y = np.asarray(intensity, dtype=float)
    result = fit_spectrum(
      v,
      y,
      method=method,
      n_random_starts=n_starts,
      allow_combinations=allow_combinations,
    )
    return {
      "idx": idx,
      "pkey": pkey,
      "fit_ok": True,
      "fit_method": result.method,
      "allow_combinations": allow_combinations,
      "n_s": result.topology.n_s,
      "n_d": result.topology.n_d,
      "n_x": result.topology.n_x,
      "fit_rmse": result.rmse,
      "fit_mean_rel_error": result.mean_rel_error,
      "fit_bic": result.bic,
      "hyperfine_params": {
        name: float(param.value)
        for name, param in result.params.items()
        if param.value is not None and np.isfinite(param.value)
      },
      "best_fit": result.best_fit.tolist(),
      "report_rows": result.report_rows,
      "areas": result.areas,
    }
  except Exception as exc:
    return {
      "idx": idx,
      "pkey": pkey,
      "fit_ok": False,
      "fit_error": str(exc),
    }
