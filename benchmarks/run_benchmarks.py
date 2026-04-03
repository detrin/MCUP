import argparse
import time
from dataclasses import dataclass
from typing import Callable, List, Type

import numpy as np

from mcup import DemingRegressor, WeightedRegressor, XYWeightedRegressor
from mcup._analytical import ols_solve
from mcup.base import BaseRegressor


class OLSRegressor(BaseRegressor):
    """Plain OLS baseline: minimises Σ(y−f(x))² and scales covariance by σ²=SSR/(n−p)."""

    def fit(self, X, y, y_err=None, x_err=None, p0=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        p0 = np.asarray(p0, dtype=float)
        params, cov = ols_solve(self.func, X, y, p0, self.optimizer)
        self.params_ = params
        self.covariance_ = cov
        self.params_std_ = np.sqrt(np.diag(cov))
        return self


# ─── config / result types ────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    name: str
    description: str
    model_str: str
    param_names: List[str]
    param_sampler: Callable      # rng → true_params
    p0_fn: Callable              # (rng, true_params) → p0
    n_data: int
    estimator_cls: Type[BaseRegressor]
    method: str
    model_fn: Callable
    data_fn: Callable            # (rng, true_params) → {"X", "y", "y_err", ...}
    n_param_samples: int = 200
    mc_n_iter: int = 200
    note: str = ""


@dataclass
class BenchmarkResult:
    name: str
    description: str
    model_str: str
    n_data: int
    n_samples: int
    n_ok: int
    estimator: str
    method: str
    param_names: List[str]
    mean_bias_pct: np.ndarray    # mean((est−true)/|true|) × 100
    rmse_pct: np.ndarray         # √mean((est−true)²/true²) × 100
    coverage_68: np.ndarray      # fraction where |est−true| < params_std
    runtime_ms: float
    note: str = ""


# ─── model functions ──────────────────────────────────────────────────────────

def _linear(x, p):
    return p[0] + p[1] * x


def _exponential(t, p):
    return p[0] * np.exp(-p[1] * t)


def _power_law(t, p):
    return p[0] * (t ** p[1])


def _gaussian_peak(x, p):
    return p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2)


def _damped_sine(t, p):
    return p[0] * np.exp(-p[1] * t) * np.sin(p[2] * t + p[3])


# ─── data generators  (rng, true_params) → {"X", "y", "y_err", ["x_err"]} ───

def _gen_linear_homo(rng, p):
    x = np.linspace(0, 10, 50)
    ye = 0.5 * np.ones(50)
    return {"X": x, "y": _linear(x, p) + rng.normal(0, ye), "y_err": ye}


def _gen_linear_hetero(rng, p):
    x = np.linspace(0.5, 10, 50)
    ye = 0.1 + 0.1 * x
    return {"X": x, "y": _linear(x, p) + rng.normal(0, ye), "y_err": ye}


def _gen_exp_decay(rng, p):
    t = np.linspace(0, 80, 20)
    y_true = _exponential(t, p)
    ye = np.sqrt(np.maximum(y_true, 1.0))
    return {"X": t, "y": np.maximum(y_true + rng.normal(0, ye), 1.0), "y_err": ye}


def _gen_power_law(rng, p):
    t = np.linspace(1, 20, 30)
    y_true = _power_law(t, p)
    ye = 0.08 * y_true
    return {"X": t, "y": np.maximum(y_true + rng.normal(0, ye), 0.01), "y_err": ye}


def _gen_gaussian_peak(rng, p):
    x = np.linspace(2, 8, 40)
    y_true = _gaussian_peak(x, p)
    ye = np.sqrt(np.maximum(y_true, 1.0))
    return {"X": x, "y": np.maximum(y_true + rng.normal(0, ye), 0.0), "y_err": ye}


def _gen_damped_sine(rng, p):
    t = np.linspace(0.1, 6, 60)
    y_true = _damped_sine(t, p)
    ye = 0.05 * np.ones(60)
    return {"X": t, "y": y_true + rng.normal(0, ye), "y_err": ye}


def _gen_exp_timing(rng, p):
    t_true = np.linspace(2, 80, 20)
    y_true = _exponential(t_true, p)
    ye = 0.05 * y_true
    xe = 2.0 * np.ones(20)
    return {
        "X": t_true + rng.normal(0, xe),
        "y": np.maximum(y_true + rng.normal(0, ye), 1.0),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_hookes_law(rng, p):
    x_true = np.linspace(0.05, 2.0, 25)
    xe, ye = 0.02 * np.ones(25), 0.15 * np.ones(25)
    return {
        "X": x_true + rng.normal(0, xe),
        "y": _linear(x_true, p) + rng.normal(0, ye),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_beer_lambert(rng, p):
    c_true = np.linspace(5, 100, 20)
    xe = 2.0 * np.ones(20)
    ye = 0.005 * np.ones(20)
    return {
        "X": c_true + rng.normal(0, xe),
        "y": _linear(c_true, p) + rng.normal(0, ye),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_method_comparison(rng, p):
    c_true = rng.uniform(10, 200, 30)
    xe, ye = 5.0 * np.ones(30), 3.0 * np.ones(30)
    return {
        "X": c_true + rng.normal(0, xe),
        "y": _linear(c_true, p) + rng.normal(0, ye),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_isotope_ratio(rng, p):
    x_true = rng.uniform(0.5, 2.0, 20)
    xe = 0.003 * np.ones(20)
    ye = 0.004 * np.ones(20)
    return {
        "X": x_true + rng.normal(0, xe),
        "y": _linear(x_true, p) + rng.normal(0, ye),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_linear_small(rng, p):
    x = np.linspace(0, 10, 8)
    ye = 0.5 * np.ones(8)
    return {"X": x, "y": _linear(x, p) + rng.normal(0, ye), "y_err": ye}


def _gen_linear_low_snr(rng, p):
    x = np.linspace(0, 10, 30)
    ye = 3.0 * np.ones(30)
    return {"X": x, "y": _linear(x, p) + rng.normal(0, ye), "y_err": ye}


def _strip_x_err(fn: Callable) -> Callable:
    def wrapped(rng, p):
        d = fn(rng, p)
        d.pop("x_err", None)
        return d
    return wrapped


# ─── parameter samplers  rng → true_params ───────────────────────────────────

def _s_linear(rng):
    return np.array([rng.uniform(0.3, 3.0), rng.uniform(0.5, 4.0)])


def _s_exp_decay(rng):
    return np.array([rng.uniform(300.0, 2000.0), rng.uniform(0.01, 0.15)])


def _s_exp_timing(rng):
    # λ range constrained so decay covers 1–4 half-lives within t=[2, 80]
    return np.array([rng.uniform(300.0, 2000.0), rng.uniform(0.02, 0.08)])


def _s_power_law(rng):
    return np.array([rng.uniform(0.5, 5.0), rng.uniform(0.8, 2.5)])


def _s_gaussian(rng):
    return np.array([rng.uniform(100.0, 1000.0), rng.uniform(3.0, 7.0), rng.uniform(0.3, 1.5)])


def _s_damped_sine(rng):
    return np.array([
        rng.uniform(1.0, 5.0),
        rng.uniform(0.1, 0.8),
        rng.uniform(3.0, 8.0),
        rng.uniform(0.1, 1.2),
    ])


def _s_hookes(rng):
    return np.array([rng.uniform(0.05, 0.5), rng.uniform(5.0, 15.0)])


def _s_beer_lambert(rng):
    return np.array([rng.uniform(0.01, 0.05), rng.uniform(0.02, 0.1)])


def _s_method_comparison(rng):
    return np.array([rng.uniform(1.0, 8.0), rng.uniform(0.85, 1.25)])


def _s_isotope(rng):
    return np.array([rng.uniform(0.0003, 0.003), rng.uniform(0.980, 0.995)])


def _p0_default(_, true_p):
    return true_p * 0.75


# ─── runner ───────────────────────────────────────────────────────────────────

def _run(cfg: BenchmarkConfig) -> BenchmarkResult:
    kwargs = {"method": cfg.method}
    if cfg.method == "mc":
        kwargs["n_iter"] = cfg.mc_n_iter

    rng_ref = np.random.default_rng(42)
    p_ref = cfg.param_sampler(rng_ref)
    d_ref = cfg.data_fn(rng_ref, p_ref)
    Xf, yf = d_ref.pop("X"), d_ref.pop("y")
    est_t = cfg.estimator_cls(cfg.model_fn, **kwargs)
    t0 = time.perf_counter()
    est_t.fit(Xf, yf, **d_ref, p0=cfg.p0_fn(rng_ref, p_ref))
    runtime_ms = (time.perf_counter() - t0) * 1000

    rng = np.random.default_rng(1)
    true_list, est_list, std_list = [], [], []

    for _ in range(cfg.n_param_samples):
        true_p = cfg.param_sampler(rng)
        p0 = cfg.p0_fn(rng, true_p)
        d = cfg.data_fn(rng, true_p)
        X, y = d.pop("X"), d.pop("y")
        ei = cfg.estimator_cls(cfg.model_fn, **kwargs)
        try:
            ei.fit(X, y, **d, p0=p0)
            if np.all(np.isfinite(ei.params_)) and np.all(np.isfinite(ei.params_std_)):
                true_list.append(true_p)
                est_list.append(ei.params_.copy())
                std_list.append(ei.params_std_.copy())
        except Exception:
            pass

    if not true_list:
        raise RuntimeError(f"All samples failed for {cfg.name} {cfg.estimator_cls.__name__}")

    ta = np.array(true_list)
    pa = np.array(est_list)
    sa = np.array(std_list)

    rel_err = (pa - ta) / np.abs(ta)
    mean_bias = rel_err.mean(axis=0) * 100
    rmse = np.sqrt((rel_err ** 2).mean(axis=0)) * 100
    coverage = (np.abs(pa - ta) < sa).mean(axis=0)

    method_label = cfg.method if cfg.method == "analytical" else f"mc({cfg.mc_n_iter})"

    return BenchmarkResult(
        name=cfg.name,
        description=cfg.description,
        model_str=cfg.model_str,
        n_data=cfg.n_data,
        n_samples=cfg.n_param_samples,
        n_ok=len(true_list),
        estimator=cfg.estimator_cls.__name__,
        method=method_label,
        param_names=cfg.param_names,
        mean_bias_pct=mean_bias,
        rmse_pct=rmse,
        coverage_68=coverage,
        runtime_ms=runtime_ms,
        note=cfg.note,
    )


# ─── scenario definitions ─────────────────────────────────────────────────────

def _build_configs(n_samples: int, analytical_only: bool) -> List[BenchmarkConfig]:
    configs = []
    methods = ["analytical"] if analytical_only else ["analytical", "mc"]

    def add(name, desc, model_str, pnames, n_data, cls, model_fn, sampler, data_fn,
            note="", mc_n_iter=200, mc_n_samples=20):
        for method in methods:
            n = n_samples if method == "analytical" else min(n_samples, mc_n_samples)
            configs.append(BenchmarkConfig(
                name=name, description=desc, model_str=model_str,
                param_names=pnames, param_sampler=sampler, p0_fn=_p0_default,
                n_data=n_data, estimator_cls=cls, method=method,
                model_fn=model_fn, data_fn=data_fn,
                n_param_samples=n, mc_n_iter=mc_n_iter, note=note,
            ))

    def add_single(name, desc, model_str, pnames, n_data, cls, model_fn, sampler,
                   data_fn, note="", mc_n_iter=200):
        configs.append(BenchmarkConfig(
            name=name, description=desc, model_str=model_str,
            param_names=pnames, param_sampler=sampler, p0_fn=_p0_default,
            n_data=n_data, estimator_cls=cls, method="analytical",
            model_fn=model_fn, data_fn=data_fn,
            n_param_samples=n_samples, mc_n_iter=mc_n_iter, note=note,
        ))

    # ── S1: Linear calibration, homoscedastic ─────────────────────────────────
    add_single("S1", "Linear calibration — homoscedastic σ_y=0.5 — OLS (ignores σ_y)",
               "y = a + b·x", ["a", "b"], 50,
               OLSRegressor, _linear, _s_linear, _gen_linear_homo, note="⚠ OLS")
    add("S1", "Linear calibration — homoscedastic σ_y=0.5, n=50",
        "y = a + b·x", ["a", "b"], 50,
        WeightedRegressor, _linear, _s_linear, _gen_linear_homo)

    # ── S2: Linear calibration, heteroscedastic ───────────────────────────────
    add_single("S2", "Linear calibration — heteroscedastic σ_y=0.1+0.1x — OLS (ignores σ_y)",
               "y = a + b·x", ["a", "b"], 50,
               OLSRegressor, _linear, _s_linear, _gen_linear_hetero, note="⚠ OLS")
    add("S2", "Linear calibration — heteroscedastic σ_y=0.1+0.1x, n=50",
        "y = a + b·x", ["a", "b"], 50,
        WeightedRegressor, _linear, _s_linear, _gen_linear_hetero)

    # ── S3: Exponential decay, Poisson counting ───────────────────────────────
    add_single("S3", "Radioactive decay — Poisson y-errors √A(t) — OLS (ignores σ_y)",
               "A(t) = A₀·exp(−λt)", ["A₀", "λ"], 20,
               OLSRegressor, _exponential, _s_exp_decay, _gen_exp_decay, note="⚠ OLS")
    add("S3", "Radioactive decay — Poisson y-errors √A(t), n=20",
        "A(t) = A₀·exp(−λt)", ["A₀", "λ"], 20,
        WeightedRegressor, _exponential, _s_exp_decay, _gen_exp_decay)

    # ── S4: Power law (anomalous diffusion) ───────────────────────────────────
    add_single("S4", "Anomalous diffusion power law — σ_y=8%·y — OLS (ignores σ_y)",
               "MSD = D·t^α", ["D", "α"], 30,
               OLSRegressor, _power_law, _s_power_law, _gen_power_law, note="⚠ OLS")
    add("S4", "Anomalous diffusion power law — σ_y=8%·y, n=30",
        "MSD = D·t^α", ["D", "α"], 30,
        WeightedRegressor, _power_law, _s_power_law, _gen_power_law)

    # ── S5: Gaussian spectral peak, photon counting ───────────────────────────
    add_single("S5", "Spectral peak (Gaussian) — photon counting — OLS (ignores σ_y)",
               "I = A·exp(-(x-μ)²/2σ²)", ["A", "μ", "σ"], 40,
               OLSRegressor, _gaussian_peak, _s_gaussian, _gen_gaussian_peak, note="⚠ OLS")
    add("S5", "Spectral peak (Gaussian) — photon counting, n=40",
        "I = A·exp(-(x-μ)²/2σ²)", ["A", "μ", "σ"], 40,
        WeightedRegressor, _gaussian_peak, _s_gaussian, _gen_gaussian_peak)

    # ── S6: Damped oscillator (NMR / mechanical) ──────────────────────────────
    add_single("S6", "Damped oscillator (NMR/vibration) — σ_y=0.05 — OLS (ignores σ_y)",
               "y = A·exp(−γt)·sin(ωt+φ)", ["A", "γ", "ω", "φ"], 60,
               OLSRegressor, _damped_sine, _s_damped_sine, _gen_damped_sine, note="⚠ OLS")
    add("S6", "Damped oscillator (NMR/vibration) — σ_y=0.05, n=60",
        "y = A·exp(−γt)·sin(ωt+φ)", ["A", "γ", "ω", "φ"], 60,
        WeightedRegressor, _damped_sine, _s_damped_sine, _gen_damped_sine, mc_n_iter=300)

    # ── S7: Exponential decay + timing errors ─────────────────────────────────
    add_single("S7", "Decay + timing errors σ_t=2s — WeightedRegressor (ignores x-err)",
               "A(t) = A₀·exp(−λt)", ["A₀", "λ"], 20,
               WeightedRegressor, _exponential, _s_exp_timing,
               _strip_x_err(_gen_exp_timing), note="⚠ ignores x-errors")
    add("S7", "Decay + timing errors σ_t=2s — XYWeightedRegressor",
        "A(t) = A₀·exp(−λt)", ["A₀", "λ"], 20,
        XYWeightedRegressor, _exponential, _s_exp_timing, _gen_exp_timing)

    # ── S8: Hooke's law — both F and x measured ───────────────────────────────
    add_single("S8", "Hooke's law σ_x=0.02m σ_F=0.15N — WeightedRegressor (ignores x-err)",
               "F = F₀ + k·x", ["F₀", "k"], 25,
               WeightedRegressor, _linear, _s_hookes,
               _strip_x_err(_gen_hookes_law), note="⚠ ignores x-errors")
    add("S8", "Hooke's law σ_x=0.02m σ_F=0.15N — XYWeightedRegressor",
        "F = F₀ + k·x", ["F₀", "k"], 25,
        XYWeightedRegressor, _linear, _s_hookes, _gen_hookes_law)

    # ── S9: Beer-Lambert absorbance vs concentration ──────────────────────────
    add_single("S9", "Beer-Lambert σ_c=2μg/mL σ_A=0.005 — WeightedRegressor (ignores x-err)",
               "A = ε·c + baseline", ["baseline", "ε"], 20,
               WeightedRegressor, _linear, _s_beer_lambert,
               _strip_x_err(_gen_beer_lambert), note="⚠ ignores x-errors")
    add("S9", "Beer-Lambert σ_c=2μg/mL σ_A=0.005 — XYWeightedRegressor",
        "A = ε·c + baseline", ["baseline", "ε"], 20,
        XYWeightedRegressor, _linear, _s_beer_lambert, _gen_beer_lambert)

    # ── S10: Method comparison — OLS vs Deming ────────────────────────────────
    add_single("S10", "Method comparison σ_x=5 σ_y=3 — WeightedRegressor (OLS bias)",
               "y_new = a + b·y_old", ["a", "b"], 30,
               WeightedRegressor, _linear, _s_method_comparison,
               _strip_x_err(_gen_method_comparison), note="⚠ OLS attenuation bias")
    add("S10", "Method comparison σ_x=5 σ_y=3 — DemingRegressor",
        "y_new = a + b·y_old", ["a", "b"], 30,
        DemingRegressor, _linear, _s_method_comparison, _gen_method_comparison,
        mc_n_iter=50, mc_n_samples=10)

    # ── S11: Isotope ratio (mass spectrometry) ────────────────────────────────
    add_single("S11", "Isotope ratio MS σ_x=σ_y≈0.003–0.004 — WeightedRegressor (ignores x-err)",
               "δ_sample = a + b·δ_ref", ["a", "b"], 20,
               WeightedRegressor, _linear, _s_isotope,
               _strip_x_err(_gen_isotope_ratio), note="⚠ ignores x-errors")
    add("S11", "Isotope ratio MS σ_x=σ_y≈0.003–0.004 — DemingRegressor",
        "δ_sample = a + b·δ_ref", ["a", "b"], 20,
        DemingRegressor, _linear, _s_isotope, _gen_isotope_ratio,
        mc_n_iter=50, mc_n_samples=5)

    # ── S12: Small sample ─────────────────────────────────────────────────────
    add("S12", "Small sample — y = a + b·x, n=8, σ_y=0.5 (coverage stress test)",
        "y = a + b·x", ["a", "b"], 8,
        WeightedRegressor, _linear, _s_linear, _gen_linear_small)

    # ── S13: Low SNR ──────────────────────────────────────────────────────────
    add("S13", "Low SNR — y = a + b·x, σ_y=3.0 (signal range 0.3–43), n=30",
        "y = a + b·x", ["a", "b"], 30,
        WeightedRegressor, _linear, _s_linear, _gen_linear_low_snr)

    return configs


# ─── printing ─────────────────────────────────────────────────────────────────

_COV_IDEAL = 0.683


def _cov_mark(c: float) -> str:
    if 0.58 <= c <= 0.78:
        return "✓"
    if 0.48 <= c <= 0.88:
        return "~"
    return "✗"


def _param_summary(r: BenchmarkResult) -> str:
    parts = []
    for name, b, rm, c in zip(r.param_names, r.mean_bias_pct, r.rmse_pct, r.coverage_68):
        parts.append(f"{name}: b={b:+.1f}% rmse={rm:.1f}% {_cov_mark(c)}{c*100:.0f}%")
    return "  ".join(parts)


def _print_results(results: List[BenchmarkResult]) -> None:
    W = 130
    print("\n" + "═" * W)
    print(f"{'MCUP Benchmark Suite':^{W}}")
    print(f"{'bias = mean((est−true)/|true|)×100  |  rmse = √mean((est−true)²/true²)×100  |  coverage = |est−true|<1σ fraction':^{W}}")
    print("═" * W)
    print(f"Coverage: ✓ within ±10pp of 68%  |  ~ within ±20pp  |  ✗ outside ±20pp\n")

    by_name: dict = {}
    for r in results:
        by_name.setdefault(r.name, []).append(r)

    for sname, group in by_name.items():
        r0 = group[0]
        base_desc = r0.description.split("—")[0].strip()
        print(f"[{sname}]  {base_desc}  (n_data={r0.n_data})")
        print(f"       Model: {r0.model_str}")
        print(f"  {'─'*25} {'─'*14} {'─'*72} {'─'*9} {'─'*8}")
        print(f"  {'Estimator':<25} {'Method':<14} {'Bias / RMSE / Coverage per param':<72} {'Time':>9} {'OK/Total':>8}")
        print(f"  {'─'*25} {'─'*14} {'─'*72} {'─'*9} {'─'*8}")
        for r in group:
            note_str = f"  {r.note}" if r.note else ""
            param_str = _param_summary(r)
            ok_str = f"{r.n_ok}/{r.n_samples}"
            print(f"  {r.estimator:<25} {r.method:<14} {param_str:<72} {r.runtime_ms:>8.1f}ms {ok_str:>8}{note_str}")
        print()

    _print_summary(results)


def _print_summary(results: List[BenchmarkResult]) -> None:
    W = 130
    print("═" * W)
    print(f"{'Summary (excluding ⚠ rows): worst-case coverage deviation from 68%':^{W}}")
    print("═" * W)

    rows = []
    for r in results:
        if r.note:
            continue
        worst_dev = max(abs(c - _COV_IDEAL) for c in r.coverage_68)
        max_rmse = max(r.rmse_pct)
        rows.append((r.name, r.estimator, r.method, max_rmse, worst_dev, r.runtime_ms))

    rows.sort(key=lambda x: x[4])

    print(f"  {'Scenario':<6} {'Estimator':<25} {'Method':<14} {'Max RMSE%':>10} {'Max cov dev':>12} {'Time':>9}")
    print(f"  {'─'*6} {'─'*25} {'─'*14} {'─'*10} {'─'*12} {'─'*9}")
    for name, est, method, rmse, dev, rt in rows:
        mark = "✓" if dev <= 0.10 else ("~" if dev <= 0.20 else "✗")
        print(f"  {name:<6} {est:<25} {method:<14} {rmse:>9.1f}% {mark}{dev*100:>10.1f}pp {rt:>8.1f}ms")
    print()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MCUP benchmark suite")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Parameter samples per scenario (default: 200; MC capped at 20)")
    parser.add_argument("--analytical-only", action="store_true",
                        help="Skip MC solver configurations")
    args = parser.parse_args()

    configs = _build_configs(args.n_samples, args.analytical_only)
    results = []
    total = len(configs)

    print(f"\nRunning {total} benchmark configurations ({args.n_samples} analytical samples, MC uses fewer samples)...\n")

    for i, cfg in enumerate(configs, 1):
        label = f"[{i}/{total}] {cfg.name} {cfg.estimator_cls.__name__} {cfg.method}"
        print(f"  {label}", end="", flush=True)
        try:
            r = _run(cfg)
            results.append(r)
            print(f"  → {r.runtime_ms:.1f}ms  (n_ok={r.n_ok}/{r.n_samples})")
        except Exception as e:
            print(f"  → FAILED: {e}")

    _print_results(results)


if __name__ == "__main__":
    main()
