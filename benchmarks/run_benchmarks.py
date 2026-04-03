import time
from dataclasses import dataclass, field
from typing import Callable, List, Type

import numpy as np

from mcup import DemingRegressor, WeightedRegressor, XYWeightedRegressor
from mcup.base import BaseRegressor


# ─── result types ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    name: str
    description: str
    model_str: str
    param_names: List[str]
    true_params: np.ndarray
    p0: np.ndarray
    n_samples: int
    estimator_cls: Type[BaseRegressor]
    method: str
    data_fn: Callable
    model_fn: Callable = None
    n_trials: int = 200
    mc_n_iter: int = 500
    note: str = ""


@dataclass
class BenchmarkResult:
    name: str
    description: str
    model_str: str
    n_samples: int
    estimator: str
    method: str
    param_names: List[str]
    true_params: np.ndarray
    mean_params: np.ndarray
    mean_std: np.ndarray
    bias_pct: np.ndarray
    coverage_68: np.ndarray
    runtime_ms: float
    n_trials_ok: int
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


# ─── data generators  (rng) → {"X", "y", "y_err", ["x_err"]} ─────────────────

def _gen_linear_homo(rng):
    x = np.linspace(0, 10, 50)
    ye = 0.5 * np.ones(50)
    return {"X": x, "y": _linear(x, [1.0, 2.0]) + rng.normal(0, ye), "y_err": ye}


def _gen_linear_hetero(rng):
    x = np.linspace(0.5, 10, 50)
    ye = 0.1 + 0.1 * x
    return {"X": x, "y": _linear(x, [1.0, 2.0]) + rng.normal(0, ye), "y_err": ye}


def _gen_exp_decay(rng):
    t = np.linspace(0, 80, 20)
    y_true = _exponential(t, [1000.0, 0.05])
    ye = np.sqrt(np.maximum(y_true, 1.0))
    return {"X": t, "y": np.maximum(y_true + rng.normal(0, ye), 1.0), "y_err": ye}


def _gen_power_law(rng):
    t = np.linspace(1, 20, 30)
    y_true = _power_law(t, [2.0, 1.5])
    ye = 0.08 * y_true
    return {"X": t, "y": np.maximum(y_true + rng.normal(0, ye), 0.01), "y_err": ye}


def _gen_gaussian_peak(rng):
    x = np.linspace(2, 8, 40)
    y_true = _gaussian_peak(x, [500.0, 5.0, 0.8])
    ye = np.sqrt(np.maximum(y_true, 1.0))
    return {"X": x, "y": np.maximum(y_true + rng.normal(0, ye), 0.0), "y_err": ye}


def _gen_damped_sine(rng):
    t = np.linspace(0.1, 6, 60)
    y_true = _damped_sine(t, [3.0, 0.4, 5.0, 0.3])
    ye = 0.05 * np.ones(60)
    return {"X": t, "y": y_true + rng.normal(0, ye), "y_err": ye}


def _gen_exp_timing(rng):
    t_true = np.linspace(2, 80, 20)
    y_true = _exponential(t_true, [1000.0, 0.05])
    ye = 0.05 * y_true
    xe = 2.0 * np.ones(20)
    return {
        "X": t_true + rng.normal(0, xe),
        "y": np.maximum(y_true + rng.normal(0, ye), 1.0),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_hookes_law(rng):
    x_true = np.linspace(0.05, 2.0, 25)
    xe, ye = 0.02 * np.ones(25), 0.15 * np.ones(25)
    return {
        "X": x_true + rng.normal(0, xe),
        "y": _linear(x_true, [0.1, 9.8]) + rng.normal(0, ye),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_beer_lambert(rng):
    c_true = np.linspace(5, 100, 20)
    xe = 2.0 * np.ones(20)
    ye = 0.005 * np.ones(20)
    return {
        "X": c_true + rng.normal(0, xe),
        "y": _linear(c_true, [0.02, 0.05]) + rng.normal(0, ye),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_method_comparison(rng):
    c_true = rng.uniform(10, 200, 30)
    xe, ye = 5.0 * np.ones(30), 3.0 * np.ones(30)
    return {
        "X": c_true + rng.normal(0, xe),
        "y": _linear(c_true, [3.0, 1.05]) + rng.normal(0, ye),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_isotope_ratio(rng):
    x_true = rng.uniform(0.5, 2.0, 20)
    xe = 0.003 * np.ones(20)
    ye = 0.004 * np.ones(20)
    return {
        "X": x_true + rng.normal(0, xe),
        "y": _linear(x_true, [0.001, 0.9878]) + rng.normal(0, ye),
        "x_err": xe,
        "y_err": ye,
    }


def _gen_linear_small(rng):
    x = np.linspace(0, 10, 8)
    ye = 0.5 * np.ones(8)
    return {"X": x, "y": _linear(x, [1.0, 2.0]) + rng.normal(0, ye), "y_err": ye}


def _gen_linear_low_snr(rng):
    x = np.linspace(0, 10, 30)
    ye = 3.0 * np.ones(30)
    return {"X": x, "y": _linear(x, [1.0, 2.0]) + rng.normal(0, ye), "y_err": ye}


def _strip_x_err(fn: Callable) -> Callable:
    def wrapped(rng):
        d = fn(rng)
        d.pop("x_err", None)
        return d
    return wrapped


# ─── runner ───────────────────────────────────────────────────────────────────

def _run(cfg: BenchmarkConfig) -> BenchmarkResult:
    kwargs = {"method": cfg.method}
    if cfg.method == "mc":
        kwargs["n_iter"] = cfg.mc_n_iter

    fixed = cfg.data_fn(np.random.default_rng(42))
    Xf, yf = fixed.pop("X"), fixed.pop("y")
    est_t = cfg.estimator_cls(cfg.model_fn, **kwargs)
    t0 = time.perf_counter()
    est_t.fit(Xf, yf, **fixed, p0=cfg.p0)
    runtime_ms = (time.perf_counter() - t0) * 1000

    rng = np.random.default_rng(0)
    params_list, stds_list = [], []

    for _ in range(cfg.n_trials):
        d = cfg.data_fn(rng)
        Xi, yi = d.pop("X"), d.pop("y")
        ei = cfg.estimator_cls(cfg.model_fn, **kwargs)
        try:
            ei.fit(Xi, yi, **d, p0=cfg.p0)
            if np.all(np.isfinite(ei.params_)) and np.all(np.isfinite(ei.params_std_)):
                params_list.append(ei.params_.copy())
                stds_list.append(ei.params_std_.copy())
        except Exception:
            pass

    if not params_list:
        raise RuntimeError(f"All trials failed for {cfg.name} {cfg.estimator_cls.__name__}")

    pa = np.array(params_list)
    sa = np.array(stds_list)
    mean_p = pa.mean(axis=0)
    mean_s = sa.mean(axis=0)
    bias = (mean_p - cfg.true_params) / np.abs(cfg.true_params) * 100
    cov = (np.abs(pa - cfg.true_params) < sa).mean(axis=0)

    method_label = cfg.method if cfg.method == "analytical" else f"mc({cfg.mc_n_iter})"

    return BenchmarkResult(
        name=cfg.name,
        description=cfg.description,
        model_str=cfg.model_str,
        n_samples=cfg.n_samples,
        estimator=cfg.estimator_cls.__name__,
        method=method_label,
        param_names=cfg.param_names,
        true_params=cfg.true_params,
        mean_params=mean_p,
        mean_std=mean_s,
        bias_pct=bias,
        coverage_68=cov,
        runtime_ms=runtime_ms,
        n_trials_ok=len(params_list),
        note=cfg.note,
    )


# ─── scenario definitions ─────────────────────────────────────────────────────

def _build_configs() -> List[BenchmarkConfig]:
    configs = []

    def add(name, desc, model_str, pnames, true_p, p0, n, cls, model_fn, data_fn,
            note="", mc_n_iter=200):
        for method, n_trials in [("analytical", 200), ("mc", 10)]:
            configs.append(BenchmarkConfig(
                name=name, description=desc, model_str=model_str,
                param_names=pnames, true_params=np.array(true_p),
                p0=np.array(p0), n_samples=n, estimator_cls=cls,
                method=method, model_fn=model_fn, data_fn=data_fn,
                n_trials=n_trials, mc_n_iter=mc_n_iter, note=note,
            ))

    def add_single(name, desc, model_str, pnames, true_p, p0, n, cls, method,
                   model_fn, data_fn, n_trials=200, mc_n_iter=200, note=""):
        configs.append(BenchmarkConfig(
            name=name, description=desc, model_str=model_str,
            param_names=pnames, true_params=np.array(true_p),
            p0=np.array(p0), n_samples=n, estimator_cls=cls,
            method=method, model_fn=model_fn, data_fn=data_fn,
            n_trials=n_trials, mc_n_iter=mc_n_iter, note=note,
        ))

    # ── S1: Linear calibration, homoscedastic ─────────────────────────────────
    add("S1", "Linear calibration — homoscedastic σ_y=0.5, n=50",
        "y = a + b·x", ["a", "b"], [1.0, 2.0], [0.0, 0.0], 50,
        WeightedRegressor, _linear, _gen_linear_homo)

    # ── S2: Linear calibration, heteroscedastic ───────────────────────────────
    add("S2", "Linear calibration — heteroscedastic σ_y=0.1+0.1x, n=50",
        "y = a + b·x", ["a", "b"], [1.0, 2.0], [0.0, 0.0], 50,
        WeightedRegressor, _linear, _gen_linear_hetero)

    # ── S3: Exponential decay, Poisson counting ───────────────────────────────
    add("S3", "Radioactive decay — Poisson y-errors √A(t), n=20",
        "A(t) = A₀·exp(−λt)", ["A₀", "λ"], [1000.0, 0.05], [800.0, 0.03], 20,
        WeightedRegressor, _exponential, _gen_exp_decay)

    # ── S4: Power law (anomalous diffusion) ───────────────────────────────────
    add("S4", "Anomalous diffusion power law — σ_y=8%·y, n=30",
        "MSD = D·t^α", ["D", "α"], [2.0, 1.5], [1.5, 1.2], 30,
        WeightedRegressor, _power_law, _gen_power_law)

    # ── S5: Gaussian spectral peak, photon counting ───────────────────────────
    add("S5", "Spectral peak (Gaussian) — photon counting, n=40",
        "I = A·exp(-(x-μ)²/2σ²)", ["A", "μ", "σ"],
        [500.0, 5.0, 0.8], [400.0, 4.8, 1.0], 40,
        WeightedRegressor, _gaussian_peak, _gen_gaussian_peak)

    # ── S6: Damped oscillator (NMR / mechanical) ──────────────────────────────
    add("S6", "Damped oscillator (NMR/vibration) — σ_y=0.05, n=60",
        "y = A·exp(−γt)·sin(ωt+φ)", ["A", "γ", "ω", "φ"],
        [3.0, 0.4, 5.0, 0.3], [2.5, 0.3, 5.2, 0.0], 60,
        WeightedRegressor, _damped_sine, _gen_damped_sine, mc_n_iter=300)

    # ── S7: Exponential decay + timing errors ─────────────────────────────────
    # comparison: WeightedRegressor (wrong) vs XYWeightedRegressor (correct)
    add_single("S7", "Decay + timing errors σ_t=2s — WeightedRegressor (ignores x-err)",
               "A(t) = A₀·exp(−λt)", ["A₀", "λ"],
               [1000.0, 0.05], [800.0, 0.03], 20,
               WeightedRegressor, "analytical", _exponential,
               _strip_x_err(_gen_exp_timing), note="⚠ ignores x-errors")
    add("S7", "Decay + timing errors σ_t=2s — XYWeightedRegressor",
        "A(t) = A₀·exp(−λt)", ["A₀", "λ"],
        [1000.0, 0.05], [800.0, 0.03], 20,
        XYWeightedRegressor, _exponential, _gen_exp_timing)

    # ── S8: Hooke's law — both F and x measured ───────────────────────────────
    add_single("S8", "Hooke's law σ_x=0.02m σ_F=0.15N — WeightedRegressor (ignores x-err)",
               "F = F₀ + k·x", ["F₀", "k"],
               [0.1, 9.8], [0.0, 10.0], 25,
               WeightedRegressor, "analytical", _linear,
               _strip_x_err(_gen_hookes_law), note="⚠ ignores x-errors")
    add("S8", "Hooke's law σ_x=0.02m σ_F=0.15N — XYWeightedRegressor",
        "F = F₀ + k·x", ["F₀", "k"],
        [0.1, 9.8], [0.0, 10.0], 25,
        XYWeightedRegressor, _linear, _gen_hookes_law)

    # ── S9: Beer-Lambert absorbance vs concentration ──────────────────────────
    add_single("S9", "Beer-Lambert σ_c=2μg/mL σ_A=0.005 — WeightedRegressor (ignores x-err)",
               "A = ε·c + baseline", ["baseline", "ε"],
               [0.02, 0.05], [0.0, 0.04], 20,
               WeightedRegressor, "analytical", _linear,
               _strip_x_err(_gen_beer_lambert), note="⚠ ignores x-errors")
    add("S9", "Beer-Lambert σ_c=2μg/mL σ_A=0.005 — XYWeightedRegressor",
        "A = ε·c + baseline", ["baseline", "ε"],
        [0.02, 0.05], [0.0, 0.04], 20,
        XYWeightedRegressor, _linear, _gen_beer_lambert)

    # ── S10: Method comparison — OLS vs Deming ────────────────────────────────
    add_single("S10", "Method comparison σ_x=5 σ_y=3 — WeightedRegressor (OLS bias)",
               "y_new = a + b·y_old", ["a", "b"],
               [3.0, 1.05], [0.0, 1.0], 30,
               WeightedRegressor, "analytical", _linear,
               _strip_x_err(_method_comparison), note="⚠ OLS attenuation bias")
    add("S10", "Method comparison σ_x=5 σ_y=3 — DemingRegressor",
        "y_new = a + b·y_old", ["a", "b"],
        [3.0, 1.05], [0.0, 1.0], 30,
        DemingRegressor, _linear, _method_comparison)

    # ── S11: Isotope ratio (mass spectrometry) ────────────────────────────────
    add_single("S11", "Isotope ratio MS σ_x=σ_y≈0.003–0.004 — WeightedRegressor (ignores x-err)",
               "δ_sample = a + b·δ_ref", ["a", "b"],
               [0.001, 0.9878], [0.0, 1.0], 20,
               WeightedRegressor, "analytical", _linear,
               _strip_x_err(_gen_isotope_ratio), note="⚠ ignores x-errors")
    add("S11", "Isotope ratio MS σ_x=σ_y≈0.003–0.004 — DemingRegressor",
        "δ_sample = a + b·δ_ref", ["a", "b"],
        [0.001, 0.9878], [0.0, 1.0], 20,
        DemingRegressor, _linear, _gen_isotope_ratio)

    # ── S12: Small sample ─────────────────────────────────────────────────────
    add("S12", "Small sample — y = a + b·x, n=8, σ_y=0.5 (coverage stress test)",
        "y = a + b·x", ["a", "b"], [1.0, 2.0], [0.0, 0.0], 8,
        WeightedRegressor, _linear, _gen_linear_small)

    # ── S13: Low SNR ──────────────────────────────────────────────────────────
    add("S13", "Low SNR — y = a + b·x, σ_y=3.0 (signal range 1–21), n=30",
        "y = a + b·x", ["a", "b"], [1.0, 2.0], [0.0, 0.0], 30,
        WeightedRegressor, _linear, _gen_linear_low_snr)

    return configs


def _method_comparison(rng):
    return _gen_method_comparison(rng)


# ─── printing ─────────────────────────────────────────────────────────────────

_COV_IDEAL = 0.683


def _cov_mark(c: float) -> str:
    if 0.58 <= c <= 0.78:
        return "✓"
    if 0.48 <= c <= 0.88:
        return "~"
    return "✗"


def _param_summary(result: BenchmarkResult) -> str:
    parts = []
    for name, b, c in zip(result.param_names, result.bias_pct, result.coverage_68):
        parts.append(f"{name}: {b:+.1f}% / {_cov_mark(c)}{c*100:.0f}%")
    return "  ".join(parts)


def _print_results(results: List[BenchmarkResult]) -> None:
    W = 110
    print("\n" + "═" * W)
    print(f"{'MCUP Benchmark Suite':^{W}}")
    print(f"{'Metrics: bias=(mean_estimated−true)/true  |  coverage=fraction of trials where |est−true| < 1σ':^{W}}")
    print("═" * W)
    print(f"Coverage calibration: ✓ within ±10pp of ideal 68%  |  ~ within ±20pp  |  ✗ outside ±20pp\n")

    by_name: dict = {}
    for r in results:
        by_name.setdefault(r.name, []).append(r)

    for sname, group in by_name.items():
        r0 = group[0]
        base_desc = r0.description.split("—")[0].strip()
        print(f"[{sname}]  {base_desc}")
        print(f"       Model: {r0.model_str}  |  n={r0.n_samples}  |  True: " +
              ", ".join(f"{p}={v}" for p, v in zip(r0.param_names, r0.true_params)))
        print(f"  {'─'*25} {'─'*14} {'─'*50} {'─'*9} {'─'*6}")
        print(f"  {'Estimator':<25} {'Method':<14} {'Bias / Coverage per param':<50} {'Time':>9} {'Trials':>6}")
        print(f"  {'─'*25} {'─'*14} {'─'*50} {'─'*9} {'─'*6}")
        for r in group:
            note_str = f"  {r.note}" if r.note else ""
            param_str = _param_summary(r)
            print(f"  {r.estimator:<25} {r.method:<14} {param_str:<50} {r.runtime_ms:>8.1f}ms {r.n_trials_ok:>6}{note_str}")
        print()

    _print_summary(results)


def _print_summary(results: List[BenchmarkResult]) -> None:
    W = 110
    print("═" * W)
    print(f"{'Summary: worst-case coverage deviation from ideal 68%':^{W}}")
    print("═" * W)

    rows = []
    for r in results:
        if r.note:
            continue
        worst_dev = max(abs(c - _COV_IDEAL) for c in r.coverage_68)
        max_abs_bias = max(abs(b) for b in r.bias_pct)
        rows.append((r.name, r.estimator, r.method, max_abs_bias, worst_dev,
                     r.runtime_ms, r.n_trials_ok))

    rows.sort(key=lambda x: x[4])

    print(f"  {'Scenario':<6} {'Estimator':<25} {'Method':<14} {'Max|bias|':>10} {'Max cov dev':>12} {'Time':>9}")
    print(f"  {'─'*6} {'─'*25} {'─'*14} {'─'*10} {'─'*12} {'─'*9}")
    for name, est, method, bias, dev, rt, _ in rows:
        mark = "✓" if dev <= 0.10 else ("~" if dev <= 0.20 else "✗")
        print(f"  {name:<6} {est:<25} {method:<14} {bias:>9.1f}% {mark}{dev*100:>10.1f}pp {rt:>8.1f}ms")
    print()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    configs = _build_configs()
    results = []
    total = len(configs)

    print(f"\nRunning {total} benchmark configurations...\n")

    for i, cfg in enumerate(configs, 1):
        label = f"[{i}/{total}] {cfg.name} {cfg.estimator_cls.__name__} {cfg.method}"
        print(f"  {label}", end="", flush=True)
        try:
            r = _run(cfg)
            results.append(r)
            print(f"  → {r.runtime_ms:.1f}ms  (n_ok={r.n_trials_ok})")
        except Exception as e:
            print(f"  → FAILED: {e}")

    _print_results(results)


if __name__ == "__main__":
    main()
