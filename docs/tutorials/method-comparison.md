# Tutorial 3: Comparing Two Measurement Methods with Deming Regression

## The problem

You have developed a new spectroscopic method for measuring protein concentration. A colleague has validated the old method (Bradford assay) for years — it's trusted but slow. You want to show your new method agrees with the old one so you can replace it.

You measure the same 60 samples with both methods. The result is a scatter plot of `y = new method` vs `x = old method`. If they agree perfectly, the points lie on `y = x` (identity line). The true relationship has a 5% systematic gain: `y = 3 + 1.05·x`.

**The problem:** both measurements are noisy. The old method (Bradford assay) has `σ_x = 25 μg/mL` — pipetting variability and absorbance noise combine to large errors when samples span the full 10–200 μg/mL range. The new spectroscopic method has `σ_y = 8 μg/mL`.

When you apply OLS to this data, regression dilution (attenuation bias) pulls the slope toward zero. With `σ_x = 25` on a 190 μg/mL range, OLS reports `b ≈ 0.86` instead of the true 1.05 — a **19% underestimate**. You'd conclude the methods disagree when they actually agree.

---

## The attenuation bias formula

For a linear model with x-errors, OLS converges to:

```
b_OLS → b_true × σ²_xtrue / (σ²_xtrue + σ²_xerr)
```

With a concentration range of ~95 μg/mL (half-range ≈ 60), `σ_xtrue ≈ 55 μg/mL`:

```
b_OLS → 1.05 × 55² / (55² + 25²) ≈ 1.05 × 0.83 ≈ 0.87
```

This matches what we observe. Deming regression eliminates the bias by treating both variables as noisy observations of the same latent true value.

---

## Setup

```python
import numpy as np
from mcup import DemingRegressor, WeightedRegressor

np.random.seed(7)

# True concentrations (unknown in practice)
c_true = np.random.uniform(10, 200, 60)

# True relationship: new method reads 5% higher (systematic gain)
a_true, b_true = 3.0, 1.05

# Large measurement noise: Bradford assay is imprecise at this scale
sigma_x = 25.0   # old method — pipetting + absorbance noise
sigma_y = 8.0    # new method — spectroscopic readout

x_meas = c_true + np.random.normal(0, sigma_x, 60)
y_meas = a_true + b_true * c_true + np.random.normal(0, sigma_y, 60)

x_err = sigma_x * np.ones(60)
y_err = sigma_y * np.ones(60)
```

---

## OLS: biased by regression dilution

```python
def line(x, p):
    return p[0] + p[1] * x

est_ols = WeightedRegressor(line, method="analytical")
est_ols.fit(x_meas, y_meas, y_err=y_err, p0=[0.0, 1.0])

bias = (est_ols.params_[1] - b_true) / b_true * 100
print(f"OLS   b = {est_ols.params_[1]:.3f} ± {est_ols.params_std_[1]:.3f}  (bias: {bias:+.0f}%)")
```

```
OLS   b = 0.856 ± 0.019  (bias: -19%)
```

The slope is biased by −19%. Your report would say the methods disagree (b ≪ 1.05), and the new method appears to systematically underestimate relative to the old one. You might shelve a perfectly good method.

---

## Deming regression: correct approach

```python
est_dem = DemingRegressor(line, method="analytical")
est_dem.fit(x_meas, y_meas, x_err=x_err, y_err=y_err, p0=[0.0, 1.0])

print(f"Deming  b = {est_dem.params_[1]:.3f} ± {est_dem.params_std_[1]:.3f}")
print(f"True    b = {b_true}")
```

```
Deming  b = 1.058 ± 0.073
True    b = 1.05
```

The slope is now 1.058 — within 0.8% of the true value. The uncertainty correctly reflects that both instruments are noisy.

---

## Results at a glance

![Method comparison](../assets/tutorial3_method_comparison.png)

The left panel tells the story at a glance. The OLS line (dashed orange) is pulled away from both the true relationship and the identity line. The Deming line (solid green) passes through the data scatter correctly. The right panel confirms: OLS slope biased −19%, Deming recovers truth.

| | True | OLS (ignores σ_x) | DemingRegressor |
|---|---|---|---|
| Intercept a (μg/mL) | 3.00 | large bias | close to truth |
| Slope b | 1.050 | **0.856 (−19% bias)** | **1.058 ✓** |

The −19% bias is large enough to matter clinically. If the old method reads 100 μg/mL and your OLS calibration gives `y = 0.86·x + C`, you'd report 86 μg/mL — potentially a misdiagnosis in a concentration-dependent assay.

---

## Interpreting the Deming result

```python
a, b = est_dem.params_
a_std, b_std = est_dem.params_std_

# Does slope = 1 (proportional agreement)?
t_slope = abs(b - 1.0) / b_std
print(f"Slope t-test (H0: b=1): t = {t_slope:.2f}")
print(f"  {'Cannot reject' if t_slope < 2 else 'Reject'} H0 at 95% confidence")
```

```
Slope t-test (H0: b=1): t = 0.79
  Cannot reject H0 at 95% confidence
```

With Deming, the test correctly does not reject b = 1. With OLS, the test would reject it — a false conclusion driven entirely by measurement error in x.

---

## Sensitivity to the assumed error ratio

Deming regression depends on the variance ratio `λ = σ_y² / σ_x²`. Using the correct uncertainties is essential:

```python
for assumed_sigma_x in [5.0, 15.0, 25.0, 40.0]:
    x_err_assumed = assumed_sigma_x * np.ones(60)
    est_test = DemingRegressor(line, method="analytical")
    est_test.fit(x_meas, y_meas, x_err=x_err_assumed, y_err=y_err, p0=[0.0, 1.0])
    print(f"σ_x assumed={assumed_sigma_x:.0f}:  slope={est_test.params_[1]:.3f}")
```

```
σ_x assumed=5:   slope=0.892  ← under-corrects, still biased
σ_x assumed=15:  slope=0.988  ← better
σ_x assumed=25:  slope=1.058  ← correct
σ_x assumed=40:  slope=1.121  ← over-corrects
```

This confirms that knowing your measurement uncertainties is not optional — it is the input. Rough estimates of `σ_x` are still far better than assuming `σ_x = 0` (OLS).

---

## Monte Carlo for robustness

With 60 samples and this noise level, the analytical covariance is well-conditioned. The MC solver provides a non-parametric check:

```python
est_mc = DemingRegressor(line, method="mc", n_iter=2000)
np.random.seed(7)
est_mc.fit(x_meas, y_meas, x_err=x_err, y_err=y_err, p0=[0.0, 1.0])

print(f"MC Deming  b = {est_mc.params_[1]:.3f} ± {est_mc.params_std_[1]:.3f}")
```

If MC and analytical agree, you can trust the analytical result. If they diverge, the problem is poorly constrained and you need more data or tighter x-measurements.

---

## Key takeaways

- OLS in method comparison studies **always underestimates the slope** when x has noise — by 19% here with typical assay precision.
- `DemingRegressor` recovers the unbiased slope and correctly reports wider uncertainties that reflect both instruments' noise.
- The bias grows with `σ_x / σ_xtrue` — the larger the x-error relative to the spread of true x-values, the worse the attenuation.
- Classic uses: method comparison in analytical chemistry, clinical diagnostics, metrology, instrument cross-validation.
