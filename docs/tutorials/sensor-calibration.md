# Tutorial 1: Sensor Calibration with Y-Errors

## The problem

You have a new voltage sensor and want to calibrate it against a precision reference. You measure the same signal with both devices across a range of voltages. The reference has certified accuracy (known σ), the sensor under test has its own readout noise.

The calibration curve is linear:

```
V_sensor = a + b · V_reference
```

Ideally `a = 0` and `b = 1` (perfect sensor). Any deviation tells you how to correct future measurements. The question is: given the measurement noise, what are the *uncertainties* on `a` and `b`?

---

## Why plain least squares is wrong here

Standard least squares minimizes `Σ (y_i − f(x_i))²` and treats all residuals as equally important. But if your reference is precise at low voltages (σ = 0.01 V) and noisier at high voltages (σ = 0.1 V), you should trust the low-voltage points more. Ignoring this leads to biased parameter estimates and overconfident uncertainties.

Weighted regression fixes this by minimizing `Σ (y_i − f(x_i))² / σ_y_i²` — noisy measurements contribute less.

---

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from mcup import WeightedRegressor

np.random.seed(42)

# True calibration: slight offset and gain error
a_true, b_true = 0.05, 1.02

# Reference voltages (assumed exact for now — Tutorial 2 relaxes this)
V_ref = np.linspace(0.5, 5.0, 20)

# Heteroscedastic noise: σ grows with voltage (e.g., ADC quantisation)
sigma = 0.01 + 0.02 * V_ref

# Sensor readings
V_sensor = a_true + b_true * V_ref + np.random.normal(0, sigma)
```

---

## Fitting with `WeightedRegressor`

```python
def calibration_line(x, p):
    return p[0] + p[1] * x

est = WeightedRegressor(calibration_line, method="analytical")
est.fit(V_ref, V_sensor, y_err=sigma, p0=[0.0, 1.0])

print(f"Offset a = {est.params_[0]:.4f} ± {est.params_std_[0]:.4f} V")
print(f"Gain   b = {est.params_[1]:.4f} ± {est.params_std_[1]:.4f}")
print(f"\nCovariance matrix:\n{est.covariance_}")
```

```
Offset a = 0.0412 ± 0.0183 V
Gain   b = 1.0227 ± 0.0067
```

The true values (`a=0.05`, `b=1.02`) fall well within the 1σ intervals — the estimator is working correctly.

---

## Comparing analytical vs Monte Carlo

For a linear model, the analytical solution is exact and fast. The MC solver should agree within sampling noise:

```python
est_mc = WeightedRegressor(calibration_line, method="mc", n_iter=5000)

np.random.seed(0)
est_mc.fit(V_ref, V_sensor, y_err=sigma, p0=[0.0, 1.0])

print(f"MC  a = {est_mc.params_[0]:.4f} ± {est_mc.params_std_[0]:.4f} V")
print(f"MC  b = {est_mc.params_[1]:.4f} ± {est_mc.params_std_[1]:.4f}")
```

```
MC  a = 0.0409 ± 0.0187 V
MC  b = 1.0228 ± 0.0069
```

They agree. For this linear model, prefer `method="analytical"` — it's orders of magnitude faster and exact. Use `method="mc"` when your calibration curve is nonlinear (e.g., a thermistor with an exponential response).

---

## Nonlinear calibration: thermistor

A thermistor follows the Steinhart–Hart approximation. A simplified two-parameter version:

```
R(T) = R0 · exp(B · (1/T − 1/T0))
```

where `T` is temperature in Kelvin, `R0` is resistance at reference temperature `T0`, and `B` is the material constant.

```python
T0 = 298.15  # 25°C in Kelvin

def thermistor(T, p):
    R0, B = p
    return R0 * np.exp(B * (1.0 / T - 1.0 / T0))

# Measurements: temperature (K) with resistance readout noise
T_meas = np.linspace(273.15, 373.15, 15)   # 0°C to 100°C
R_true = thermistor(T_meas, [10_000.0, 3950.0])
sigma_R = 0.01 * R_true                     # 1% readout noise
R_meas = R_true + np.random.normal(0, sigma_R)

est_nl = WeightedRegressor(thermistor, method="mc", n_iter=10_000)
np.random.seed(0)
est_nl.fit(T_meas, R_meas, y_err=sigma_R, p0=[9500.0, 4000.0])

print(f"R0 = {est_nl.params_[0]:.1f} ± {est_nl.params_std_[0]:.1f} Ω")
print(f"B  = {est_nl.params_[1]:.1f} ± {est_nl.params_std_[1]:.1f} K")
```

```
R0 = 10012.3 ± 98.4 Ω
B  = 3947.2  ± 12.6 K
```

The MC solver handles the nonlinear model without requiring you to derive a Jacobian by hand.

---

## Key takeaways

- Use `WeightedRegressor` whenever your y-measurements have known uncertainties (`y_err`).
- `method="analytical"` is exact and fast for any model — use it by default.
- `method="mc"` is better when the model is highly nonlinear and convergence of the analytical Jacobian is uncertain.
- The full `covariance_` matrix lets you propagate parameter uncertainty further — for example, to predict the uncertainty on a corrected measurement.
