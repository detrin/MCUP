# Choosing an Estimator

MCUP provides three estimators targeting different error scenarios. Use this guide to pick the right one.

---

## Decision tree

```
Do your x measurements have significant errors?
│
├─ No  → WeightedRegressor
│
└─ Yes → Do you need the most exact treatment?
         │
         ├─ No  (mild nonlinearity, speed matters) → XYWeightedRegressor
         │
         └─ Yes (large x errors, strong nonlinearity) → DemingRegressor
```

---

## WeightedRegressor

**Use when:** only y has measurement errors; x is known exactly (or its errors are negligible).

Minimises the weighted chi-squared objective:

```
Σ (y_i - f(x_i, β))² / σ_yi²
```

This is the standard weighted least squares formulation and is the fastest option.

---

## XYWeightedRegressor

**Use when:** both x and y carry measurement errors, and your model is mildly nonlinear.

Combines x and y variances via first-order error propagation:

```
σ_combined,i² = σ_yi² + (∂f/∂x_i)² · σ_xi²
```

The gradient `∂f/∂x` is computed numerically at each iteration. The fit is iterated
(IRLS — Iteratively Reweighted Least Squares) until convergence, typically in ~10 steps.

**Tradeoff vs DemingRegressor:** faster and simpler, but the first-order propagation
becomes inaccurate when x errors are large or the model curves sharply.

---

## DemingRegressor

**Use when:** both x and y have errors and you need the most rigorous treatment.

Performs joint optimisation over model parameters *and* latent true x values `η`:

```
min_{β, η}  Σ (x_i - η_i)²/σ_xi²  +  Σ (y_i - f(η_i, β))²/σ_yi²
```

This is total least squares (Deming regression) generalised to arbitrary nonlinear models.

**Tradeoff vs XYWeightedRegressor:** more accurate when x errors are large, but
the parameter space grows with the number of data points, making each optimisation
step slower.

---

## Analytical vs Monte Carlo solver

Both solvers are available for all three estimators via the `method` argument.

| | `method="analytical"` | `method="mc"` |
|---|---|---|
| **Speed** | Fast — one optimisation pass | Slower — thousands of optimisation runs |
| **Covariance** | `(J^T W J)^{-1}` Jacobian approximation | Welford online covariance over MC samples |
| **Best for** | Linear or mildly nonlinear models | Strongly nonlinear models |
| **Robustness** | May underestimate errors if model is nonlinear | Correct by construction regardless of nonlinearity |
| **Convergence control** | Not applicable | `rtol` / `atol` stopping criteria available |

### When to prefer `"mc"`

- Your model is strongly nonlinear (e.g. exponentials, power laws, oscillatory functions)
- You want confidence intervals that don't rely on the linear approximation
- You need to verify that analytical results are trustworthy for your model

### When to prefer `"analytical"`

- Your model is linear or close to linear
- Speed is important (e.g. fitting many datasets in a loop)
- You're doing a quick exploratory fit and can validate later with MC
