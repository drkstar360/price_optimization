# Optimization Approach Explanation

## 1. Objective Function and Its Appropriateness

**Objective Function:**
Minimize the **sum of squared errors (SSE)** between the PMS rule engine prices and the target prices in the 30×30 CSV matrix:

```
SSE = sum_over_(d,k in ValidStays) [ PMS_Price(d, k) - Target_Price(d, k) ]^2
```

- `d`: check-in day (1 to 30)
- `k`: length of stay (1 to 30)
- `ValidStays`: all (d, k) where the stay ends within the month (i.e., non-blank in the CSV)
- `PMS_Price(d, k)`: computed using the current base rates and discounts
- `Target_Price(d, k)`: from the CSV

**Why appropriate?**
- Standard for regression: SSE is the classic loss for fitting models to data.
- Penalizes large errors: Squaring emphasizes outliers, so the optimizer focuses on reducing big mismatches.
- Differentiable: Required for gradient-based solvers.
- Directly measures fit: Lower SSE means your PMS engine's output is closer to the real/target prices.

---

## 2. Decision Variables and Constraints

**Decision Variables:**
- **Base rates:** 30 variables, one for each day (`R_1, R_2, ..., R_30`), representing the "rack rate" for each night.
- **Discounts:** 30 × 8 = 240 variables, one for each day and each of the 8 cut-off tiers (`D_d,1, ..., D_d,8`), representing the percentage discount for each cut-off on each night.

**Constraints:**
- **Bounds:**
  - Base rates: `1.0 <= R_d <= 1500.0`
  - Discounts: `0.0 <= D_d,j <= 0.999` (0% to 99.9%)
- **Monotonicity (strictly increasing discounts):**
  - For each day d: `D_d,1 < D_d,2 < ... < D_d,8`
  - Enforced numerically as `D_d,j+1 - D_d,j >= epsilon` for a small epsilon (e.g., 1e-7)
- **Feasibility:** Only stays that end within the month are considered (others are ignored/NaN).

---

## 3. Algorithm or Optimisation Technique

- **Type:** Nonlinear constrained optimization.
- **Tool:** `scipy.optimize.minimize` (Python), using the SLSQP (Sequential Least Squares Programming) algorithm.
- **Why SLSQP?**
  - Handles both bounds and nonlinear constraints.
  - Well-suited for problems with many variables and constraints.
- **How it works:**
  - Starts from an initial guess (base rates from 1-night stays, random strictly increasing discounts).
  - Iteratively adjusts variables to reduce SSE, while always respecting bounds and monotonicity constraints.
  - Uses numerical gradients (finite differences) to guide the search.

---

## 4. Computational Complexity and Practical Considerations

- **Variables:** 270 (30 base rates + 240 discounts)
- **Constraints:** 210 monotonicity constraints (30 days × 7 per day) + 270 bounds
- **Each function evaluation:**
  - Loops over all valid (d, k) pairs (465 in total), and for each, sums up k nights' prices.
- **Complexity:**
  - No closed-form; depends on number of iterations, function/gradient evaluations, and constraint checks.
  - Each iteration is moderately expensive due to the nested loops.
- **Practical considerations:**
  - **Convergence:**
    - May terminate at a local minimum or with a "positive directional derivative" message (can't improve further).
    - May not always strictly satisfy monotonicity due to numerical precision; post-processing can fix this.
  - **Performance:**
    - Can be slow for large problems; increasing max iterations or tightening constraints increases runtime.
    - Initial guess and constraint tightness affect convergence.
  - **Numerical issues:**
    - Floating-point precision can cause near-equality or slight violations in monotonicity.

---

## 5. Validation and Reporting of Residual Error

- **Validation:**
  - After optimization, the script checks for monotonicity violations and prints warnings if any are found.
  - Post-processing can be applied to strictly enforce monotonicity if needed.
  - All results are saved to CSV for inspection.
- **Reporting:**
  - **SSE (Sum of Squared Errors):** Directly reported; lower is better.
  - **RMSE (Root Mean Squared Error):**
    - `RMSE = sqrt(SSE / N)`, where N is the number of valid stays (465). This gives an average error per stay, in the same units as price.
  - **Console output:** Shows base rates, discounts, SSE, RMSE, and any monotonicity warnings.
  - **CSV output:** Both base rates and discounts (with day column) are saved for further analysis.

---