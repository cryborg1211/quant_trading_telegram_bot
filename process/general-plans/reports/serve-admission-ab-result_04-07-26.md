# Serve-Mirror Admission A/B — Result

- **Run date:** 04-07-26 19:25
- **Plan:** `process/general-plans/active/serve-admission-tranche-ab_PLAN_04-07-26.md` (Task A)
- **Checkpoint:** `models/saved/v3_training_checkpoint.joblib` (T+20, seeds=[42, 43, 44, 45])
- **OOS cutoff:** 2022-11-02
- **Command:** `python scripts/ab_serve_admission.py`
- **DSR convention:** n_trials=28 (7 configs x 4 seeds)

## Comparison table (4-seed means)

| # | Config | admission_mode | floor | top-N | Mean Net PnL (VND) | Mean Sharpe | Mean MaxDD | Mean Zero-Cand Days | Mean Gross Exp | Seeds OK |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1. Baseline (incumbent) | cross_sectional | n/a | 5 | +2,818,603,884 | +0.629 | -13.00% | 0.0 | 0.441 | 4 |
| 2 | 2. Serve-mirror @0.45/N5 | absolute_gate | 0.45 | 5 | +2,629,294,153 | +0.592 | -13.47% | 8.0 | 0.440 | 4 |
| 3 | 3. Serve-mirror @0.45/N3 | absolute_gate | 0.45 | 3 | +2,442,567,177 | +0.549 | -13.88% | 8.0 | 0.444 | 4 |
| 4 | 4. Floor sweep @0.40/N3 | absolute_gate | 0.40 | 3 | +2,548,374,337 | +0.568 | -13.44% | 0.0 | 0.447 | 4 |
| 5 | 5. Floor sweep @0.40/N5 | absolute_gate | 0.40 | 5 | +2,818,603,884 | +0.629 | -13.00% | 0.0 | 0.441 | 4 |
| 6 | 6. Floor sweep @0.35/N3 | absolute_gate | 0.35 | 3 | +2,548,374,337 | +0.568 | -13.44% | 0.0 | 0.447 | 4 |
| 7 | 7. Floor sweep @0.35/N5 **<= WINNER** | absolute_gate | 0.35 | 5 | +2,905,539,059 | +0.644 | -13.00% | 0.0 | 0.441 | 4 |

**Winner by mean Net PnL:** 7. Floor sweep @0.35/N5

- **DSR (winner, n_trials=28):** SR=+0.668, SR0=+1.074, p_dsr=0.2241 (FAIL <0.95)
- **PBO (CSCV, winner per-seed):** 85.6% (FAIL >10%)

## Monthly NET RETURN by config (4-seed mean)

Isolates narrow-leadership / negative-breadth months in the OOS window.

```
Month            1        2        3        4        5        6        7
------------------------------------------------------------------------
2022-11      3.34%    3.34%    3.50%    3.50%    3.34%    3.50%    3.34%
2022-12     -4.10%   -4.22%   -4.59%   -4.55%   -4.10%   -4.55%   -4.10%
2023-01      9.90%    9.84%    9.21%    9.12%    9.90%    9.12%    9.90%
2023-02     -6.02%   -5.72%   -5.78%   -6.02%   -6.02%   -6.02%   -6.02%
2023-03      3.56%    3.90%    4.12%    3.90%    3.56%    3.90%    3.56%
2023-04      1.37%    1.27%    1.05%    1.06%    1.37%    1.06%    1.37%
2023-05      1.60%    1.67%    1.77%    1.73%    1.60%    1.73%    1.60%
2023-06      0.70%    0.67%    0.72%    0.75%    0.70%    0.75%    0.70%
2023-07      3.19%    3.30%    3.16%    3.11%    3.19%    3.11%    3.19%
2023-08      0.83%    0.81%    0.69%    0.71%    0.83%    0.71%    0.83%
2023-09     -5.27%   -5.28%   -5.51%   -5.51%   -5.27%   -5.51%   -5.27%
2023-10     -6.95%   -7.02%   -7.07%   -7.03%   -6.95%   -7.03%   -6.95%
2023-11      7.29%    7.51%    7.66%    7.56%    7.29%    7.56%    7.29%
2023-12      3.64%    3.68%    3.75%    3.72%    3.64%    3.72%    3.64%
2024-01      0.07%    0.16%    0.09%    0.11%    0.07%    0.11%    0.07%
2024-02      3.74%    3.76%    3.73%    3.74%    3.74%    3.74%    3.74%
2024-03      2.62%    2.61%    2.62%    2.65%    2.62%    2.65%    2.62%
2024-04     -6.41%   -6.60%   -6.96%   -6.76%   -6.41%   -6.76%   -6.41%
2024-05      3.78%    3.67%    3.70%    3.75%    3.78%    3.75%    3.78%
2024-06     -0.02%   -0.00%   -0.01%   -0.03%   -0.02%   -0.03%   -0.02%
2024-07     -0.52%   -0.53%   -0.55%   -0.52%   -0.52%   -0.52%   -0.52%
2024-08      1.13%    1.11%    0.95%    0.89%    1.13%    0.89%    1.13%
2024-09      1.34%    1.47%    1.28%    1.24%    1.34%    1.24%    1.34%
2024-10     -2.70%   -2.65%   -2.53%   -2.55%   -2.70%   -2.55%   -2.70%
2024-11     -1.81%   -1.81%   -1.88%   -1.90%   -1.81%   -1.90%   -1.81%
2024-12      0.48%    0.47%    0.46%    0.47%    0.48%    0.47%    0.48%
2025-01     -0.38%   -0.41%   -0.45%   -0.45%   -0.38%   -0.45%   -0.38%
2025-02      3.29%    3.32%    3.57%    3.57%    3.29%    3.57%    3.29%
2025-03     -0.83%   -0.92%   -1.11%   -1.07%   -0.83%   -1.07%   -0.83%
2025-04     -2.82%   -3.12%   -3.35%   -3.06%   -2.82%   -3.06%   -2.79%
2025-05      6.31%    6.17%    7.03%    7.10%    6.31%    7.10%    6.39%
2025-06      0.91%    0.89%    0.67%    0.67%    0.91%    0.67%    0.91%
2025-07     10.71%   10.77%   11.37%   11.36%   10.71%   11.36%   10.70%
2025-08      5.31%    5.26%    5.23%    5.24%    5.31%    5.24%    5.31%
2025-09     -1.95%   -2.23%   -2.34%   -2.25%   -1.95%   -2.25%   -1.95%
2025-10     -1.76%   -1.74%   -1.75%   -1.70%   -1.76%   -1.70%   -1.76%
2025-11     -0.80%   -0.99%   -0.96%   -0.88%   -0.80%   -0.88%   -0.80%
2025-12     -1.23%   -1.31%   -1.57%   -1.61%   -1.23%   -1.61%   -1.23%
2026-01      0.38%    0.39%    0.35%    0.31%    0.38%    0.31%    0.37%
2026-02      3.57%    3.73%    4.33%    4.31%    3.57%    4.31%    3.57%
2026-03     -5.33%   -5.78%   -6.09%   -5.78%   -5.33%   -5.78%   -5.33%
2026-04      0.89%    0.32%    0.39%    0.69%    0.89%    0.69%    1.30%
2026-05     -1.55%   -1.55%   -1.84%   -1.78%   -1.55%   -1.78%   -1.36%
2026-06     -1.26%   -1.34%   -1.36%   -1.37%   -1.26%   -1.37%   -1.25%
2026-07     -0.09%   -0.12%   -0.17%   -0.16%   -0.09%   -0.16%   -0.09%
```

## Raw stdout comparison table

```
Config                        NetPnL(VND)   Sharpe     MaxDD  ZeroCandDay  MeanGross   N
----------------------------------------------------------------------------------------
1. Baseline (incumbent)    +2,818,603,884   +0.629   -13.00%          0.0      0.441   4  
2. Serve-mirror @0.45/N5   +2,629,294,153   +0.592   -13.47%          8.0      0.440   4  
3. Serve-mirror @0.45/N3   +2,442,567,177   +0.549   -13.88%          8.0      0.444   4  
4. Floor sweep @0.40/N3    +2,548,374,337   +0.568   -13.44%          0.0      0.447   4  
5. Floor sweep @0.40/N5    +2,818,603,884   +0.629   -13.00%          0.0      0.441   4  
6. Floor sweep @0.35/N3    +2,548,374,337   +0.568   -13.44%          0.0      0.447   4  
7. Floor sweep @0.35/N5    +2,905,539,059   +0.644   -13.00%          0.0      0.441   4 *

(* = winner by mean Net PnL; NetPnL/Sharpe/MaxDD/ZeroCandDay/MeanGross are 4-seed means. N = seeds OK.)
```

## Decision-rule read (Task B gate — see plan)

- Baseline (cross_sectional) mean Sharpe: +0.629, mean MaxDD: -13.00%
- Best absolute_gate config: 7. Floor sweep @0.35/N5 (Sharpe +0.644, MaxDD -13.00%)
- Best-gate Sharpe delta vs baseline: +0.015
- Does the gate earn its lost return via lower risk? NO (MaxDD not better)
- Max zero-candidate-day fraction (any gate config): 0.9% of OOS days

**Read:** Evidence does NOT clearly support a Task-B serve-path change. The absolute gate is close to (or better than) baseline on risk-adjusted terms and/or does not block a material fraction of days. Serve's current defensiveness — while it happened to block June's SSB winner — is not measurably costly on average across this OOS sample. File the June episode as an acceptable false-defensive instance; no serve-path change is warranted from this evidence alone.

_Wall-clock: 1937.0s._
