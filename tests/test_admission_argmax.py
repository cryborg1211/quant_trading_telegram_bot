"""Argmax admission mode — admit on the CLASS DECISION, not a P(UP) level.

WHY THIS MODE EXISTS (11-08-26, from live paperlog evidence)
───────────────────────────────────────────────────────────
`absolute_gate` admits on `p_up >= admission_floor`, but P(UP) turned out to be
anti-informative across exactly that range: the [0.2,0.3) bin realised a 45.9%
hit rate against 29.3% for [0.4,0.5), and the T+20 Platt slope is NEGATIVE
(−0.274). Rows where UP was the ARGMAX did beat the baseline (+1.19% vs
−2.43%). Argmax is also structurally immune to gate starvation: it compares the
three classes against one another instead of against a frozen absolute
threshold the output distribution can drift below (serve p90 fell 0.463 → 0.423
while tau stayed 0.46).

Fixture mirrors tests/test_admission_ab.py but with a THREE-CLASS oracle, since
argmax needs the class structure that a p_up-only oracle cannot express.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.walk_forward import WalkForwardConfig, WalkForwardEngine

N_DAYS = 20
HOLD = 5
PRICE = 20.0
MAX_POS = 2


def _panel(triples: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    """Constant-price panel carrying each ticker's [p_down, p_side, p_up]."""
    days = pd.bdate_range("2024-01-02", periods=N_DAYS).date
    frames = []
    for tk, (p_dn, p_sd, p_up) in triples.items():
        frames.append(pd.DataFrame({
            "ticker": tk, "date": days,
            "open": PRICE, "high": PRICE, "low": PRICE, "close": PRICE,
            "volume": 10_000_000,
            "f_dn": float(p_dn), "f_sd": float(p_sd), "f_up": float(p_up),
        }))
    return pd.concat(frames, ignore_index=True)


def _oracle3(X: np.ndarray) -> np.ndarray:
    """(n, 3) class probabilities read straight off the last timestep."""
    return X[:, -1, :3].astype(np.float64)


def _engine(*, admission_mode: str, admission_floor: float = 0.45,
            admission_pool_cap: int = 6) -> WalkForwardEngine:
    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["f_dn", "f_sd", "f_up"],
        rebalance_mode="tranche", tranche_hold_days=HOLD,
        max_positions=MAX_POS, signal_threshold=0.40,
        liquid_top_n=None, initial_capital=1_000_000_000.0,
        admission_mode=admission_mode, admission_floor=admission_floor,
        admission_pool_cap=admission_pool_cap,
    )
    return WalkForwardEngine(cfg, _oracle3)


def _bought(res) -> set[str]:
    fills = pd.DataFrame(res.fills)
    if fills.empty:
        return set()
    return set(fills.query("side == 'buy'")["ticker"])


def test_argmax_admits_only_names_whose_top_class_is_up():
    # UPP: UP is argmax (0.50). DNN: DOWN is argmax even though p_up=0.35.
    # SID: SIDE is argmax. Only UPP may be bought.
    panel = _panel({
        "UPP": (0.20, 0.30, 0.50),
        "DNN": (0.45, 0.20, 0.35),
        "SID": (0.25, 0.50, 0.25),
    })
    res = _engine(admission_mode="argmax").run(panel)
    assert _bought(res) == {"UPP"}


def test_argmax_ignores_the_absolute_floor_entirely():
    # p_up=0.40 is BELOW the 0.45 floor, so absolute_gate buys nothing, but UP
    # is still the argmax. This is the gate-starvation case the mode targets.
    panel = _panel({"LOW": (0.35, 0.25, 0.40)})

    gated = _engine(admission_mode="absolute_gate", admission_floor=0.45).run(panel)
    assert _bought(gated) == set()
    assert gated.zero_candidate_days > 0

    arg = _engine(admission_mode="argmax", admission_floor=0.45).run(panel)
    assert _bought(arg) == {"LOW"}


def test_argmax_admits_nothing_when_no_name_has_up_on_top():
    panel = _panel({
        "A": (0.50, 0.30, 0.20),
        "B": (0.40, 0.40, 0.20),
    })
    res = _engine(admission_mode="argmax").run(panel)
    assert _bought(res) == set()
    # Cash carry is recorded the same way absolute_gate records it.
    assert res.zero_candidate_days > 0


def test_argmax_ranks_by_p_up_within_the_admitted_set():
    # Four argmax-UP names, max_positions=2 → the two highest p_up win.
    panel = _panel({
        "P90": (0.03, 0.07, 0.90),
        "P80": (0.05, 0.15, 0.80),
        "P60": (0.15, 0.25, 0.60),
        "P40": (0.30, 0.30, 0.40),
    })
    res = _engine(admission_mode="argmax").run(panel)
    assert _bought(res) == {"P90", "P80"}


def test_argmax_tie_does_not_count_as_up():
    # A flat split must not read as a BUY call — the comparison is strict.
    panel = _panel({"TIE": (1 / 3, 1 / 3, 1 / 3)})
    res = _engine(admission_mode="argmax").run(panel)
    assert _bought(res) == set()


def test_argmax_respects_the_pool_cap_before_the_top_n_slice():
    panel = _panel({
        "A": (0.05, 0.15, 0.80),
        "B": (0.06, 0.16, 0.78),
        "C": (0.07, 0.17, 0.76),
        "D": (0.08, 0.18, 0.74),
    })
    res = _engine(admission_mode="argmax", admission_pool_cap=1).run(panel)
    # Cap=1 leaves a single survivor, so max_positions=2 cannot fill two names.
    assert _bought(res) == {"A"}


def test_one_dim_oracle_admits_nothing_rather_than_guessing():
    """A p_up-only oracle exposes no class structure.

    Inventing an argmax from a threshold would silently turn this mode into a
    second absolute gate, which is the very thing it replaces.
    """
    panel = _panel({"X": (0.10, 0.20, 0.70)})

    def _oracle1(X: np.ndarray) -> np.ndarray:
        return X[:, -1, 2].astype(np.float64)

    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["f_dn", "f_sd", "f_up"],
        rebalance_mode="tranche", tranche_hold_days=HOLD,
        max_positions=MAX_POS, signal_threshold=0.40,
        liquid_top_n=None, initial_capital=1_000_000_000.0,
        admission_mode="argmax",
    )
    res = WalkForwardEngine(cfg, _oracle1).run(panel)
    assert _bought(res) == set()


def test_default_mode_unchanged_by_the_argmax_plumbing():
    # The 3-tuple inference cache and _argmax_up_today bookkeeping must not
    # perturb the pre-existing cross_sectional path.
    panel = _panel({
        "A": (0.05, 0.15, 0.80),
        "B": (0.15, 0.25, 0.60),
        "C": (0.60, 0.20, 0.20),
    })
    res = _engine(admission_mode="cross_sectional").run(panel)
    # signal_threshold=0.40 → A and B clear it, C does not.
    assert _bought(res) == {"A", "B"}


def test_inference_cache_carries_argmax_across_runs():
    # run_backtest shares one cache across the whole threshold sweep; if the
    # argmax were dropped on a cache hit, admission would silently empty out.
    panel = _panel({"UPP": (0.20, 0.30, 0.50), "DNN": (0.45, 0.20, 0.35)})
    cache: dict = {}
    first = _engine(admission_mode="argmax").run(panel, inference_cache=cache)
    assert cache, "cache was not populated"
    second = _engine(admission_mode="argmax").run(panel, inference_cache=cache)
    assert _bought(first) == _bought(second) == {"UPP"}
