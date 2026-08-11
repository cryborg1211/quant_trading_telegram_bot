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
            admission_pool_cap: int = 6, rank_mode: str = "p_up",
            rank_seed: int = 0, max_positions: int = MAX_POS) -> WalkForwardEngine:
    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["f_dn", "f_sd", "f_up"],
        rebalance_mode="tranche", tranche_hold_days=HOLD,
        max_positions=max_positions, signal_threshold=0.40,
        liquid_top_n=None, initial_capital=1_000_000_000.0,
        admission_mode=admission_mode, admission_floor=admission_floor,
        admission_pool_cap=admission_pool_cap,
        rank_mode=rank_mode, rank_seed=rank_seed,
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


# ── absolute_gate_argmax: what SERVE actually requires ─────────────────────

def test_gate_argmax_requires_both_the_floor_and_the_class():
    """Serve needs BOTH: `_select_candidates` gates on tau AND dispatches only
    `final_decision == 2`, which `make_final_decision` only returns on an UP
    argmax. The backtest validated the floor ALONE, so it admits more."""
    panel = _panel({
        "BOTH": (0.20, 0.20, 0.60),   # clears 0.45 AND argmax UP  -> admitted
        "GATE": (0.50, 0.02, 0.48),   # clears 0.45 but argmax DOWN -> rejected
        "ARGM": (0.20, 0.40, 0.40),   # argmax... SIDE, below floor -> rejected
    })
    res = _engine(admission_mode="absolute_gate_argmax", admission_floor=0.45).run(panel)
    assert _bought(res) == {"BOTH"}


def test_gate_argmax_is_stricter_than_the_gate_alone():
    # The one name clears tau but its argmax is DOWN. absolute_gate buys it;
    # the serve-equivalent mode does not. This gap IS the divergence measured
    # by scripts/analyze_serve_stack_ab.py.
    panel = _panel({"X": (0.50, 0.02, 0.48)})
    assert _bought(_engine(admission_mode="absolute_gate",
                           admission_floor=0.45).run(panel)) == {"X"}
    assert _bought(_engine(admission_mode="absolute_gate_argmax",
                           admission_floor=0.45).run(panel)) == set()


def test_gate_argmax_with_one_dim_oracle_admits_nothing():
    panel = _panel({"X": (0.10, 0.20, 0.70)})

    def _oracle1(X: np.ndarray) -> np.ndarray:
        return X[:, -1, 2].astype(np.float64)

    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["f_dn", "f_sd", "f_up"],
        rebalance_mode="tranche", tranche_hold_days=HOLD,
        max_positions=MAX_POS, signal_threshold=0.40,
        liquid_top_n=None, initial_capital=1_000_000_000.0,
        admission_mode="absolute_gate_argmax", admission_floor=0.45,
    )
    assert _bought(WalkForwardEngine(cfg, _oracle1).run(panel)) == set()


# ── rank_mode: the backtestable proxy for serve's sentiment-first ordering ──

def test_default_rank_mode_picks_the_highest_p_up():
    panel = _panel({
        "HI": (0.05, 0.15, 0.80),
        "MD": (0.15, 0.25, 0.60),
        "LO": (0.25, 0.28, 0.47),
    })
    res = _engine(admission_mode="absolute_gate", admission_floor=0.45).run(panel)
    assert _bought(res) == {"HI", "MD"}      # max_positions=2


def test_random_rank_mode_is_deterministic_for_a_seed():
    # Reproducibility matters: an A/B arm that shuffles differently every run
    # cannot be compared against anything.
    panel = _panel({
        "A": (0.05, 0.15, 0.80), "B": (0.10, 0.20, 0.70),
        "C": (0.15, 0.25, 0.60), "D": (0.20, 0.28, 0.52),
    })
    first = _bought(_engine(admission_mode="absolute_gate", admission_floor=0.45,
                            rank_mode="random", rank_seed=7).run(panel))
    again = _bought(_engine(admission_mode="absolute_gate", admission_floor=0.45,
                            rank_mode="random", rank_seed=7).run(panel))
    assert first == again


def test_random_rank_mode_can_pick_something_other_than_the_top_p_up():
    # If the shuffle never changed the selection the proxy would be vacuous.
    panel = _panel({
        "A": (0.05, 0.15, 0.80), "B": (0.10, 0.20, 0.70),
        "C": (0.15, 0.25, 0.60), "D": (0.20, 0.28, 0.52),
        "E": (0.22, 0.26, 0.52), "F": (0.24, 0.24, 0.52),
    })
    by_p_up = _bought(_engine(admission_mode="absolute_gate", admission_floor=0.45,
                              rank_mode="p_up").run(panel))
    shuffled = {
        frozenset(_bought(_engine(admission_mode="absolute_gate",
                                  admission_floor=0.45, rank_mode="random",
                                  rank_seed=s).run(panel)))
        for s in range(6)
    }
    assert any(s != frozenset(by_p_up) for s in shuffled)


def test_random_rank_mode_still_respects_the_floor():
    # Shuffling must reorder the ADMITTED set only — never smuggle in a name
    # that failed the gate.
    panel = _panel({"OK": (0.20, 0.20, 0.60), "NO": (0.60, 0.20, 0.20)})
    for s in range(4):
        assert "NO" not in _bought(
            _engine(admission_mode="absolute_gate", admission_floor=0.45,
                    rank_mode="random", rank_seed=s).run(panel))


def test_max_positions_three_takes_fewer_names_than_five():
    panel = _panel({
        "A": (0.05, 0.15, 0.80), "B": (0.10, 0.20, 0.70),
        "C": (0.15, 0.25, 0.60), "D": (0.20, 0.28, 0.52),
        "E": (0.22, 0.26, 0.52),
    })
    three = _bought(_engine(admission_mode="absolute_gate", admission_floor=0.45,
                            max_positions=3).run(panel))
    five = _bought(_engine(admission_mode="absolute_gate", admission_floor=0.45,
                           max_positions=5).run(panel))
    assert len(three) == 3 and len(five) == 5
    assert three < five


def test_inference_cache_carries_argmax_across_runs():
    # run_backtest shares one cache across the whole threshold sweep; if the
    # argmax were dropped on a cache hit, admission would silently empty out.
    panel = _panel({"UPP": (0.20, 0.30, 0.50), "DNN": (0.45, 0.20, 0.35)})
    cache: dict = {}
    first = _engine(admission_mode="argmax").run(panel, inference_cache=cache)
    assert cache, "cache was not populated"
    second = _engine(admission_mode="argmax").run(panel, inference_cache=cache)
    assert _bought(first) == _bought(second) == {"UPP"}
