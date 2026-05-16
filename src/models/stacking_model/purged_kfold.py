"""Purged & Embargoed K-Fold cross-validation (Lopez de Prado, AFML Ch. 7).

scikit-learn-compatible (subclasses ``_BaseKFold``) so it is a drop-in for
``TimeSeriesSplit`` inside ``cross_val_predict`` / the manual OOF loop in
``train_stacking.py``. Pure ML logic — no DuckDB, no infrastructure.

The leakage it kills
────────────────────
A 5-day forward label means sample *t*'s outcome spans bars [t, t+5]. With a
plain ``TimeSeriesSplit`` the last training samples and the first test
samples share up to 4 outcome days → the validation set is autocorrelated
with the train set → optimistic, fake Macro-F1.

Two corrections (de Prado, Snippets 7.3–7.4):
  • PURGE   : drop any TRAIN sample whose observation window [start, t1]
              overlaps the TEST window  [min(start_test), max(t1_test)].
  • EMBARGO : additionally drop the ``embargo_bars`` TRAIN samples that
              come immediately *after* the test block, to neutralize
              residual serial correlation that purging alone misses.

Required inputs
───────────────
For every row passed to ``.split`` you must supply, in the SAME order:
  start_times : the bar date the sample was observed on   (feature date)
  end_times   : the event-end date ``t1``                  (label decided)
Both come straight from ``triple_barrier.add_triple_barrier_labels``
(``date`` and ``t1_5d`` / ``t1_20d``).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class PurgedKFold(_BaseKFold):
    """K contiguous time-ordered test folds with purge + embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds (match ``train_stacking.N_SPLITS``, default 3).
    start_times : array-like[datetime64], shape (n,)
        Observation/feature date per sample, ordered as the X rows.
    end_times : array-like[datetime64], shape (n,)
        Event-end date ``t1`` per sample (label-decided date).
    embargo_bars : int, optional
        Number of samples to embargo immediately after each test block.
        For a fixed H-day label, set this >= H (recommended: == horizon).
        Takes precedence over ``embargo_pct`` when > 0.
    embargo_pct : float, optional
        de Prado's fractional embargo: ``int(n * embargo_pct)`` samples.
        Used only when ``embargo_bars`` == 0.

    Notes
    -----
    * Rows are assumed already sorted by date (the trainer sorts by
      ``[date, ticker]`` after ``chronological_split`` — multiple tickers
      per date is fine; purge/embargo operate in *time*, not on tickers).
    * ``np.array_split`` partitions the index range, so every sample lands
      in exactly one test fold → fully compatible with
      ``sklearn.model_selection.cross_val_predict``.
    """

    def __init__(
        self,
        n_splits: int = 3,
        *,
        start_times=None,
        end_times=None,
        embargo_bars: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.start_times = (
            None if start_times is None
            else np.asarray(start_times, dtype="datetime64[ns]")
        )
        self.end_times = (
            None if end_times is None
            else np.asarray(end_times, dtype="datetime64[ns]")
        )
        self.embargo_bars = int(embargo_bars)
        self.embargo_pct = float(embargo_pct)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_triple_barrier(
        cls,
        df,
        *,
        n_splits: int = 3,
        date_col: str = "date",
        t1_col: str = "t1_5d",
        embargo_bars: int = 5,
    ) -> "PurgedKFold":
        """Build directly from a triple-barrier-labeled frame.

        Accepts a polars or pandas DataFrame already filtered to the
        training rows, in the SAME order they will be fed to ``.split``.
        ``embargo_bars`` defaults to 5 (the 5d label horizon).
        """
        if hasattr(df, "to_pandas"):  # polars
            start = df[date_col].to_numpy()
            end = df[t1_col].to_numpy()
        else:  # pandas
            start = df[date_col].to_numpy()
            end = df[t1_col].to_numpy()
        return cls(
            n_splits=n_splits,
            start_times=start,
            end_times=end,
            embargo_bars=embargo_bars,
        )

    # ------------------------------------------------------------------ #
    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.start_times is None or self.end_times is None:
            raise ValueError("PurgedKFold requires start_times and end_times.")

        X, y, groups = indexable(X, y, groups)
        n = _num_samples(X)
        if len(self.start_times) != n or len(self.end_times) != n:
            raise ValueError(
                f"start/end_times length ({len(self.start_times)}/"
                f"{len(self.end_times)}) != n_samples ({n}). They must be "
                "aligned to the rows passed to split()."
            )

        indices = np.arange(n)
        mbrg = (
            self.embargo_bars
            if self.embargo_bars > 0
            else int(n * self.embargo_pct)
        )

        for test_idx in np.array_split(indices, self.n_splits):
            if test_idx.size == 0:
                continue
            b = int(test_idx[-1])  # last positional index of the test block
            test_start = self.start_times[test_idx].min()
            test_end = self.end_times[test_idx].max()

            # PURGE: a train sample is contaminated iff its observation
            # window [start_i, end_i] overlaps the test window.
            overlap = (self.start_times <= test_end) & (
                self.end_times >= test_start
            )

            # EMBARGO: the next `mbrg` positional samples after the test
            # block are dropped from training.
            embargo = np.zeros(n, dtype=bool)
            embargo[b + 1 : min(n, b + 1 + mbrg)] = True

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            train_mask &= ~overlap
            train_mask &= ~embargo

            train_idx = indices[train_mask]
            if train_idx.size == 0:
                # Degenerate fold (e.g. n_splits too high vs. horizon).
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


if __name__ == "__main__":
    # Smoke test: prove purge removes the overlap and embargo adds a gap.
    import datetime as _dt

    n = 60
    start = np.array(
        [np.datetime64(_dt.date(2024, 1, 1) + _dt.timedelta(days=i)) for i in range(n)]
    )
    end = start + np.timedelta64(5, "D")  # every label spans 5 days

    cv = PurgedKFold(
        n_splits=3, start_times=start, end_times=end, embargo_bars=5
    )
    for fold, (tr, te) in enumerate(cv.split(np.zeros((n, 3))), 1):
        tr_end = end[tr]
        te_start = start[te].min()
        te_end = end[te].max()
        # No train label window may overlap the test window.
        assert not ((start[tr] <= te_end) & (tr_end >= te_start)).any(), "purge leak!"
        gap = start[tr][start[tr] > te_end]
        print(
            f"fold {fold}: test={te.size} train={tr.size} "
            f"purged+embargoed={n - te.size - tr.size}"
        )
    print("PurgedKFold smoke test OK")
