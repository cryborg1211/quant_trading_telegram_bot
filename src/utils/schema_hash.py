"""Deterministic feature-schema hashing for train/serve parity enforcement.

Hashes the *structural contract* of a feature pipeline — column names (ordered),
their Polars/numpy dtype strings, and the ``frac_diff_d`` hyperparameter — into
a short, version-prefixed hex digest.  Any change to the schema (added/removed
column, reordered column, dtype change, frac-diff tuning) produces a different
hash, forcing a retrain before the serve path accepts the artifact.

The hash is computed at import time and stamped into model artifacts.  The serve
path compares the stamped hash against the live constant; a mismatch is a hard
error (artifact was trained on a different feature contract).

No Polars or project imports — this module is pure stdlib so tests can exercise
it without loading the ML stack.
"""
from __future__ import annotations

import hashlib

__all__ = ["compute_feature_schema_hash"]


def compute_feature_schema_hash(
    schema: list[tuple[str, str]],
    frac_diff_d: float | None,
) -> str:
    """Return a deterministic ``v2-sha8:<hex8>`` hash for a feature schema.

    Parameters
    ----------
    schema
        Ordered ``(column_name, dtype_string)`` pairs that define the feature
        pool contract.  Order is load-bearing: a reorder produces a new hash.
    frac_diff_d
        Fractional-differentiation *d* parameter.  Included because it changes
        the numerical pipeline output without changing column names.  Pass
        ``None`` for pipelines that do not use frac-diff (e.g. MR).

    Returns
    -------
    str
        ``"v2-sha8:"`` followed by the first 8 hex characters of the SHA-256
        digest of the canonical serialization.
    """
    parts = [f"{name}:{dtype}" for name, dtype in schema]
    parts.append(f"frac_diff_d:{frac_diff_d}")
    serialized = "|".join(parts)
    hex8 = hashlib.sha256(serialized.encode()).hexdigest()[:8]
    return f"v2-sha8:{hex8}"
