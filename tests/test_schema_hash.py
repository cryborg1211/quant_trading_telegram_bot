"""Unit tests for the automated feature-schema hashing system (Phase 2).

Tests 1-6 exercise the pure-Python hash utility (no ML stack).
Tests 7-10 verify the integration with the pipeline and MR modules.
"""
from __future__ import annotations

from src.utils.schema_hash import compute_feature_schema_hash


# ── Pure utility tests ───────────────────────────────────────────────────────


def test_hash_format():
    result = compute_feature_schema_hash([("close_fd_xsz", "Float32")], 0.4)
    assert result.startswith("v2-sha8:")
    assert len(result) == len("v2-sha8:") + 8


def test_hash_deterministic():
    schema = [("a", "Float32"), ("b", "Int8")]
    h1 = compute_feature_schema_hash(schema, 0.4)
    h2 = compute_feature_schema_hash(schema, 0.4)
    assert h1 == h2


def test_hash_column_order_matters():
    h1 = compute_feature_schema_hash([("a", "Float32"), ("b", "Float32")], 0.4)
    h2 = compute_feature_schema_hash([("b", "Float32"), ("a", "Float32")], 0.4)
    assert h1 != h2


def test_hash_dtype_matters():
    h1 = compute_feature_schema_hash([("col", "Float32")], 0.4)
    h2 = compute_feature_schema_hash([("col", "Float64")], 0.4)
    assert h1 != h2


def test_hash_frac_diff_matters():
    schema = [("col", "Float32")]
    h1 = compute_feature_schema_hash(schema, 0.4)
    h2 = compute_feature_schema_hash(schema, 0.5)
    assert h1 != h2


def test_hash_none_frac_diff():
    result = compute_feature_schema_hash([("col", "Float32")], None)
    assert result.startswith("v2-sha8:")


# ── Integration tests (pipeline / MR constants) ─────────────────────────────


def test_feature_recipe_version_format():
    from src.backtest.pipeline import FEATURE_RECIPE_VERSION

    assert FEATURE_RECIPE_VERSION.startswith("v2-sha8:")
    assert len(FEATURE_RECIPE_VERSION) == 16


def test_mr_schema_hash_format():
    from src.features.mr_features import MR_SCHEMA_HASH

    assert MR_SCHEMA_HASH.startswith("v2-sha8:")
    assert len(MR_SCHEMA_HASH) == 16


def test_feature_schema_names_match_pipeline():
    from src.backtest.pipeline import CATEGORICAL_FEATURES, FEATURE_SCHEMA

    schema_names = [name for name, _ in FEATURE_SCHEMA]
    for cat in CATEGORICAL_FEATURES:
        assert cat in schema_names, f"{cat} missing from FEATURE_SCHEMA"
    assert len(FEATURE_SCHEMA) == 15


def test_hash_different_from_old_manual_version():
    from src.backtest.pipeline import FEATURE_RECIPE_VERSION

    assert FEATURE_RECIPE_VERSION != "v1.1"
