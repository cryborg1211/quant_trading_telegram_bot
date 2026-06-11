"""Phase-2 serve-path tests: artifact `strategy` dict → dispatch → Telegram card.

Backward compatibility is the contract: artifacts without a strategy dict
(or with mode != "tranche") must leave the legacy half-Kelly path untouched.
"""
from __future__ import annotations

import joblib
import pytest

from main import _tranche_signal_fields
from src.bot.bot_inference import V3BotInference
from src.utils.telegram_alerter import TelegramBot

_BASE_BUNDLE = {
    "ensemble": {"stub": True},        # never invoked by these tests
    "tabular_features": ["f1", "f2"],
    "up_threshold": 0.45,
    "signal_threshold": 0.40,
}


class TestArtifactStrategyParsing:
    def test_strategy_dict_round_trips(self, tmp_path) -> None:
        bundle = dict(_BASE_BUNDLE)
        bundle["strategy"] = {"mode": "tranche", "hold_days": 30,
                              "signal_threshold": 0.40,
                              "pt_sigma": 3.0, "sl_sigma": 2.0}
        p = tmp_path / "artifact.joblib"
        joblib.dump(bundle, p)
        bot = V3BotInference.from_artifact(p)
        assert bot.strategy["mode"] == "tranche"
        assert bot.strategy["hold_days"] == 30
        assert bot.strategy["pt_sigma"] == 3.0

    def test_legacy_artifact_defaults_empty(self, tmp_path) -> None:
        p = tmp_path / "legacy.joblib"
        joblib.dump(dict(_BASE_BUNDLE), p)
        bot = V3BotInference.from_artifact(p)
        assert bot.strategy == {}


class TestTrancheSignalFields:
    def test_tranche_weight_is_cohort_split(self) -> None:
        fields = _tranche_signal_fields(
            {"mode": "tranche", "hold_days": 30}, n_picks=3)
        assert fields["suggested_weight"] == pytest.approx(1.0 / 90)
        assert "30 phiên" in fields["hold_label"]

    def test_barrier_rule_rendered(self) -> None:
        fields = _tranche_signal_fields(
            {"mode": "tranche", "hold_days": 30,
             "pt_sigma": 3.0, "sl_sigma": 2.0}, n_picks=5)
        assert "+3.0σ" in fields["exit_rule"]
        assert "2.0σ" in fields["exit_rule"]

    def test_legacy_strategy_is_noop(self) -> None:
        assert _tranche_signal_fields(None, 3) == {}
        assert _tranche_signal_fields({}, 3) == {}
        assert _tranche_signal_fields({"mode": "grid"}, 3) == {}

    def test_zero_picks_guard(self) -> None:
        fields = _tranche_signal_fields({"mode": "tranche", "hold_days": 20}, 0)
        assert fields["suggested_weight"] == pytest.approx(1.0 / 20)


class TestCardRendering:
    _SIGNAL = {
        "ticker": "HPG", "price": "27,500 VND", "horizon_label": "T+20",
        "suggested_weight": 0.0111, "prob_up": 55.0, "prob_side": 5.0,
        "prob_down": 40.0, "conclusion": "ok", "article_urls": [],
    }

    def test_hold_line_rendered_when_present(self) -> None:
        data = dict(self._SIGNAL)
        data["hold_label"] = "30 phiên (đến ~24/07/2026)"
        data["exit_rule"] = "chốt lời sớm tại +3.0σ / cắt lỗ tại −2.0σ"
        msg = TelegramBot._build_message(data)
        assert "Nắm giữ: <b>30 phiên" in msg
        assert "Quy tắc thoát:" in msg
        assert "1.1% NAV" in msg

    def test_legacy_card_has_no_hold_line(self) -> None:
        msg = TelegramBot._build_message(dict(self._SIGNAL))
        assert "Nắm giữ" not in msg
