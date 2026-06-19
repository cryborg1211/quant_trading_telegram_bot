"""Settings tab — REAL .env + config/settings.json writer.

This is the ONLY tab with real (non-stub) logic in P1. It reads the current
``.env`` and ``config/settings.json`` on render, displays masked secret fields,
and — ONLY when the Save button is explicitly clicked — persists the values to
disk.

Safety guarantees (per plan + program safety constraints):
  - No write happens at import or on plain render. Writes run ONLY inside the
    ``if submitted:`` branch of the form.
  - ``.env`` is updated via python-dotenv ``set_key()`` (preserves unrelated
    lines) and is backed up to ``.env.dashboard.bak`` before the first write.
  - ``config/settings.json`` uses read-parse-merge-write so unrelated keys
    (paths, model, training, crawler, sentiment, universe_filter, and other
    trading keys) are never clobbered.
  - Secrets are never logged or echoed back in plaintext.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values, set_key

# Repo root = two levels up from this file (dashboard/tabs/settings.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO_ROOT / ".env"
_ENV_BACKUP_PATH = _REPO_ROOT / ".env.dashboard.bak"
_SETTINGS_JSON_PATH = _REPO_ROOT / "config" / "settings.json"

# Masked placeholder shown when a secret already exists; leaving the field at
# this value on Save means "keep the existing secret unchanged".
_MASK = "********"


def _read_env() -> dict[str, str]:
    """Read current .env values (empty dict if the file does not exist)."""
    if not _ENV_PATH.exists():
        return {}
    # dotenv_values returns Optional[str] values; coerce None -> "".
    return {k: (v or "") for k, v in dotenv_values(_ENV_PATH).items()}


def _read_settings_json() -> dict:
    """Read config/settings.json, returning {} if missing or unparseable."""
    if not _SETTINGS_JSON_PATH.exists():
        return {}
    try:
        return json.loads(_SETTINGS_JSON_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_env(updates: dict[str, str]) -> None:
    """Persist updates to .env via set_key (preserves unrelated lines).

    Backs up the existing .env to .env.dashboard.bak before the first write so
    a user-edited file is never silently lost. Empty-string values are skipped
    (treated as "no change") so we do not blank out an existing secret.
    """
    # Ensure the file exists so set_key has a target.
    if _ENV_PATH.exists():
        shutil.copy2(_ENV_PATH, _ENV_BACKUP_PATH)
    else:
        _ENV_PATH.touch()

    for key, value in updates.items():
        if value == "":
            # Skip blanks — never overwrite an existing secret with empty.
            continue
        set_key(str(_ENV_PATH), key, value)


def _write_settings_json(horizon_default: int, sentiment_threshold: float) -> None:
    """Merge horizon + sentiment threshold into config/settings.json.

    Read-parse-merge-write: only the targeted keys are touched; everything
    else in the file is preserved verbatim.
    """
    data = _read_settings_json()
    trading = data.setdefault("trading", {})
    trading["sentiment_entry_threshold"] = round(float(sentiment_threshold), 2)
    # Dashboard-local preference (not consumed by the engine yet; P2 reads it).
    trading["dashboard_horizon_default"] = int(horizon_default)

    _SETTINGS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_JSON_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render() -> None:
    """Render the Settings tab (real read + on-click write)."""
    st.header("Settings — Cài đặt")
    st.caption("Lưu khóa Gemini, Telegram và tùy chọn mô hình vào .env + settings.json.")

    env = _read_env()
    settings = _read_settings_json()
    trading = settings.get("trading", {}) if isinstance(settings, dict) else {}

    has_gemini = bool(env.get("GEMINI_API_KEY"))
    has_token = bool(env.get("TELEGRAM_BOT_TOKEN"))
    existing_threshold = float(trading.get("sentiment_entry_threshold", 0.7))
    existing_horizon = int(trading.get("dashboard_horizon_default", 20))

    with st.form("settings_form"):
        gemini_key = st.text_input(
            "GEMINI_API_KEY",
            value=_MASK if has_gemini else "",
            type="password",
            help="Để nguyên dấu *** nếu không muốn đổi khóa hiện tại.",
        )
        telegram_token = st.text_input(
            "TELEGRAM_BOT_TOKEN",
            value=_MASK if has_token else "",
            type="password",
            help="Để nguyên dấu *** nếu không muốn đổi token hiện tại.",
        )
        telegram_chat_id = st.text_input(
            "TELEGRAM_CHAT_ID",
            value=env.get("TELEGRAM_CHAT_ID", ""),
        )
        horizon_default = st.selectbox(
            "Khung thời gian mặc định",
            options=[5, 20],
            index=(1 if existing_horizon == 20 else 0),
            format_func=lambda h: f"T+{h}",
        )
        sentiment_threshold = st.slider(
            "Ngưỡng sentiment",
            min_value=0.5,
            max_value=0.95,
            value=min(max(existing_threshold, 0.5), 0.95),
            step=0.05,
        )
        submitted = st.form_submit_button("💾 Lưu cài đặt")

    # ---- WRITES happen ONLY here, on explicit Save click --------------------
    if submitted:
        try:
            env_updates: dict[str, str] = {}
            # Only stage secrets the user actually changed (mask left intact =
            # keep existing). Empty + no existing = nothing to write.
            if gemini_key and gemini_key != _MASK:
                env_updates["GEMINI_API_KEY"] = gemini_key
            if telegram_token and telegram_token != _MASK:
                env_updates["TELEGRAM_BOT_TOKEN"] = telegram_token
            # chat_id is not secret; always persist the current field value.
            env_updates["TELEGRAM_CHAT_ID"] = telegram_chat_id.strip()

            _write_env(env_updates)
            _write_settings_json(horizon_default, sentiment_threshold)
            st.success("Đã lưu cài đặt.")
            st.caption(
                "Lưu ý: thay đổi settings.json cần khởi động lại dashboard để áp dụng "
                "(CONFIG được dựng một lần lúc import)."
            )
        except Exception as exc:  # noqa: BLE001 - surface any write failure to the UI
            st.error(f"Lưu cài đặt thất bại: {exc}")
