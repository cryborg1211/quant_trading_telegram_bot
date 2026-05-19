#!/usr/bin/env python3
"""Background rolling-retrain — triggered by an ADMIN /audit_* command.

BOUNDED BY DESIGN
─────────────────
This re-fits ONLY the lightweight Mean-Reversion sub-model
(`python -m src.models.train_mr_lgbm`, ~30-60s on the full 850k rows) —
a genuine rolling re-fit on the latest data that matches the
"hoàn thành sau 1-2 phút" promise shown to the Admin.

The heavy 3-model 5d Stacking retrain is INTENTIONALLY NOT triggered per
audit: it is ~10+ minutes and would blow the stated time budget. Run that
separately via `main.py --task build_alpha360 && train_stacking`.

Writes a plain-Vietnamese one-liner to
``models/mr/last_retrain_summary.txt`` which the bot reads back to notify
the Admin on completion. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MR_DIR = ROOT / "models" / "mr"
SUMMARY_PATH = MR_DIR / "last_retrain_summary.txt"


def _write(msg: str) -> None:
    # The UTF-8 file is the SOURCE OF TRUTH the bot reads back. Console
    # echo is best-effort: a Windows cp1252 stdout cannot encode VN/emoji
    # and must NEVER raise (a print crash here previously clobbered the
    # good summary via the caller's except-fallback).
    MR_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(msg, encoding="utf-8")
    try:
        print(msg, flush=True)
    except Exception:  # noqa: BLE001
        try:
            sys.stdout.buffer.write((msg + "\n").encode("utf-8", "replace"))
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    t0 = time.perf_counter()
    cmd = [sys.executable, "-m", "src.models.train_mr_lgbm"]
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        _write("❌ Học lại MR quá thời gian (>10 phút) — đã hủy.")
        return 1
    dur = time.perf_counter() - t0

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-300:]
        _write(
            f"❌ Học lại MR thất bại sau {dur:.0f}s (mã lỗi "
            f"{proc.returncode}). Chi tiết: {tail.strip()[:200]}"
        )
        return proc.returncode

    # Read the freshly written MR report for a plain-VN summary.
    try:
        rep = json.loads(
            (MR_DIR / "mr_report.json").read_text(encoding="utf-8")
        )
        hold = rep.get("holdout", {})
        sel = rep.get("oof_selection", {})
        _write(
            f"✅ Học lại xong sau {dur:.0f}s. "
            f"Tỷ lệ tín hiệu hiếm (train): "
            f"{rep.get('train_pos_rate', 0.0) * 100:.2f}% | "
            f"Độ chính xác kiểm định 1 năm: "
            f"{hold.get('precision', 0.0) * 100:.0f}% trên "
            f"{hold.get('fires', 0)} cảnh báo | "
            f"Ngưỡng kích hoạt: {sel.get('tau', '?')}."
        )
    except Exception as exc:  # noqa: BLE001
        _write(
            f"✅ Học lại xong sau {dur:.0f}s "
            f"(không đọc được tóm tắt chi tiết: {exc})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
