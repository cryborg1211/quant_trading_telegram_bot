"""Standalone smoke test for Gemini sentiment JSON pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.models.quant_agent_arbitrator import get_batch_sentiment_scores  # noqa: E402


def main() -> None:
    print("=" * 80)
    print("APG SENTIMENT PIPELINE SMOKE TEST")
    print("=" * 80)
    print(f"GEMINI_API_KEY present: {bool(os.environ.get('GEMINI_API_KEY'))}")
    print(f"GEMINI_MODEL: {os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}")
    print("-" * 80)

    fake_payload = {
        "APG": [
            "Source URL: https://cafef.vn/fake-apg-test.htm\n"
            "Title: APG công bố kế hoạch kinh doanh mới\n"
            "Full Article Body:\n"
            "APG ghi nhận kết quả kinh doanh tích cực trong quý gần nhất. "
            "Doanh thu tăng trưởng nhờ hoạt động môi giới và tự doanh cải thiện. "
            "Ban lãnh đạo cho biết công ty sẽ kiểm soát rủi ro margin và tối ưu chi phí vốn. "
            "Tuy nhiên, biến động thị trường chứng khoán vẫn là rủi ro chính có thể ảnh hưởng lợi nhuận.\n"
            "---\n"
        ]
    }

    result = get_batch_sentiment_scores(fake_payload)

    print("-" * 80)
    print("FINAL RESULT:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()