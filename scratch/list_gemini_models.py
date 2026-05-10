"""
List Gemini models available to the configured API key.

Usage:
    python scratch/list_gemini_models.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install google-generativeai python-dotenv") from exc


def main() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")

    genai.configure(api_key=api_key)

    print("Available Gemini models")
    print("=" * 80)

    generate_content_models: list[str] = []

    for model in genai.list_models():
        methods = list(getattr(model, "supported_generation_methods", []) or [])
        name = getattr(model, "name", "")
        display_name = getattr(model, "display_name", "")
        version = getattr(model, "version", "")
        supports_generate_content = "generateContent" in methods

        if supports_generate_content:
            generate_content_models.append(name)

        marker = "✅ generateContent" if supports_generate_content else "  "
        print(f"{marker} | {name} | display={display_name} | version={version} | methods={methods}")

    print("\nModels supporting generateContent")
    print("=" * 80)
    for name in generate_content_models:
        print(name)


if __name__ == "__main__":
    main()