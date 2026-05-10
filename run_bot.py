"""Entrypoint for the Quant V6 Telegram bot service.

Thin wrapper so process supervisors (systemd / pm2 / supervisord) can invoke
the long-running interactive bot via a single command at the project root:

    python run_bot.py

For local development you can equivalently run:

    python -m src.utils.telegram_bot

Both call the same `main()` and produce identical behaviour.
"""

from src.utils.telegram_bot import main

if __name__ == "__main__":
    main()
