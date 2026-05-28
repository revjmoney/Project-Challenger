"""
Webhook notifications — Discord and Telegram.

Usage:
    from notifications import notify
    notify("Bot stopped.", title="E-STOP", level="danger")

Levels: "info" | "success" | "danger" | "warning"
Webhooks are read from CONFIG["NOTIFICATIONS"] at call time so settings
changes take effect without a restart.
"""
import json as _json
import threading
import urllib.error
import urllib.request
from config import CONFIG

_COLORS = {
    "info":    0x5865F2,
    "success": 0x57F287,
    "danger":  0xED4245,
    "warning": 0xFEE75C,
}


def _post(url: str, payload: dict) -> None:
    data = _json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except urllib.error.HTTPError as e:
        print(f"[NOTIFY] HTTP {e.code}: {url}")
    except Exception as e:
        print(f"[NOTIFY] {e}")


def notify(message: str, title: str = "", level: str = "info") -> None:
    """Send message to all configured webhooks (fire-and-forget in a daemon thread)."""
    cfg = CONFIG.get("NOTIFICATIONS", {})
    discord_url = cfg.get("DISCORD_WEBHOOK", "").strip()
    tg_token    = cfg.get("TELEGRAM_TOKEN",  "").strip()
    tg_chat     = cfg.get("TELEGRAM_CHAT_ID", "").strip()

    if not discord_url and not (tg_token and tg_chat):
        return

    color = _COLORS.get(level, _COLORS["info"])

    def _fire():
        if discord_url:
            embed = {
                "title":       title or "Project Challenger",
                "description": message,
                "color":       color,
            }
            _post(discord_url, {"embeds": [embed]})

        if tg_token and tg_chat:
            text = f"*{title}*\n{message}" if title else message
            _post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                {"chat_id": tg_chat, "text": text, "parse_mode": "Markdown"},
            )

    threading.Thread(target=_fire, daemon=True, name="notify").start()
