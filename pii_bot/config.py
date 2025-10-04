"""Configuration helpers for the PII bot."""

from __future__ import annotations

import os
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
UPDATE_STATE_PATH = PROJECT_ROOT / ".last_update_state.json"


def _load_env_file() -> None:
    """Load environment variables from a local ``.env`` file if it exists."""

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    try:
        content = env_path.read_text(encoding="utf-8")
    except OSError:
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if _is_quoted(value):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def _is_quoted(value: str) -> bool:
    return (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    )


_load_env_file()

DB_PATH = os.getenv("DB_PATH", "planets.db")
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO")

TOKEN_ENV_NAME = os.getenv("PII_BOT_TOKEN_ENV", "DISCORD_TOKEN")
DISCORD_TOKEN = os.getenv(TOKEN_ENV_NAME)
BOT_SERVICE_NAME = os.getenv("BOT_SERVICE_NAME")

if not DISCORD_TOKEN:
    raise RuntimeError(
        "Не удалось найти токен Discord. Установите переменную окружения "
        f"{TOKEN_ENV_NAME!r} или укажите корректное имя переменной в "
        "PII_BOT_TOKEN_ENV."
    )

DEFAULT_SLOTS = int(os.getenv("DEFAULT_SLOTS", "10"))
DEFAULT_DRILLS = int(os.getenv("DEFAULT_DRILLS", "22"))
DEFAULT_HOURS = int(os.getenv("DEFAULT_HOURS", "168"))
DEFAULT_BETA = float(os.getenv("DEFAULT_BETA", "0.85"))
MANDATORY_PLANET_COUNT = int(os.getenv("MANDATORY_PLANET_COUNT", "6"))

RES_REMINDER_DELAY_HOURS = int(os.getenv("RES_REMINDER_DELAY_HOURS", "24"))
RES_REMINDER_CHECK_SECONDS = int(
    os.getenv("RES_REMINDER_CHECK_SECONDS", str(60 * 60))
)

ALLOW_EPHEMERAL_RESPONSES = os.getenv("ALLOW_EPHEMERAL_RESPONSES", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

__all__ = [
    "ALLOW_EPHEMERAL_RESPONSES",
    "BOT_SERVICE_NAME",
    "DB_PATH",
    "DEFAULT_BETA",
    "DEFAULT_DRILLS",
    "DEFAULT_HOURS",
    "DEFAULT_SLOTS",
    "DISCORD_TOKEN",
    "LOG_DIR",
    "LOG_LEVEL_NAME",
    "MANDATORY_PLANET_COUNT",
    "PROJECT_ROOT",
    "RES_REMINDER_CHECK_SECONDS",
    "RES_REMINDER_DELAY_HOURS",
    "TOKEN_ENV_NAME",
    "UPDATE_STATE_PATH",
]
