# bot_pos.py
import os
import sys
import json
import logging
import sqlite3
import pathlib
import asyncio
import shutil
import re
import base64
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
UPDATE_STATE_PATH = PROJECT_ROOT / ".last_update_state.json"


def _load_env_file() -> None:
    """Загружает переменные окружения из .env, если такой файл существует."""

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
        if (value.startswith("\"") and value.endswith("\"")) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


_load_env_file()

import discord
from discord import app_commands
from discord.abc import Snowflake

# ==================== НАСТРОЙКИ (без config.json) ====================
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
DEFAULT_HOURS = int(os.getenv("DEFAULT_HOURS", "168"))  # горизонт для покрытия в расчётах
DEFAULT_BETA = float(os.getenv("DEFAULT_BETA", "0.85"))

RES_REMINDER_DELAY_HOURS = int(os.getenv("RES_REMINDER_DELAY_HOURS", "24"))
RES_REMINDER_CHECK_SECONDS = int(os.getenv("RES_REMINDER_CHECK_SECONDS", str(60 * 60)))

reminder_task: Optional[asyncio.Task] = None

# ==================== ЛОГИ ====================
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME.upper(), logging.INFO)
log_dir = pathlib.Path(LOG_DIR); log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"bot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logger = logging.getLogger("pos_bot"); logger.setLevel(LOG_LEVEL)
for h in list(logger.handlers): logger.removeHandler(h)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s | %(message)s")
ch = logging.StreamHandler(); ch.setLevel(LOG_LEVEL); ch.setFormatter(fmt)
fh = logging.FileHandler(log_filename, encoding="utf-8"); fh.setLevel(LOG_LEVEL); fh.setFormatter(fmt)
logger.addHandler(ch); logger.addHandler(fh)
logger.info("=== Bot started, log file: %s ===", log_filename)

# ==================== ЭФЕМЕРНЫЕ СООБЩЕНИЯ ====================
ALLOW_EPHEMERAL_RESPONSES = os.getenv("ALLOW_EPHEMERAL_RESPONSES", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def should_use_ephemeral(interaction: Optional[discord.Interaction]) -> bool:
    """Возвращает, нужно ли отправлять ответ как ephemeral."""

    if not ALLOW_EPHEMERAL_RESPONSES:
        return False
    if interaction is None:
        return False
    return interaction.guild is not None

# ==================== ХЕЛПЕРЫ ====================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def format_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def format_ts(ts: Optional[str]) -> str:
    if not ts:
        return "?"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    return format_dt(dt)

def parse_iso_dt(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def clean_resource_name(resource: str) -> str:
    return (resource or "").strip().strip('"').strip("'").strip()

def is_admin_user(interaction: discord.Interaction) -> bool:
    perms = getattr(interaction.user, "guild_permissions", None)
    return bool(perms and (perms.administrator or perms.manage_guild))

# --- безопасная отправка длинных сообщений ---
MAX_DISCORD_MSG = 2000
SAFE_LEN = 1900

async def send_long(interaction: discord.Interaction, text: str, ephemeral: bool = False, title: str = "Message"):
    import io
    if len(text) <= SAFE_LEN:
        await interaction.followup.send(text, ephemeral=ephemeral)
        return
    lines = text.split("\n")
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
        add = len(ln) + 1
        if cur_len + add > SAFE_LEN and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [ln], len(ln) + 1
        else:
            cur.append(ln); cur_len += add
    if cur: chunks.append("\n".join(cur))
    if len(chunks) > 5:
        buf = io.BytesIO(text.encode("utf-8")); buf.seek(0)
        await interaction.followup.send(
            content=f"📄 {title}: список слишком большой, приложил файлом.",
            file=discord.File(buf, filename=f"{title.lower()}.txt"),
            ephemeral=ephemeral,
        )
        return
    total = len(chunks)
    for i, ch in enumerate(chunks, 1):
        header = f"{title} (часть {i}/{total})\n" if total > 1 else ""
        await interaction.followup.send(header + ch, ephemeral=ephemeral)


async def run_subprocess(cmd: List[str], cwd: Optional[pathlib.Path] = None) -> Tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode,
        stdout.decode("utf-8", errors="ignore"),
        stderr.decode("utf-8", errors="ignore"),
    )


def _sanitize_log_line(line: str) -> Optional[str]:
    if line is None:
        return None
    stripped = line.strip()
    if not stripped:
        return None
    sanitized = re.sub(
        r"https?://github\.com/\S+",
        "GitHub (ссылка скрыта)",
        stripped,
        flags=re.IGNORECASE,
    )
    return sanitized


def _extend_logs(logs: List[str], text: Optional[str]) -> None:
    if not text:
        return
    for line in text.splitlines():
        sanitized = _sanitize_log_line(line)
        if sanitized:
            logs.append(sanitized)


async def update_bot_repository(branch: Optional[str], reinstall_deps: bool) -> Tuple[bool, str]:
    git_path = shutil.which("git")
    if not git_path:
        return False, "Команда git не найдена на сервере."

    repo_dir = PROJECT_ROOT
    if not (repo_dir / ".git").exists():
        return False, f"Каталог {repo_dir} не является git-репозиторием."

    logs: List[str] = []

    target_branch = branch or os.getenv("BOT_UPDATE_BRANCH")

    if not target_branch:
        code, out, err = await run_subprocess(
            [git_path, "rev-parse", "--abbrev-ref", "HEAD"],
            repo_dir,
        )
        _extend_logs(logs, "$ git rev-parse --abbrev-ref HEAD")
        _extend_logs(logs, out)
        _extend_logs(logs, err)
        if code == 0:
            current_branch = out.strip()
            if current_branch and current_branch != "HEAD":
                target_branch = current_branch
        if not target_branch:
            target_branch = "master"
            _extend_logs(
                logs,
                "Ветка не указана, используем master по умолчанию.",
            )
    commands: List[List[str]] = []

    if target_branch:
        commands.append([git_path, "fetch", "origin", target_branch])
        commands.append([git_path, "checkout", target_branch])
        commands.append([git_path, "pull", "--ff-only", "origin", target_branch])
    else:
        commands.append([git_path, "pull", "--ff-only"])

    for cmd in commands:
        code, out, err = await run_subprocess(cmd, repo_dir)
        _extend_logs(logs, "$ " + " ".join(cmd))
        _extend_logs(logs, out)
        _extend_logs(logs, err)
        if code != 0:
            _extend_logs(logs, f"Команда завершилась с кодом {code}.")
            return False, "\n".join(logs)

    if reinstall_deps:
        req_file = repo_dir / "requirements.txt"
        if not req_file.exists():
            _extend_logs(logs, "requirements.txt не найден, пропускаю установку зависимостей.")
        else:
            pip_cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            code, out, err = await run_subprocess(pip_cmd, repo_dir)
            _extend_logs(logs, "$ " + " ".join(pip_cmd))
            _extend_logs(logs, out)
            _extend_logs(logs, err)
            if code != 0:
                _extend_logs(logs, f"Команда завершилась с кодом {code}.")
                return False, "\n".join(logs)

    if not logs:
        _extend_logs(logs, "Команды выполнены.")

    return True, "\n".join(logs)


def prepare_service_restart() -> Tuple[Optional[str], Optional[asyncio.Task]]:
    """Планирует перезапуск сервиса после успешного обновления."""

    if not BOT_SERVICE_NAME:
        return "Перезапуск службы не настроен: переменная BOT_SERVICE_NAME не задана.", None

    systemctl = shutil.which("systemctl")
    if not systemctl:
        return "⚠️ Перезапуск службы пропущен: systemctl не найден в системе.", None

    async def _restart() -> None:
        try:
            await asyncio.sleep(5)
            logger.info("Перезапускаю службу %s по команде updatebot.", BOT_SERVICE_NAME)
            code, out, err = await run_subprocess(
                [systemctl, "restart", "--no-block", BOT_SERVICE_NAME]
            )
            if code != 0:
                logger.error(
                    "Не удалось перезапустить службу %s (код %s): %s %s",
                    BOT_SERVICE_NAME,
                    code,
                    out.strip(),
                    err.strip(),
                )
        except Exception:
            logger.exception("Ошибка при перезапуске службы %s", BOT_SERVICE_NAME)

    loop = asyncio.get_running_loop()
    task = loop.create_task(_restart())
    return f"🔄 Служба **{BOT_SERVICE_NAME}** будет перезапущена в ближайшие секунды.", task


def remember_successful_update(interaction: discord.Interaction) -> None:
    if interaction.channel_id is None:
        return
    payload = {
        "channel_id": interaction.channel_id,
        "guild_id": interaction.guild_id,
        "user_id": getattr(interaction.user, "id", None),
        "timestamp": now_utc_iso(),
    }
    try:
        UPDATE_STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        logger.exception("Не удалось сохранить информацию об успешном обновлении.")


async def notify_after_restart() -> None:
    if not UPDATE_STATE_PATH.exists():
        return
    try:
        data = json.loads(UPDATE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Не удалось прочитать файл состояния обновления.")
        try:
            UPDATE_STATE_PATH.unlink(missing_ok=True)
        except Exception:
            logger.exception("Не удалось удалить файл состояния обновления.")
        return

    channel_id = data.get("channel_id")
    user_id = data.get("user_id")
    timestamp = data.get("timestamp")

    try:
        channel = None
        if channel_id is not None:
            channel = bot.get_channel(int(channel_id))
            if channel is None:
                try:
                    channel = await bot.fetch_channel(int(channel_id))
                except Exception:
                    logger.exception("Не удалось получить канал %s для уведомления об обновлении.", channel_id)
        if channel is not None:
            message = "✅ Бот обновился и перезапустился."
            if user_id:
                message += f" Инициатор: <@{int(user_id)}>"
            if timestamp:
                message += f" (обновлено {format_ts(timestamp)})."
            await channel.send(message)
    except Exception:
        logger.exception("Не удалось отправить уведомление об обновлении.")
    finally:
        try:
            UPDATE_STATE_PATH.unlink(missing_ok=True)
        except Exception:
            logger.exception("Не удалось удалить файл состояния обновления после уведомления.")

# --- проверка токена Discord ---
def validate_and_clean_token(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    token = "".join(raw.split())
    token = "".join(ch for ch in token if ch.isprintable())
    parts = token.split(".")
    if len(parts) != 3 or any(len(p) == 0 for p in parts):
        logger.error("Токен выглядит некорректно: ожидалось 3 части через точку.")
        return None
    pattern = re.compile(r"^[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{20,}$")
    if not pattern.match(token):
        logger.error("Токен не проходит базовую проверку формата.")
        return None
    try:
        pad = '=' * (-len(parts[0]) % 4)
        base64.urlsafe_b64decode(parts[0] + pad)
    except Exception:
        logger.warning("Первая часть токена не декодируется как base64url — возможно, токен валиден, продолжаем.")
    return token

# ==================== БД ====================
def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

DDL = [
    """
    CREATE TABLE IF NOT EXISTS pos (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        guild_id      INTEGER NOT NULL,
        owner_user_id INTEGER NOT NULL,
        name          TEXT NOT NULL,
        system        TEXT NOT NULL,
        constellation TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS pos_defaults (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        guild_id   INTEGER NOT NULL,
        user_id    INTEGER,
        slots      INTEGER NOT NULL,
        drills     INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_pos_guild_name ON pos(guild_id, name);",
    "CREATE INDEX IF NOT EXISTS ix_pos_owner ON pos(guild_id, owner_user_id);",
    """
    CREATE TABLE IF NOT EXISTS pos_planet (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        pos_id         INTEGER NOT NULL,
        planet_id      INTEGER NOT NULL,
        resource       TEXT NOT NULL,
        drills_count   INTEGER NOT NULL,
        rate           REAL NOT NULL,
        created_at     TEXT NOT NULL,
        UNIQUE(pos_id, planet_id),
        FOREIGN KEY (pos_id) REFERENCES pos(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS pos_update_ack (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        pos_id        INTEGER NOT NULL UNIQUE,
        user_id       INTEGER NOT NULL,
        confirmed_at  TEXT NOT NULL,
        FOREIGN KEY (pos_id) REFERENCES pos(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS guild_needs_row (
        guild_id        INTEGER NOT NULL,
        resource        TEXT NOT NULL,
        amount_required REAL NOT NULL,
        updated_at      TEXT NOT NULL,
        UNIQUE(guild_id, resource)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS isk_price (
        resource  TEXT PRIMARY KEY,
        price     REAL NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS guild_have (
        guild_id     INTEGER NOT NULL,
        resource     TEXT NOT NULL,
        amount_units REAL NOT NULL,
        unit_price   REAL,
        updated_at   TEXT NOT NULL,
        UNIQUE(guild_id, resource)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS planet_resources (
        planet_id     INTEGER,
        constellation TEXT,
        system        TEXT,
        planet_name   TEXT,
        resource      TEXT,
        output        REAL
    );
    """,
    "CREATE INDEX IF NOT EXISTS ix_have_guild ON guild_have(guild_id);",
    "CREATE INDEX IF NOT EXISTS ix_pr_constellation ON planet_resources(constellation);",
    "CREATE INDEX IF NOT EXISTS ix_pr_system ON planet_resources(system);",
    "CREATE INDEX IF NOT EXISTS ix_pr_resource ON planet_resources(resource);",
    "CREATE INDEX IF NOT EXISTS ix_pos_defaults_lookup ON pos_defaults(guild_id, user_id);",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_pos_defaults_guild_only ON pos_defaults(guild_id) WHERE user_id IS NULL;",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_pos_defaults_guild_user ON pos_defaults(guild_id, user_id) WHERE user_id IS NOT NULL;",
    """
    CREATE TABLE IF NOT EXISTS resource_ping (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        guild_id       INTEGER NOT NULL,
        channel_id     INTEGER NOT NULL,
        message_id     INTEGER NOT NULL,
        author_user_id INTEGER NOT NULL,
        content        TEXT,
        role_id        INTEGER,
        created_at     TEXT NOT NULL,
        active         INTEGER NOT NULL DEFAULT 1
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS resource_ping_submission (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ping_id      INTEGER NOT NULL,
        guild_id     INTEGER NOT NULL,
        user_id      INTEGER NOT NULL,
        submitted_at TEXT NOT NULL,
        UNIQUE(ping_id, user_id),
        FOREIGN KEY (ping_id) REFERENCES resource_ping(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS resource_ping_submission_item (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ping_id      INTEGER NOT NULL,
        guild_id     INTEGER NOT NULL,
        user_id      INTEGER NOT NULL,
        resource     TEXT NOT NULL,
        amount_units REAL NOT NULL,
        unit_price   REAL,
        submitted_at TEXT NOT NULL,
        UNIQUE(ping_id, user_id, resource),
        FOREIGN KEY (ping_id) REFERENCES resource_ping(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS resource_ping_reminder (
        ping_id          INTEGER NOT NULL,
        user_id          INTEGER NOT NULL,
        last_reminded_at TEXT NOT NULL,
        PRIMARY KEY (ping_id, user_id),
        FOREIGN KEY (ping_id) REFERENCES resource_ping(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS resource_ping_thread (
        ping_id    INTEGER NOT NULL,
        user_id    INTEGER NOT NULL,
        thread_id  INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (ping_id, user_id),
        FOREIGN KEY (ping_id) REFERENCES resource_ping(id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS ix_resource_ping_guild ON resource_ping(guild_id);",
    "CREATE INDEX IF NOT EXISTS ix_resource_ping_submission_ping ON resource_ping_submission(ping_id);",
    "CREATE INDEX IF NOT EXISTS ix_resource_ping_submission_user ON resource_ping_submission(user_id);",
    "CREATE INDEX IF NOT EXISTS ix_resource_ping_submission_time ON resource_ping_submission(submitted_at);",
    "CREATE INDEX IF NOT EXISTS ix_resource_ping_submission_item_ping ON resource_ping_submission_item(ping_id);",
    "CREATE INDEX IF NOT EXISTS ix_resource_ping_submission_item_user ON resource_ping_submission_item(user_id);",
    "CREATE INDEX IF NOT EXISTS ix_resource_ping_reminder_time ON resource_ping_reminder(last_reminded_at);",
]


class PingNotFoundError(Exception):
    pass


class PingInactiveError(Exception):
    pass

def _table_cols(conn: sqlite3.Connection, table: str) -> set:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}

def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    for stmt in DDL:
        cur.execute(stmt)

    # planet_resources: гарантируем наличие таблицы и ключевых колонок
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='planet_resources'")
    if not cur.fetchone():
        logger.info("Миграция: создаю таблицу planet_resources")
        cur.execute(
            """
            CREATE TABLE planet_resources (
                planet_id     INTEGER,
                constellation TEXT,
                system        TEXT,
                planet_name   TEXT,
                resource      TEXT,
                output        REAL
            );
            """
        )
    else:
        pr_cols = _table_cols(conn, "planet_resources")
        for col_def in [
            "planet_id INTEGER",
            "constellation TEXT",
            "system TEXT",
            "planet_name TEXT",
            "resource TEXT",
            "output REAL",
        ]:
            col_name = col_def.split()[0]
            if col_name not in pr_cols:
                logger.info("Миграция: добавляю колонку planet_resources.%s", col_name)
                cur.execute(f"ALTER TABLE planet_resources ADD COLUMN {col_def}")

    # Миграции для старых баз
    def _add_col_if_missing(table: str, col_def: str):
        col = col_def.split()[0]
        if col not in _table_cols(conn, table):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")

    def _backfill_timestamp(table: str, col: str, value: Optional[str] = None):
        if col in _table_cols(conn, table):
            v = value or now_utc_iso()
            conn.execute(f"UPDATE {table} SET {col}=? WHERE {col} IS NULL OR {col}=''", (v,))

    # pos: гарантируем created_at/updated_at
    _add_col_if_missing("pos", "created_at TEXT")
    _add_col_if_missing("pos", "updated_at TEXT")
    _backfill_timestamp("pos", "created_at")
    _backfill_timestamp("pos", "updated_at")

    # pos_planet: если был end_time_utc — пересоберём таблицу без него
    cols = _table_cols(conn, "pos_planet")
    if "end_time_utc" in cols:
        logger.info("Миграция: удаляю pos_planet.end_time_utc (пересоздание таблицы)")
        cur.execute("PRAGMA foreign_keys=OFF")
        try:
            cur.execute("DROP INDEX IF EXISTS ix_pp_end")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pos_planet_new (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    pos_id         INTEGER NOT NULL,
                    planet_id      INTEGER NOT NULL,
                    resource       TEXT NOT NULL,
                    drills_count   INTEGER NOT NULL,
                    rate           REAL NOT NULL,
                    created_at     TEXT NOT NULL,
                    UNIQUE(pos_id, planet_id),
                    FOREIGN KEY (pos_id) REFERENCES pos(id) ON DELETE CASCADE
                );
            """)
            cur.execute("""
                INSERT INTO pos_planet_new (id, pos_id, planet_id, resource, drills_count, rate, created_at)
                SELECT id, pos_id, planet_id, resource, drills_count, rate, COALESCE(created_at, ?)
                FROM pos_planet
            """, (now_utc_iso(),))
            cur.execute("DROP TABLE pos_planet")
            cur.execute("ALTER TABLE pos_planet_new RENAME TO pos_planet")
        finally:
            cur.execute("PRAGMA foreign_keys=ON")

    # guild_have/isk_price/guild_needs_row: гарантируем updated_at
    _add_col_if_missing("guild_have", "unit_price REAL")
    _add_col_if_missing("guild_have", "updated_at TEXT"); _backfill_timestamp("guild_have", "updated_at")
    _add_col_if_missing("isk_price", "updated_at TEXT");  _backfill_timestamp("isk_price", "updated_at")
    _add_col_if_missing("guild_needs_row", "updated_at TEXT"); _backfill_timestamp("guild_needs_row", "updated_at")

    conn.commit()

def ensure_db_ready() -> sqlite3.Connection:
    conn = connect_db(DB_PATH)
    ensure_schema(conn)
    return conn

# ==================== ДАННЫЕ / УТИЛИТЫ ====================
def find_constellation_by_system(conn: sqlite3.Connection, system: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("""SELECT constellation FROM planet_resources WHERE system=? GROUP BY constellation""", (system,))
    row = cur.fetchone()
    return row["constellation"] if row else None

def get_distinct_systems(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("""SELECT system FROM planet_resources GROUP BY system ORDER BY system""")
    return [r["system"] for r in cur.fetchall()]

def load_guild_have(conn: sqlite3.Connection, guild_id: int) -> Dict[str, float]:
    cur = conn.cursor()
    cur.execute("""SELECT resource, amount_units FROM guild_have WHERE guild_id=?""", (guild_id,))
    return {row["resource"]: float(row["amount_units"]) for row in cur.fetchall()}

def load_have_prices(conn: sqlite3.Connection, guild_id: int) -> Dict[str, float]:
    cur = conn.cursor()
    cur.execute("""SELECT resource, unit_price FROM guild_have WHERE guild_id=? AND unit_price IS NOT NULL""", (guild_id,))
    return {row["resource"]: float(row["unit_price"]) for row in cur.fetchall()}

def load_guild_needs(conn: sqlite3.Connection, guild_id: int) -> Dict[str, float]:
    cur = conn.cursor()
    cur.execute("""SELECT resource, amount_required FROM guild_needs_row WHERE guild_id=?""", (guild_id,))
    return {row["resource"]: float(row["amount_required"]) for row in cur.fetchall()}

def active_production_rates(conn: sqlite3.Connection, guild_id: int) -> Dict[str, float]:
    cur = conn.cursor()
    cur.execute("""
        SELECT pp.resource, SUM(pp.rate * pp.drills_count) AS rph
        FROM pos_planet pp
        JOIN pos p ON p.id = pp.pos_id
        WHERE p.guild_id=?
        GROUP BY pp.resource
    """, (guild_id,))
    return {row["resource"]: float(row["rph"] or 0.0) for row in cur.fetchall()}

def active_assignments_coverage(conn: sqlite3.Connection, guild_id: int, horizon_hours: int) -> Dict[str, float]:
    cur = conn.cursor()
    cur.execute("""
        SELECT pp.resource, pp.rate, pp.drills_count
        FROM pos_planet pp
        JOIN pos p ON p.id = pp.pos_id
        WHERE p.guild_id=?
    """, (guild_id,))
    cov: Dict[str, float] = {}
    eff_h = float(horizon_hours)
    for r in cur.fetchall():
        produced = float(r["rate"]) * int(r["drills_count"]) * eff_h
        res = r["resource"]
        cov[res] = cov.get(res, 0.0) + produced
    return cov

def subtract_amounts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, av in a.items():
        left = max(0.0, float(av) - float(b.get(k, 0.0)))
        if left > 0:
            out[k] = left
    return out

def _pos_defaults_row(conn: sqlite3.Connection, guild_id: int, user_id: Optional[int]) -> Optional[sqlite3.Row]:
    cur = conn.cursor()
    if user_id is None:
        cur.execute(
            "SELECT id, slots, drills, updated_at FROM pos_defaults WHERE guild_id=? AND user_id IS NULL LIMIT 1",
            (guild_id,),
        )
    else:
        cur.execute(
            "SELECT id, slots, drills, updated_at FROM pos_defaults WHERE guild_id=? AND user_id=? LIMIT 1",
            (guild_id, user_id),
        )
    return cur.fetchone()

def set_pos_defaults(conn: sqlite3.Connection, guild_id: int, slots: int, drills: int, user_id: Optional[int] = None):
    ts = now_utc_iso()
    row = _pos_defaults_row(conn, guild_id, user_id)
    cur = conn.cursor()
    if row:
        cur.execute(
            "UPDATE pos_defaults SET slots=?, drills=?, updated_at=? WHERE id=?",
            (int(slots), int(drills), ts, int(row["id"])),
        )
    else:
        cur.execute(
            "INSERT INTO pos_defaults(guild_id, user_id, slots, drills, updated_at) VALUES(?,?,?,?,?)",
            (guild_id, user_id, int(slots), int(drills), ts),
        )
    conn.commit()

def clear_user_pos_defaults(conn: sqlite3.Connection, guild_id: int, user_id: int) -> bool:
    cur = conn.cursor()
    cur.execute("DELETE FROM pos_defaults WHERE guild_id=? AND user_id=?", (guild_id, user_id))
    changed = cur.rowcount > 0
    if changed:
        conn.commit()
    return changed

def get_effective_pos_defaults(
    conn: sqlite3.Connection, guild_id: int, user_id: Optional[int]
) -> Dict[str, object]:
    row_user = _pos_defaults_row(conn, guild_id, user_id) if user_id is not None else None
    if user_id is not None and row_user:
        return {
            "slots": int(row_user["slots"]),
            "drills": int(row_user["drills"]),
            "source": "user",
            "updated_at": row_user["updated_at"],
        }

    row_guild = _pos_defaults_row(conn, guild_id, None)
    if row_guild:
        return {
            "slots": int(row_guild["slots"]),
            "drills": int(row_guild["drills"]),
            "source": "guild",
            "updated_at": row_guild["updated_at"],
        }

    return {
        "slots": DEFAULT_SLOTS,
        "drills": DEFAULT_DRILLS,
        "source": "env",
        "updated_at": None,
    }

def describe_pos_defaults_source(source: str) -> str:
    if source == "user":
        return "личные настройки"
    if source == "guild":
        return "настройки сервера"
    return "значения окружения"

def build_pos_defaults_report(
    conn: sqlite3.Connection, guild_id: int, user_id: Optional[int]
) -> str:
    row_user = _pos_defaults_row(conn, guild_id, user_id) if user_id is not None else None
    row_guild = _pos_defaults_row(conn, guild_id, None)

    def fmt_row(row: Optional[sqlite3.Row]) -> str:
        if not row:
            return "не задано"
        ts = row["updated_at"] or "?"
        return f"слотов **{int(row['slots'])}**, буров **{int(row['drills'])}** (обновлено {ts})"

    lines = [
        "**Личные:** " + (fmt_row(row_user) if user_id is not None else "не поддерживается"),
        "**Сервер:** " + fmt_row(row_guild),
    ]

    eff = get_effective_pos_defaults(conn, guild_id, user_id)
    eff_ts = eff.get("updated_at")
    ts_tail = f" (обновлено {eff_ts})" if eff_ts else ""
    lines.append(
        f"**Активно для тебя:** слотов **{eff['slots']}**, буров **{eff['drills']}** — "
        f"{describe_pos_defaults_source(eff['source'])}{ts_tail}."
    )
    return "\n".join(lines)

def set_price(conn: sqlite3.Connection, resource: str, price: float):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO isk_price(resource, price, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(resource) DO UPDATE SET price=excluded.price, updated_at=excluded.updated_at
    """, (clean_resource_name(resource), float(price), now_utc_iso()))
    conn.commit()

def get_prices(conn: sqlite3.Connection) -> Dict[str, float]:
    cur = conn.cursor()
    cur.execute("SELECT resource, price FROM isk_price")
    return {row["resource"]: float(row["price"]) for row in cur.fetchall()}

def get_price(conn: sqlite3.Connection, resource: str) -> Optional[float]:
    cur = conn.cursor()
    cur.execute("SELECT price FROM isk_price WHERE resource=?", (clean_resource_name(resource),))
    row = cur.fetchone()
    return float(row["price"]) if row else None

# --- кандидаты планет по созвездию ---
def build_candidates(conn: sqlite3.Connection, constellation: str) -> Dict[str, List[dict]]:
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            CASE WHEN typeof(planet_id)='integer' AND planet_id>0 THEN planet_id ELSE rowid END AS planet_id,
            system, planet_name, resource, output
        FROM planet_resources
        WHERE constellation=? AND output>0
        ORDER BY resource, output DESC
    """, (constellation,))
    rows = cur.fetchall()
    out: Dict[str, List[dict]] = {}
    for r in rows:
        res = r["resource"]
        out.setdefault(res, []).append({
            "planet_id": int(r["planet_id"]),
            "system": r["system"],
            "planet": r["planet_name"],
            "output": float(r["output"]),
        })
    return out

# ==================== ПЛАНИРОВЩИК ====================
class Assignment:
    __slots__ = ("planet_id","resource","drills","rate","system","planet_name","base_out","isk_per_hour")
    def __init__(self, planet_id:int, resource:str, drills:int, rate:float,
                 system:str, planet_name:str, base_out:float, isk_per_hour:float=0.0):
        self.planet_id=planet_id; self.resource=resource; self.drills=drills
        self.rate=rate; self.system=system; self.planet_name=planet_name
        self.base_out=base_out; self.isk_per_hour=isk_per_hour

def plan_assignments(
    conn: sqlite3.Connection,
    guild_id: int,
    constellation: str,
    rest_units: Dict[str, float],
    horizon_hours: int,
    slots: int = DEFAULT_SLOTS,
    drills: int = DEFAULT_DRILLS,
    beta: float = DEFAULT_BETA,
    prices: Optional[Dict[str, float]] = None
) -> List[Assignment]:
    candidates = build_candidates(conn, constellation)
    need_left = {r: float(v) for r,v in rest_units.items() if v>0}
    next_idx: Dict[str,int] = {r:0 for r in candidates}
    used_planets: set[int] = set()
    per_res_taken: Dict[str,int] = {}
    global_rph = active_production_rates(conn, guild_id)

    def advance(res:str)->Optional[dict]:
        lst = candidates.get(res) or []
        i = next_idx.get(res,0)
        while i < len(lst) and int(lst[i]["planet_id"]) in used_planets:
            i += 1
        next_idx[res]=i
        return lst[i] if i < len(lst) else None

    def rows_by_worst_eta(include_slot: bool) -> List[Tuple[str,float,dict,float]]:
        rows=[]
        for res, left_units in need_left.items():
            if left_units <= 0: continue
            cand = advance(res)
            if not cand: continue
            base_out = max(float(cand["output"]), 1e-9)
            mult = (beta ** per_res_taken.get(res,0)) if beta<1.0 else 1.0
            slot_rph = base_out * drills * mult
            base_global = float(global_rph.get(res, 0.0))
            denom = base_global + (slot_rph if include_slot else 0.0)
            eta_h = (left_units / denom) if denom>0 else float("inf")
            rows.append((res, eta_h, cand, slot_rph))
        rows.sort(key=lambda t: t[1], reverse=True)
        return rows

    assignments: List[Assignment] = []

    while len(assignments) < slots:
        rows = rows_by_worst_eta(include_slot=True)
        if not rows: break
        res, _eta, cand, slot_rph = rows[0]
        pid = int(cand["planet_id"]); base_out=float(cand["output"])
        mult = (beta ** per_res_taken.get(res,0)) if beta<1.0 else 1.0
        produced_units = slot_rph * horizon_hours
        need_left[res] = max(0.0, need_left[res] - produced_units)
        a = Assignment(pid, res, drills, base_out*mult, cand["system"], cand["planet"], base_out,
                       (base_out*mult*drills*(prices.get(res,0.0) if prices else 0.0)))
        assignments.append(a); used_planets.add(pid); per_res_taken[res]=per_res_taken.get(res,0)+1
        global_rph[res] = global_rph.get(res,0.0) + slot_rph

    return assignments

def upsert_assignments(conn: sqlite3.Connection, pos_id: int, assignments: List[Assignment]):
    cur = conn.cursor()
    for a in assignments:
        cur.execute("""
            INSERT INTO pos_planet (pos_id, planet_id, resource, drills_count, rate, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(pos_id, planet_id) DO UPDATE
              SET resource=excluded.resource,
                  drills_count=excluded.drills_count,
                  rate=excluded.rate
        """, (pos_id, a.planet_id, a.resource, a.drills, a.rate, now_utc_iso()))
    conn.commit()

def format_discord_response(assignments: List[Assignment]) -> str:
    if not assignments:
        return "_Ничего не подобрано._"
    lines = [f"Всего назначений: **{len(assignments)}**"]
    for i, a in enumerate(assignments, 1):
        total_rate = a.rate * a.drills
        tail = f" · ≈ {a.isk_per_hour:,.0f} ISK/h".replace(",", " ") if a.isk_per_hour else ""
        lines.append(
            f"{i}. {a.system} · {a.planet_name} · **{a.resource}** · drills={a.drills} · "
            f"base={a.base_out:.2f}/h/bore → **{total_rate:,.2f}/ч**{tail}".replace(",", " ")
        )
    return "\n".join(lines)

def compute_pos_assignments(
    conn: sqlite3.Connection,
    guild_id: int,
    constellation: str,
    slots: int,
    drills: int,
) -> List[Assignment]:
    needs_units = load_guild_needs(conn, guild_id)
    have_units = load_guild_have(conn, guild_id)
    needs_after_have = subtract_amounts(needs_units, have_units)
    coverage_units = active_assignments_coverage(conn, guild_id, DEFAULT_HOURS)
    rest_units = subtract_amounts(needs_after_have, coverage_units)

    prices_have = load_have_prices(conn, guild_id)
    prices_general = get_prices(conn)
    merged_prices = dict(prices_general)
    merged_prices.update(prices_have)

    return plan_assignments(
        conn=conn,
        guild_id=guild_id,
        constellation=constellation,
        rest_units=rest_units,
        horizon_hours=DEFAULT_HOURS,
        slots=slots,
        drills=drills,
        beta=DEFAULT_BETA,
        prices=merged_prices,
    )

def build_pos_assignment_message(
    name: str,
    system: str,
    constellation: str,
    assignments: List[Assignment],
    slots_val: int,
    drills_val: int,
    defaults: Dict[str, object],
    slots_override: bool,
    drills_override: bool,
) -> str:
    def format_source(overridden: bool) -> str:
        if overridden:
            return "указано вручную"
        label = describe_pos_defaults_source(str(defaults.get("source", "env")))
        ts = defaults.get("updated_at")
        return f"{label} (обновлено {ts})" if ts else label

    assignment_text = format_discord_response(assignments)
    return (
        f"✅ POS **{name}** ({system}, {constellation}) обновлён.\n"
        f"Слотов: **{slots_val}** ({format_source(slots_override)}), "
        f"буров/планету: **{drills_val}** ({format_source(drills_override)}).\n"
        f"Расчёт целей: **цели − склад − текущая добыча**.\n\n"
        f"Назначения для POS в **{system}** (ед/час):\n"
        f"{assignment_text}"
    )


def split_pos_assignment_message(full_message: str) -> Tuple[str, str]:
    marker = "Назначения для POS"
    idx = full_message.find(marker)
    if idx == -1:
        return full_message, ""
    header = full_message[:idx].rstrip()
    assignments = full_message[idx:].lstrip()
    return header, assignments


class RefreshPosModal(discord.ui.Modal):
    def __init__(self, guild_id: int, defaults: Dict[str, object]):
        super().__init__(title="Обновить POS")
        self.guild_id = guild_id
        self.defaults = defaults
        self.default_slots = int(defaults.get("slots", DEFAULT_SLOTS))
        self.default_drills = int(defaults.get("drills", DEFAULT_DRILLS))

        self.slots_input = discord.ui.TextInput(
            label="Количество слотов",
            required=False,
            placeholder=f"Оставь пустым — по умолчанию {self.default_slots}",
            style=discord.TextStyle.short,
        )
        self.drills_input = discord.ui.TextInput(
            label="Количество буров на планету",
            required=False,
            placeholder=f"Оставь пустым — по умолчанию {self.default_drills}",
            style=discord.TextStyle.short,
        )
        self.add_item(self.slots_input)
        self.add_item(self.drills_input)

    async def on_submit(self, interaction: discord.Interaction):
        ephemeral = should_use_ephemeral(interaction)
        await interaction.response.defer(thinking=True, ephemeral=ephemeral)

        conn = ensure_db_ready()
        try:
            defaults = get_effective_pos_defaults(conn, self.guild_id, interaction.user.id)
            base_slots = int(defaults.get("slots", self.default_slots))
            base_drills = int(defaults.get("drills", self.default_drills))

            slots_val = base_slots
            drills_val = base_drills
            slots_override = False
            drills_override = False

            raw_slots = (self.slots_input.value or "").strip()
            if raw_slots:
                try:
                    slots_val = int(raw_slots)
                except ValueError:
                    await interaction.followup.send("`slots` должен быть целым числом.", ephemeral=ephemeral)
                    return
                err = _validate_default_bounds(slots_val, "slots")
                if err:
                    await interaction.followup.send(err, ephemeral=ephemeral)
                    return
                slots_override = True

            raw_drills = (self.drills_input.value or "").strip()
            if raw_drills:
                try:
                    drills_val = int(raw_drills)
                except ValueError:
                    await interaction.followup.send("`drills` должен быть целым числом.", ephemeral=ephemeral)
                    return
                err = _validate_default_bounds(drills_val, "drills")
                if err:
                    await interaction.followup.send(err, ephemeral=ephemeral)
                    return
                drills_override = True

            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, name, system, constellation
                FROM pos
                WHERE guild_id=? AND owner_user_id=?
                ORDER BY name
                """,
                (self.guild_id, interaction.user.id),
            )
            rows = cur.fetchall()
            if not rows:
                await interaction.followup.send("Для этого сервера у тебя нет POS.", ephemeral=ephemeral)
                return

            for row in rows:
                pos_id = int(row["id"])
                name = row["name"]
                system = row["system"]
                constellation = row["constellation"]

                cur.execute("DELETE FROM pos_planet WHERE pos_id=?", (pos_id,))
                assignments = compute_pos_assignments(
                    conn=conn,
                    guild_id=self.guild_id,
                    constellation=constellation,
                    slots=slots_val,
                    drills=drills_val,
                )
                upsert_assignments(conn, pos_id, assignments)
                cur.execute("UPDATE pos SET updated_at=? WHERE id=?", (now_utc_iso(), pos_id))
                conn.commit()

                full_message = build_pos_assignment_message(
                    name=name,
                    system=system,
                    constellation=constellation,
                    assignments=assignments,
                    slots_val=slots_val,
                    drills_val=drills_val,
                    defaults=defaults,
                    slots_override=slots_override,
                    drills_override=drills_override,
                )

                header_text, assignments_text = split_pos_assignment_message(full_message)
                ack_ts = get_pos_ack_timestamp(conn, pos_id)
                view = PosUpdateAckView(pos_id, already_confirmed=bool(ack_ts))

                note = (
                    f"ℹ️ Последняя отметка: {format_ts(ack_ts)}."
                    if ack_ts
                    else "🛠️ Нажми кнопку ниже, когда обновишь планеты на POS."
                )
                header_with_note = header_text.strip()
                if note:
                    header_with_note = (
                        f"{header_with_note}\n\n{note}" if header_with_note else note
                    )

                await interaction.followup.send(
                    header_with_note,
                    ephemeral=ephemeral,
                    view=view,
                )

                if assignments_text:
                    await send_long(
                        interaction,
                        assignments_text,
                        ephemeral=ephemeral,
                        title=f"Назначения POS {name}",
                    )

        except Exception as e:
            logger.exception("refresh_pos_modal error: %s", e)
            await interaction.followup.send(f"Ошибка: {e}", ephemeral=ephemeral)
        finally:
            conn.close()

class RefreshPosView(discord.ui.View):
    def __init__(self, guild_id: int):
        super().__init__(timeout=None)
        self.guild_id = guild_id

    @discord.ui.button(label="Обновить", style=discord.ButtonStyle.primary, custom_id="refresh_pos_button")
    async def refresh_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        conn = ensure_db_ready()
        try:
            defaults = get_effective_pos_defaults(conn, self.guild_id, interaction.user.id)
        finally:
            conn.close()
        await interaction.response.send_modal(RefreshPosModal(self.guild_id, defaults))


class PosUpdateAckButton(discord.ui.Button):
    def __init__(self, pos_id: int, already_confirmed: bool):
        label = "Обновить отметку" if already_confirmed else "Отметить обновление"
        super().__init__(
            label=label,
            style=discord.ButtonStyle.success,
            custom_id=f"pos_ack:{pos_id}",
        )
        self.pos_id = pos_id

    async def callback(self, interaction: discord.Interaction):
        conn = ensure_db_ready()
        now_iso = now_utc_iso()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT owner_user_id, name FROM pos WHERE id=?",
                (self.pos_id,),
            )
            row = cur.fetchone()
            if not row:
                await interaction.response.send_message(
                    "POS не найден (возможно, удалён).",
                    ephemeral=should_use_ephemeral(interaction),
                )
                return

            owner_id = int(row["owner_user_id"])
            if owner_id != interaction.user.id:
                await interaction.response.send_message(
                    "⛔ Отметить обновление может только владелец POS.",
                    ephemeral=should_use_ephemeral(interaction),
                )
                return

            cur.execute(
                """
                INSERT INTO pos_update_ack(pos_id, user_id, confirmed_at)
                VALUES(?,?,?)
                ON CONFLICT(pos_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    confirmed_at=excluded.confirmed_at
                """,
                (self.pos_id, interaction.user.id, now_iso),
            )
            conn.commit()
        except Exception as e:
            logger.exception("Не удалось сохранить отметку обновления POS %s: %s", self.pos_id, e)
            await interaction.response.send_message(
                "Ошибка при сохранении отметки. Попробуй позже.",
                ephemeral=should_use_ephemeral(interaction),
            )
            return
        finally:
            conn.close()

        note_text = f"ℹ️ Отметка обновлена: {format_ts(now_iso)}"
        content = interaction.message.content or ""
        lines = content.splitlines() if content else []
        replaced = False
        for idx in range(len(lines) - 1, -1, -1):
            if lines[idx].startswith("ℹ️") or lines[idx].startswith("🛠️"):
                lines[idx] = note_text
                replaced = True
                break
        if not replaced:
            if lines:
                lines.append(note_text)
            else:
                lines = [note_text]

        self.disabled = True
        self.label = "Отмечено"

        await interaction.response.edit_message(
            content="\n".join(lines),
            view=self.view,
        )


class PosUpdateAckView(discord.ui.View):
    def __init__(self, pos_id: int, already_confirmed: bool):
        super().__init__(timeout=7 * 24 * 3600)
        self.add_item(PosUpdateAckButton(pos_id, already_confirmed))


def get_user_resource_assignments(
    conn: sqlite3.Connection, guild_id: int, user_id: int
) -> Dict[str, Dict[str, object]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            pp.planet_id,
            pp.resource,
            pp.drills_count,
            pp.rate,
            pr.system AS system,
            pr.planet_name AS planet_name
        FROM pos_planet pp
        JOIN pos p ON p.id = pp.pos_id
        LEFT JOIN planet_resources pr
          ON (CASE WHEN typeof(pr.planet_id)='integer' AND pr.planet_id>0
                   THEN pr.planet_id ELSE pr.rowid END) = pp.planet_id
        WHERE p.guild_id=? AND p.owner_user_id=?
        ORDER BY LOWER(pp.resource), system, planet_name
        """,
        (guild_id, user_id),
    )
    rows = cur.fetchall()
    result: Dict[str, Dict[str, object]] = OrderedDict()
    for r in rows:
        name = clean_resource_name(r["resource"])
        if not name:
            continue
        key = name.lower()
        info = result.get(key)
        if not info:
            info = {
                "name": name,
                "rate_total": 0.0,
                "planets": [],
            }
            result[key] = info
        drills = int(r["drills_count"] or 0)
        rate = float(r["rate"] or 0.0)
        info["rate_total"] += rate * drills
        info["planets"].append(
            {
                "planet_id": int(r["planet_id"] or 0),
                "system": r["system"] or "?",
                "planet": r["planet_name"] or (f"#{int(r['planet_id'])}" if r["planet_id"] is not None else "?"),
                "drills": drills,
                "rate": rate,
            }
        )
    return result


def get_pos_ack_timestamp(conn: sqlite3.Connection, pos_id: int) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT confirmed_at FROM pos_update_ack WHERE pos_id=?",
        (pos_id,),
    )
    row = cur.fetchone()
    return row["confirmed_at"] if row else None


def clear_pos_ack(conn: sqlite3.Connection, pos_ids: Sequence[int]) -> int:
    if not pos_ids:
        return 0
    placeholders = ",".join("?" * len(pos_ids))
    cur = conn.cursor()
    cur.execute(
        f"DELETE FROM pos_update_ack WHERE pos_id IN ({placeholders})",
        tuple(pos_ids),
    )
    return cur.rowcount or 0


def get_guild_resource_producers(conn: sqlite3.Connection, guild_id: int) -> List[int]:
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT owner_user_id FROM pos WHERE guild_id=? AND owner_user_id IS NOT NULL",
        (guild_id,),
    )
    return [int(r["owner_user_id"]) for r in cur.fetchall() if r["owner_user_id"] is not None]


def compute_resping_delivery_audit(
    conn: sqlite3.Connection, guild_id: int, since_iso: str
) -> Tuple[
    Dict[int, Dict[str, Dict[str, object]]],
    Dict[int, int],
    Dict[int, Dict[str, Dict[str, object]]],
]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.user_id, s.ping_id, s.submitted_at
        FROM resource_ping_submission s
        JOIN resource_ping p ON p.id = s.ping_id
        WHERE p.guild_id=? AND s.submitted_at>=?
        ORDER BY s.user_id, s.submitted_at
        """,
        (guild_id, since_iso),
    )
    submission_rows = cur.fetchall()
    if not submission_rows:
        return {}, {}, {}

    cur.execute(
        """
        SELECT s.user_id, MAX(s.submitted_at) AS prev_at
        FROM resource_ping_submission s
        JOIN resource_ping p ON p.id = s.ping_id
        WHERE p.guild_id=? AND s.submitted_at<?
        GROUP BY s.user_id
        """,
        (guild_id, since_iso),
    )
    prev_rows = cur.fetchall()
    prev_map: Dict[int, Optional[datetime]] = {
        int(r["user_id"]): parse_iso_dt(r["prev_at"])
        for r in prev_rows
        if r["prev_at"]
    }

    cur.execute(
        """
        SELECT ping_id, user_id, resource, amount_units, submitted_at
        FROM resource_ping_submission_item
        WHERE guild_id=? AND submitted_at>=?
        """,
        (guild_id, since_iso),
    )
    item_rows = cur.fetchall()
    items_map: Dict[Tuple[int, int], Dict[str, Dict[str, object]]] = {}
    for row in item_rows:
        key = (int(row["ping_id"]), int(row["user_id"]))
        resource = clean_resource_name(row["resource"])
        if not resource:
            continue
        res_key = resource.lower()
        entry = items_map.setdefault(key, {})
        info = entry.get(res_key)
        if not info:
            info = {"name": resource, "amount": 0.0}
            entry[res_key] = info
        info["amount"] += float(row["amount_units"] or 0.0)

    stats: Dict[int, Dict[str, Dict[str, object]]] = {}
    intervals: Dict[int, int] = {}
    extras: Dict[int, Dict[str, Dict[str, object]]] = {}
    assignment_cache: Dict[int, Dict[str, Dict[str, object]]] = {}

    for row in submission_rows:
        user_id = int(row["user_id"])
        ping_id = int(row["ping_id"])
        submitted_at = parse_iso_dt(row["submitted_at"])
        if submitted_at is None:
            continue
        prev_ts = prev_map.get(user_id)
        assignments = assignment_cache.get(user_id)
        if assignments is None:
            assignments = get_user_resource_assignments(conn, guild_id, user_id)
            assignment_cache[user_id] = assignments
        if not assignments:
            prev_map[user_id] = submitted_at
            continue
        if prev_ts is None:
            prev_map[user_id] = submitted_at
            continue

        delta = submitted_at - prev_ts
        hours = delta.total_seconds() / 3600.0
        if hours <= 0:
            prev_map[user_id] = submitted_at
            continue

        stats_user = stats.setdefault(user_id, {})
        intervals[user_id] = intervals.get(user_id, 0) + 1

        submission_items = items_map.get((ping_id, user_id), {})

        for key, info in assignments.items():
            expected = float(info.get("rate_total") or 0.0) * hours
            if expected <= 0:
                continue
            res_entry = stats_user.setdefault(
                key,
                {
                    "name": info.get("name") or key,
                    "expected": 0.0,
                    "actual": 0.0,
                    "intervals": 0,
                },
            )
            res_entry["expected"] += expected
            item_info = submission_items.get(key)
            if item_info:
                res_entry["actual"] += float(item_info.get("amount") or 0.0)
            res_entry["intervals"] += 1

        if submission_items:
            extra_map = extras.setdefault(user_id, {})
            for key, info in submission_items.items():
                if key in assignments:
                    continue
                extra_entry = extra_map.get(key)
                if not extra_entry:
                    extra_entry = {"name": info.get("name") or key, "amount": 0.0}
                    extra_map[key] = extra_entry
                extra_entry["amount"] += float(info.get("amount") or 0.0)

        prev_map[user_id] = submitted_at

    return stats, intervals, extras


def _compute_ping_submission_totals(
    cur: sqlite3.Cursor, ping_id: int, guild_id: int
) -> Tuple[int, int, int]:
    cur.execute(
        "SELECT COUNT(*) AS cnt FROM resource_ping_submission WHERE ping_id=?",
        (ping_id,),
    )
    total_users_row = cur.fetchone()
    total_users = int(total_users_row["cnt"] or 0) if total_users_row else 0

    cur.execute(
        "SELECT COUNT(*) AS cnt FROM pos WHERE guild_id=?",
        (guild_id,),
    )
    total_pos_row = cur.fetchone()
    total_pos = int(total_pos_row["cnt"] or 0) if total_pos_row else 0

    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM pos p
        WHERE p.guild_id=? AND EXISTS (
            SELECT 1
            FROM resource_ping_submission s
            WHERE s.ping_id=? AND s.user_id = p.owner_user_id
        )
        """,
        (guild_id, ping_id),
    )
    submitted_pos_row = cur.fetchone()
    submitted_pos = int(submitted_pos_row["cnt"] or 0) if submitted_pos_row else 0

    return total_users, submitted_pos, total_pos


def record_ping_submission(
    cur: sqlite3.Cursor, ping_id: int, user_id: int, submitted_at: Optional[str] = None
) -> Tuple[bool, int, int, int, int, int]:
    cur.execute("SELECT guild_id, active FROM resource_ping WHERE id=?", (ping_id,))
    ping_row = cur.fetchone()
    if not ping_row:
        raise PingNotFoundError(f"Ping {ping_id} not found")
    if not bool(ping_row["active"]):
        raise PingInactiveError(f"Ping {ping_id} is inactive")

    guild_id = int(ping_row["guild_id"])
    cur.execute(
        "SELECT submitted_at FROM resource_ping_submission WHERE ping_id=? AND user_id=?",
        (ping_id, user_id),
    )
    already = cur.fetchone() is not None
    ts = submitted_at or now_utc_iso()
    cur.execute(
        """
        INSERT INTO resource_ping_submission (ping_id, guild_id, user_id, submitted_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ping_id, user_id) DO UPDATE
          SET submitted_at=excluded.submitted_at
        """,
        (ping_id, guild_id, user_id, ts),
    )
    cur.execute(
        "DELETE FROM resource_ping_reminder WHERE ping_id=? AND user_id=?",
        (ping_id, user_id),
    )
    cur.execute(
        "SELECT COUNT(*) AS cnt FROM pos WHERE guild_id=? AND owner_user_id=?",
        (guild_id, user_id),
    )
    user_pos_row = cur.fetchone()
    user_pos_count = int(user_pos_row["cnt"] or 0) if user_pos_row else 0

    total_users, submitted_pos, total_pos = _compute_ping_submission_totals(cur, ping_id, guild_id)

    return already, total_users, guild_id, user_pos_count, submitted_pos, total_pos


def get_ping_thread_id(conn: sqlite3.Connection, ping_id: int, user_id: int) -> Optional[int]:
    cur = conn.cursor()
    cur.execute(
        "SELECT thread_id FROM resource_ping_thread WHERE ping_id=? AND user_id=?",
        (ping_id, user_id),
    )
    row = cur.fetchone()
    return int(row["thread_id"]) if row and row["thread_id"] is not None else None


def upsert_ping_thread(conn: sqlite3.Connection, ping_id: int, user_id: int, thread_id: int) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO resource_ping_thread (ping_id, user_id, thread_id, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ping_id, user_id) DO UPDATE
          SET thread_id=excluded.thread_id,
              created_at=excluded.created_at
        """,
        (ping_id, user_id, thread_id, now_utc_iso()),
    )


def delete_ping_thread(conn: sqlite3.Connection, ping_id: int, user_id: int) -> None:
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM resource_ping_thread WHERE ping_id=? AND user_id=?",
        (ping_id, user_id),
    )


async def resolve_thread(thread_id: int) -> Optional[discord.Thread]:
    channel = bot.get_channel(thread_id)
    if isinstance(channel, discord.Thread):
        return channel
    try:
        fetched = await bot.fetch_channel(thread_id)
    except discord.NotFound:
        return None
    except discord.Forbidden:
        logger.warning("Нет доступа к ветке %s", thread_id)
        return None
    except Exception as e:
        logger.exception("Ошибка при получении ветки %s: %s", thread_id, e)
        return None
    return fetched if isinstance(fetched, discord.Thread) else None


async def get_or_create_ping_thread(
    ping_id: int,
    user_id: int,
    message: discord.Message,
    user: Optional[Union[discord.Member, discord.User]],
) -> Tuple[Optional[discord.Thread], bool]:
    conn = ensure_db_ready()
    try:
        existing_id = get_ping_thread_id(conn, ping_id, user_id)
    finally:
        conn.close()

    if existing_id:
        thread = await resolve_thread(existing_id)
        if thread is not None:
            if getattr(thread, "archived", False):
                try:
                    await thread.edit(archived=False)
                except Exception as e:
                    logger.warning("Не удалось разархивировать ветку %s: %s", thread.id, e)
            return thread, False

    display_name = None
    if user is not None:
        display_name = getattr(user, "display_name", None) or getattr(user, "name", None)
    if not display_name:
        display_name = f"User {user_id}"
    base_name = f"Сдача • {display_name}"
    thread_name = base_name[:95]

    try:
        thread = await message.create_thread(
            name=thread_name,
            auto_archive_duration=1440,
        )
    except discord.HTTPException as e:
        logger.warning("Не удалось создать ветку для ping %s и пользователя %s: %s", ping_id, user_id, e)
        return None, False
    except Exception as e:
        logger.exception("Ошибка при создании ветки для ping %s и пользователя %s: %s", ping_id, user_id, e)
        return None, False

    conn = ensure_db_ready()
    try:
        upsert_ping_thread(conn, ping_id, user_id, int(thread.id))
        conn.commit()
    finally:
        conn.close()

    return thread, True


async def close_ping_thread(ping_id: int, user_id: int, reason: Optional[str] = None) -> None:
    conn = ensure_db_ready()
    try:
        thread_id = get_ping_thread_id(conn, ping_id, user_id)
    finally:
        conn.close()

    if not thread_id:
        return

    thread = await resolve_thread(thread_id)
    delete_reason = reason or "Resource submission received"
    if thread is not None:
        try:
            await thread.delete(reason=delete_reason)
        except discord.Forbidden:
            try:
                await thread.edit(archived=True, locked=True, reason=delete_reason)
            except Exception as e:
                logger.warning(
                    "Не удалось удалить или заархивировать ветку %s для ping %s/%s: %s",
                    thread_id,
                    ping_id,
                    user_id,
                    e,
                )
        except discord.NotFound:
            pass
        except Exception as e:
            logger.exception(
                "Ошибка при удалении ветки %s для ping %s/%s: %s",
                thread_id,
                ping_id,
                user_id,
                e,
            )

    conn = ensure_db_ready()
    try:
        delete_ping_thread(conn, ping_id, user_id)
        conn.commit()
    finally:
        conn.close()


async def update_ping_message_embed(
    message: Optional[discord.Message],
    total_users: int,
    *,
    submitted_pos: Optional[int] = None,
    total_pos: Optional[int] = None,
    view: Optional[discord.ui.View] = None,
):
    if not message:
        return
    embeds = list(message.embeds)
    if not embeds:
        return
    embed = discord.Embed.from_dict(embeds[0].to_dict())
    field_index = None
    for idx, field in enumerate(embed.fields):
        if field.name.lower().startswith("отметил"):
            field_index = idx
            break
    value_lines = [f"**{total_users}**"]
    if total_pos is not None and total_pos > 0:
        covered = submitted_pos or 0
        value_lines.append(f"POS покрыто: **{covered}/{total_pos}**")
    value_text = "\n".join(value_lines)
    if field_index is not None:
        field_name = embed.fields[field_index].name
        embed.set_field_at(field_index, name=field_name, value=value_text, inline=False)
    else:
        embed.add_field(name="Отметились", value=value_text, inline=False)
    embed.timestamp = datetime.now(timezone.utc)
    kwargs = {"embed": embed}
    if view is not None:
        kwargs["view"] = view
    await message.edit(**kwargs)


async def update_ping_message_from_db(ping_id: int):
    conn = ensure_db_ready()
    channel_id = None
    message_id = None
    total_users = 0
    submitted_pos = 0
    total_pos = 0
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT guild_id, channel_id, message_id FROM resource_ping WHERE id=?",
            (ping_id,),
        )
        row = cur.fetchone()
        if not row:
            return
        guild_id = int(row["guild_id"])
        channel_id = int(row["channel_id"])
        message_id = int(row["message_id"])
        total_users, submitted_pos, total_pos = _compute_ping_submission_totals(cur, ping_id, guild_id)
    finally:
        conn.close()

    channel = bot.get_channel(channel_id) if channel_id is not None else None
    if channel is None and channel_id is not None:
        try:
            channel = await bot.fetch_channel(channel_id)
        except Exception as e:
            logger.warning(
                "Не удалось получить канал %s для обновления напоминания %s: %s",
                channel_id,
                ping_id,
                e,
            )
            return

    if channel is None or not hasattr(channel, "fetch_message"):
        return

    try:
        message = await channel.fetch_message(message_id)
    except Exception as e:
        logger.warning(
            "Не удалось получить сообщение %s в канале %s для напоминания %s: %s",
            message_id,
            channel_id,
            ping_id,
            e,
        )
        return

    await update_ping_message_embed(
        message,
        total_users,
        submitted_pos=submitted_pos,
        total_pos=total_pos,
    )


class ResourceSubmitButton(discord.ui.Button):
    def __init__(self, ping_id: int):
        super().__init__(
            label="Сдал",
            style=discord.ButtonStyle.success,
            custom_id=f"resource_ping_submit:{ping_id}",
        )
        self.ping_id = ping_id

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.bot:
            return
        await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

        conn = ensure_db_ready()
        already = False
        total_users = 0
        user_pos_count = 0
        submitted_pos = 0
        total_pos = 0
        try:
            cur = conn.cursor()
            (
                already,
                total_users,
                _guild_id,
                user_pos_count,
                submitted_pos,
                total_pos,
            ) = record_ping_submission(cur, self.ping_id, interaction.user.id)
            conn.commit()
        except PingNotFoundError:
            await interaction.followup.send("Напоминание больше не доступно.", ephemeral=should_use_ephemeral(interaction))
            return
        except PingInactiveError:
            await interaction.followup.send("Это напоминание уже закрыто.", ephemeral=should_use_ephemeral(interaction))
            return
        except Exception as e:
            logger.exception("resource_ping submit error: %s", e)
            await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
            return
        finally:
            conn.close()

        try:
            await update_ping_message_embed(
                interaction.message,
                total_users,
                submitted_pos=submitted_pos,
                total_pos=total_pos,
                view=self.view,
            )
        except Exception as e:
            logger.exception("Не удалось обновить сообщение напоминания %s: %s", self.ping_id, e)

        try:
            await close_ping_thread(
                self.ping_id,
                interaction.user.id,
                reason="Пользователь отметил сдачу через кнопку",
            )
        except Exception as e:
            logger.exception(
                "Не удалось закрыть ветку для ping %s пользователя %s: %s",
                self.ping_id,
                interaction.user.id,
                e,
            )

        if already:
            info = "ℹ️ Ты уже отмечался, время обновлено."
        else:
            if user_pos_count > 1:
                info = f"✅ Отметил сдачу по {user_pos_count} POS. Спасибо!"
            elif user_pos_count == 1:
                info = "✅ Отметил сдачу. Спасибо!"
            else:
                info = "✅ Отметил сдачу. Спасибо!"
        if total_pos > 0:
            info += f"\nПокрытие POS по напоминанию: {submitted_pos}/{total_pos}."
        await interaction.followup.send(info, ephemeral=should_use_ephemeral(interaction))

class ResourcePingView(discord.ui.View):
    def __init__(self, ping_id: int):
        super().__init__(timeout=None)
        self.ping_id = ping_id
        self.add_item(ResourceSubmitButton(ping_id))

# ==================== ПРАВА ====================
def get_pos_owner(conn: sqlite3.Connection, pos_id: int) -> Optional[int]:
    cur = conn.cursor()
    cur.execute("SELECT owner_user_id FROM pos WHERE id=?", (pos_id,))
    row = cur.fetchone()
    return int(row["owner_user_id"]) if row else None

def ensure_owner_or_admin(conn: sqlite3.Connection, interaction: discord.Interaction, pos_id: int) -> bool:
    if is_admin_user(interaction):
        return True
    owner_id = get_pos_owner(conn, pos_id)
    return owner_id is not None and owner_id == interaction.user.id

# ==================== DISCORD ====================
intents = discord.Intents.default()
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

# --- автодополнение систем ---
async def system_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    conn = ensure_db_ready()
    try:
        items = get_distinct_systems(conn)
    finally:
        conn.close()
    q = (current or "").lower()
    if q:
        items = [s for s in items if q in s.lower()]
    return [app_commands.Choice(name=s, value=s) for s in items[:25]]

# --- автодополнение ресурсов (объединённое) ---
def resource_autocomplete_choices(conn: sqlite3.Connection, guild_id: Optional[int] = None) -> List[str]:
    cur = conn.cursor()
    seen, out = set(), []

    def add(name: Optional[str]):
        if not name: return
        n = clean_resource_name(name)
        if not n: return
        k = n.lower()
        if k not in seen:
            seen.add(k); out.append(n)

    # 1) глобальные из planet_resources
    cur.execute("""SELECT resource FROM planet_resources GROUP BY resource""")
    for r in cur.fetchall(): add(r["resource"])
    # 2) цели этого сервера
    if guild_id is not None:
        cur.execute("""SELECT resource FROM guild_needs_row WHERE guild_id=? GROUP BY resource""", (guild_id,))
        for r in cur.fetchall(): add(r["resource"])
    # 3) склад этого сервера
    if guild_id is not None:
        cur.execute("""SELECT resource FROM guild_have WHERE guild_id=? GROUP BY resource""", (guild_id,))
        for r in cur.fetchall(): add(r["resource"])
    # 4) заданные цены
    cur.execute("""SELECT resource FROM isk_price GROUP BY resource""")
    for r in cur.fetchall(): add(r["resource"])

    out.sort()
    return out[:2000]

async def resource_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    conn = ensure_db_ready()
    try:
        gid = interaction.guild.id if interaction.guild else None
        items = resource_autocomplete_choices(conn, guild_id=gid)
    finally:
        conn.close()
    q = clean_resource_name(current or "")
    ql = q.lower()
    filtered = [x for x in items if ql in x.lower()] if q else items
    if q and all(ql != x.lower() for x in filtered):
        filtered = [q] + filtered
    return [app_commands.Choice(name=r, value=r) for r in filtered[:25]]

# --- настройки значений по умолчанию для POS ---
posdefaults_group = app_commands.Group(
    name="posdefaults",
    description="Управление значениями по умолчанию для /addpos",
)

def _validate_default_bounds(value: int, kind: str) -> Optional[str]:
    if kind == "slots" and not (1 <= value <= 20):
        return "`slots` должен быть в диапазоне 1..20."
    if kind == "drills" and not (1 <= value <= 50):
        return "`drills` должен быть в диапазоне 1..50."
    return None

@posdefaults_group.command(name="setguild", description="Установить значения по умолчанию для всего сервера.")
@app_commands.describe(slots="Количество планет-слотов", drills="Количество буров на планету")
async def posdefaults_setguild(interaction: discord.Interaction, slots: int, drills: int):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))
    if not is_admin_user(interaction):
        await interaction.followup.send("⛔ Команда доступна только администраторам сервера.", ephemeral=should_use_ephemeral(interaction))
        return

    err = _validate_default_bounds(int(slots), "slots") or _validate_default_bounds(int(drills), "drills")
    if err:
        await interaction.followup.send(err, ephemeral=should_use_ephemeral(interaction))
        return

    conn = ensure_db_ready()
    try:
        set_pos_defaults(conn, guild.id, int(slots), int(drills), user_id=None)
        report = build_pos_defaults_report(conn, guild.id, interaction.user.id)
        await interaction.followup.send(
            "✅ Значения по умолчанию для сервера обновлены.\n" + report,
            ephemeral=should_use_ephemeral(interaction),
        )
    except Exception as e:
        logger.exception("posdefaults_setguild error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@posdefaults_group.command(name="set", description="Установить личные значения по умолчанию.")
@app_commands.describe(slots="Количество планет-слотов", drills="Количество буров на планету")
async def posdefaults_setuser(interaction: discord.Interaction, slots: int, drills: int):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

    err = _validate_default_bounds(int(slots), "slots") or _validate_default_bounds(int(drills), "drills")
    if err:
        await interaction.followup.send(err, ephemeral=should_use_ephemeral(interaction))
        return

    conn = ensure_db_ready()
    try:
        set_pos_defaults(conn, guild.id, int(slots), int(drills), user_id=interaction.user.id)
        report = build_pos_defaults_report(conn, guild.id, interaction.user.id)
        await interaction.followup.send(
            "✅ Личные значения по умолчанию сохранены.\n" + report,
            ephemeral=should_use_ephemeral(interaction),
        )
    except Exception as e:
        logger.exception("posdefaults_setuser error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@posdefaults_group.command(name="clear", description="Очистить личные значения и использовать серверные/ENV.")
async def posdefaults_clear(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

    conn = ensure_db_ready()
    try:
        removed = clear_user_pos_defaults(conn, guild.id, interaction.user.id)
        report = build_pos_defaults_report(conn, guild.id, interaction.user.id)
        head = "🧹 Личные значения удалены." if removed else "ℹ️ Личные значения не были заданы."
        await interaction.followup.send(head + "\n" + report, ephemeral=should_use_ephemeral(interaction))
    except Exception as e:
        logger.exception("posdefaults_clear error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@posdefaults_group.command(name="show", description="Показать активные значения по умолчанию для /addpos.")
async def posdefaults_show(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

    conn = ensure_db_ready()
    try:
        report = build_pos_defaults_report(conn, guild.id, interaction.user.id)
        await interaction.followup.send(report, ephemeral=should_use_ephemeral(interaction))
    except Exception as e:
        logger.exception("posdefaults_show error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

tree.add_command(posdefaults_group)

resping_group = app_commands.Group(
    name="resping",
    description="Напоминания о сдаче ресурсов.",
)

@resping_group.command(name="ping", description="Создать напоминание о сдаче ресурсов с кнопкой 'Сдал'.")
@app_commands.describe(
    text="Текст напоминания",
    role="Роль для упоминания (опционально)",
    mention_everyone="Упомянуть @everyone вместе с сообщением",
)
async def resping_ping(
    interaction: discord.Interaction,
    text: str,
    role: Optional[discord.Role] = None,
    mention_everyone: bool = False,
):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return
    if not is_admin_user(interaction):
        await interaction.response.send_message("⛔ Команда доступна только администраторам сервера.", ephemeral=should_use_ephemeral(interaction))
        return
    channel = interaction.channel
    if channel is None or not hasattr(channel, "send"):
        await interaction.response.send_message("Не удалось определить канал для отправки сообщения.", ephemeral=should_use_ephemeral(interaction))
        return

    bot_user = interaction.client.user if interaction.client else None
    bot_member: Optional[discord.Member] = None
    if guild:
        possible_member = getattr(guild, "me", None)
        if isinstance(possible_member, discord.Member):
            bot_member = possible_member
        elif bot_user:
            bot_member = guild.get_member(bot_user.id)

    if bot_member is None and bot_user and guild:
        try:
            bot_member = await guild.fetch_member(bot_user.id)
        except discord.HTTPException as exc:
            logger.warning(
                "resping_ping: failed to fetch bot member in guild %s (%s): %s", guild.id, guild.name, exc
            )
            bot_member = None

    fallback_identity: Optional[Snowflake] = None
    if bot_member is not None:
        fallback_identity = bot_member
    elif bot_user is not None:
        fallback_identity = discord.Object(id=bot_user.id)

    if fallback_identity is None:
        await interaction.response.send_message(
            "Не удалось определить права бота на сервере. Проверьте, что бот добавлен на сервер.",
            ephemeral=should_use_ephemeral(interaction),
        )
        return

    channel_permissions: Optional[discord.Permissions] = None
    if isinstance(bot_member, discord.Member) and hasattr(channel, "permissions_for"):
        channel_permissions = channel.permissions_for(bot_member)
    else:
        logger.warning(
            "resping_ping: не удалось определить права бота (guild_id=%s, channel_id=%s), пропускаю предварительную проверку.",
            getattr(guild, "id", "?"),
            getattr(channel, "id", "?"),
        )

    missing_permissions: List[str] = []
    if channel_permissions is not None:
        if not channel_permissions.view_channel:
            missing_permissions.append("просматривать канал")
        if isinstance(channel, discord.Thread):
            if not channel_permissions.send_messages_in_threads:
                missing_permissions.append("отправлять сообщения в ветке")
        elif not channel_permissions.send_messages:
            missing_permissions.append("отправлять сообщения")
        if not channel_permissions.embed_links:
            missing_permissions.append("вставлять embed-сообщения")
        if mention_everyone and not channel_permissions.mention_everyone:
            missing_permissions.append("упоминать @everyone")
        if role is not None and not channel_permissions.mention_everyone:
            missing_permissions.append("упоминать роли")

    if missing_permissions:
        missing_text = ", ".join(sorted(set(missing_permissions)))
        await interaction.response.send_message(
            f"⛔ У бота нет необходимых прав для отправки напоминания в этом канале: {missing_text}.",
            ephemeral=should_use_ephemeral(interaction),
        )
        return

    body = (text or "").strip()
    if not body:
        await interaction.response.send_message("Текст напоминания не должен быть пустым.", ephemeral=should_use_ephemeral(interaction))
        return
    if len(body) > 2000:
        await interaction.response.send_message("Текст напоминания слишком длинный (максимум 2000 символов).", ephemeral=should_use_ephemeral(interaction))
        return

    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO resource_ping (guild_id, channel_id, message_id, author_user_id, content, role_id, created_at, active)
            VALUES (?, ?, 0, ?, ?, ?, ?, 1)
            """,
            (
                guild.id,
                getattr(channel, "id", guild.id),
                interaction.user.id,
                body,
                role.id if role else None,
                now_utc_iso(),
            ),
        )
        ping_id = int(cur.lastrowid)
        conn.commit()
    except Exception as e:
        logger.exception("resping_ping insert error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
        return
    finally:
        conn.close()

    view = ResourcePingView(ping_id)
    embed = discord.Embed(
        title="Сдача ресурсов",
        description=body,
        color=discord.Color.blurple(),
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Отметились", value="**0**", inline=False)
    display_name = getattr(interaction.user, "display_name", interaction.user.name)
    embed.set_footer(text=f"ID: {ping_id} • Создал {display_name}")

    mention_parts: List[str] = []
    if mention_everyone:
        mention_parts.append("@everyone")
    if role is not None:
        mention_parts.append(role.mention)
    mention_text = " ".join(mention_parts) if mention_parts else None
    allowed_mentions = discord.AllowedMentions(everyone=mention_everyone, roles=role is not None, users=False)

    try:
        message = await channel.send(
            content=mention_text,
            embed=embed,
            view=view,
            allowed_mentions=allowed_mentions,
        )
    except Exception as e:
        logger.exception("resping_ping send error: %s", e)
        cleanup_conn = ensure_db_ready()
        try:
            cur = cleanup_conn.cursor()
            cur.execute("DELETE FROM resource_ping WHERE id=?", (ping_id,))
            cleanup_conn.commit()
        finally:
            cleanup_conn.close()
        await interaction.followup.send(f"Не удалось отправить сообщение: {e}", ephemeral=should_use_ephemeral(interaction))
        return

    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE resource_ping SET message_id=?, channel_id=? WHERE id=?",
            (message.id, getattr(message.channel, "id", getattr(channel, "id", 0)), ping_id),
        )
        conn.commit()
    except Exception as e:
        logger.exception("resping_ping update error: %s", e)
    finally:
        conn.close()

    conn_notify = ensure_db_ready()
    try:
        targets = get_guild_resource_producers(conn_notify, guild.id)
        user_payloads: List[Tuple[int, Dict[str, Dict[str, object]]]] = []
        for user_id in targets:
            try:
                assignments = get_user_resource_assignments(conn_notify, guild.id, user_id)
            except Exception:
                assignments = {}
            user_payloads.append((user_id, assignments))
    finally:
        conn_notify.close()

    notified = 0
    failed: List[int] = []
    for user_id, assignments in user_payloads:
        user = interaction.client.get_user(user_id) if interaction.client else None
        if user is None:
            try:
                user = await interaction.client.fetch_user(user_id) if interaction.client else None
            except Exception as exc:
                logger.warning("Не удалось получить пользователя %s: %s", user_id, exc)
                user = None
        if user is None:
            failed.append(user_id)
            continue

        assignment_lines: List[str] = []
        for info in assignments.values():
            rate_total = float(info.get("rate_total", 0.0))
            rate_txt = f" ≈ {rate_total:,.0f}/ч".replace(",", " ") if rate_total > 0 else ""
            assignment_lines.append(f"- {info['name']}{rate_txt}")

        lines = [
            f"👋 Привет! Администратор сервера **{guild.name}** создал напоминание о сдаче ресурсов.",
            f"ID напоминания: #{ping_id}.",
        ]
        if body:
            short = body if len(body) <= 200 else body[:197] + "…"
            lines.append(f"Текст: {short}")
        lines.append("")
        if assignment_lines:
            lines.append("Твои ресурсы:")
            lines.extend(assignment_lines)
        else:
            lines.append("Тебе пока не назначены ресурсы — отметь сдачу при их наличии.")
        lines.extend(
            [
                "",
                "Нажми кнопку ниже, чтобы отметить сдачу.",
                "Можно также использовать команду `/resping submit` на сервере.",
            ]
        )

        try:
            await user.send("\n".join(lines), view=ResourcePingView(ping_id))
            notified += 1
        except discord.Forbidden:
            failed.append(user_id)
        except Exception as exc:
            logger.exception("Не удалось отправить DM пользователю %s: %s", user_id, exc)
            failed.append(user_id)

    summary_lines = [f"✅ Напоминание #{ping_id} создано. {message.jump_url}"]
    total_targets = len(user_payloads)
    if total_targets:
        summary_lines.append(
            f"DM отправлено: **{notified}** из **{total_targets}**."
        )
    if failed:
        failed_mentions = ", ".join(f"<@{uid}>" for uid in failed)
        summary_lines.append("Не удалось отправить DM: " + failed_mentions)

    await interaction.followup.send("\n".join(summary_lines), ephemeral=should_use_ephemeral(interaction))


def resolve_active_ping_for_user(
    conn: sqlite3.Connection,
    guild_id: int,
    user_id: int,
    *,
    thread_id: Optional[int] = None,
) -> Tuple[Optional[int], List[int]]:
    """Определяет актуальное напоминание для пользователя.

    Возвращает (ping_id, ambiguous_ids). Если ping_id отсутствует и список ambiguous
    непустой — требуется уточнение (несколько напоминаний без отметки).
    """
    cur = conn.cursor()

    if thread_id is not None:
        cur.execute(
            """
            SELECT rpt.ping_id
            FROM resource_ping_thread rpt
            JOIN resource_ping p ON p.id = rpt.ping_id
            WHERE rpt.thread_id=? AND rpt.user_id=? AND p.guild_id=? AND p.active=1
            """,
            (int(thread_id), int(user_id), int(guild_id)),
        )
        row = cur.fetchone()
        if row:
            return int(row["ping_id"]), []

    cur.execute(
        """
        SELECT p.id,
               CASE WHEN s.user_id IS NULL THEN 0 ELSE 1 END AS submitted
        FROM resource_ping p
        LEFT JOIN resource_ping_submission s
          ON s.ping_id = p.id AND s.user_id = ?
        WHERE p.guild_id=? AND p.active=1
        ORDER BY p.created_at DESC, p.id DESC
        """,
        (int(user_id), int(guild_id)),
    )
    rows = cur.fetchall()
    if not rows:
        return None, []

    pending = [int(r["id"]) for r in rows if not bool(r["submitted"])]
    if len(pending) == 1:
        return pending[0], []
    if len(pending) > 1:
        return None, pending

    return int(rows[0]["id"]), []


@resping_group.command(name="submit", description="Отправить список сданных ресурсов по напоминанию.")
@app_commands.describe(
    data="Таблица (TSV/CSV или одной строкой): [ID] Название Количество [Оценка]",
)
async def resping_submit(interaction: discord.Interaction, data: Optional[str] = None):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return

    channel = interaction.channel
    thread_id: Optional[int] = channel.id if isinstance(channel, discord.Thread) else None

    payload = (data or "").strip()
    had_payload = bool(payload)

    conn = ensure_db_ready()
    aggregated: OrderedDict[str, Dict[str, object]] = OrderedDict()
    assignments: Dict[str, Dict[str, object]] = {}
    ping_content = ""
    already = False
    total_users = 0
    user_pos_count = 0
    submitted_pos = 0
    total_pos = 0
    submission_saved = False
    try:
        ping_id, ambiguous = resolve_active_ping_for_user(
            conn,
            guild.id,
            interaction.user.id,
            thread_id=thread_id,
        )
        if ping_id is None:
            if ambiguous:
                listed = ", ".join(f"#{pid}" for pid in ambiguous[:5])
                if len(ambiguous) > 5:
                    listed += f" и ещё {len(ambiguous) - 5}"
                await interaction.response.send_message(
                    "На сервере есть несколько активных напоминаний, по которым ты ещё не отметился: "
                    + listed
                    + ". Запусти команду из ветки нужного напоминания или отметься кнопкой \"Сдал\".",
                    ephemeral=should_use_ephemeral(interaction),
                )
            else:
                await interaction.response.send_message(
                    "Нет активных напоминаний, к которым можно отправить отчёт.",
                    ephemeral=should_use_ephemeral(interaction),
                )
            return

        if payload:
            items = parse_have_table(payload)
            if not items:
                await interaction.response.send_message(
                    "Не удалось распарсить данные. Проверь колонки/формат.",
                    ephemeral=should_use_ephemeral(interaction),
                )
                return

            invalid_amounts: List[str] = []
            for name, amount, unit_price in items:
                resource = clean_resource_name(name)
                if not resource or amount is None:
                    continue
                try:
                    amount_val = float(amount)
                except Exception:
                    continue
                if amount_val < 0:
                    invalid_amounts.append(resource)
                    continue
                key = resource.lower()
                entry = aggregated.get(key)
                if not entry:
                    entry = {"name": resource, "amount": 0.0, "prices": []}
                    aggregated[key] = entry
                entry["amount"] += amount_val
                if unit_price is not None:
                    try:
                        entry.setdefault("prices", []).append(float(unit_price))
                    except Exception:
                        pass

            if invalid_amounts:
                pretty = ", ".join(sorted(set(invalid_amounts)))
                await interaction.response.send_message(
                    f"Количество не может быть отрицательным. Проверь: {pretty}.",
                    ephemeral=should_use_ephemeral(interaction),
                )
                return

        for entry in aggregated.values():
            prices = entry.get("prices", [])
            entry["unit_price"] = sum(prices) / len(prices) if prices else None
            entry.pop("prices", None)

        await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

        cur = conn.cursor()
        cur.execute(
            "SELECT guild_id, content FROM resource_ping WHERE id=?",
            (ping_id,),
        )
        ping_row = cur.fetchone()
        if not ping_row:
            raise PingNotFoundError(f"Ping {ping_id} not found")
        if int(ping_row["guild_id"]) != guild.id:
            await interaction.followup.send("Это напоминание относится к другому серверу.", ephemeral=should_use_ephemeral(interaction))
            return

        ping_content = ping_row["content"] or ""

        assignments = get_user_resource_assignments(conn, guild.id, interaction.user.id)
        if not assignments:
            await interaction.followup.send(
                "У тебя нет назначенных планет на этом сервере.",
                ephemeral=should_use_ephemeral(interaction),
            )
            return

        unexpected = [entry["name"] for key, entry in aggregated.items() if key not in assignments]
        if unexpected:
            available = ", ".join(info["name"] for info in assignments.values()) or "—"
            await interaction.followup.send(
                "Эти ресурсы не закреплены за тобой: "
                + ", ".join(unexpected)
                + f". Доступные: {available}.",
                ephemeral=should_use_ephemeral(interaction),
            )
            return

        ts = now_utc_iso()
        cur.execute(
            "DELETE FROM resource_ping_submission_item WHERE ping_id=? AND user_id=?",
            (ping_id, interaction.user.id),
        )
        for entry in aggregated.values():
            cur.execute(
                """
                INSERT INTO resource_ping_submission_item (
                    ping_id, guild_id, user_id, resource, amount_units, unit_price, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ping_id, user_id, resource) DO UPDATE
                  SET amount_units=excluded.amount_units,
                      unit_price=excluded.unit_price,
                      submitted_at=excluded.submitted_at
                """,
                (
                    ping_id,
                    guild.id,
                    interaction.user.id,
                    entry["name"],
                    float(entry["amount"]),
                    float(entry["unit_price"]) if entry["unit_price"] is not None else None,
                    ts,
                ),
            )

        (
            already,
            total_users,
            _guild_id,
            user_pos_count,
            submitted_pos,
            total_pos,
        ) = record_ping_submission(cur, ping_id, interaction.user.id, submitted_at=ts)
        conn.commit()
        submission_saved = True
    except PingNotFoundError:
        await interaction.followup.send("Напоминание не найдено.", ephemeral=should_use_ephemeral(interaction))
        return
    except PingInactiveError:
        await interaction.followup.send("Это напоминание уже закрыто.", ephemeral=should_use_ephemeral(interaction))
        return
    except Exception as e:
        logger.exception("resping_submit error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
        return
    finally:
        conn.close()

    if not submission_saved:
        return

    try:
        await update_ping_message_from_db(ping_id)
    except Exception as e:
        logger.exception("Не удалось обновить сообщение напоминания %s: %s", ping_id, e)

    try:
        await close_ping_thread(
            ping_id,
            interaction.user.id,
            reason="Пользователь загрузил отчёт через /resping submit",
        )
    except Exception as e:
        logger.exception(
            "Не удалось закрыть ветку для ping %s пользователя %s после submit: %s",
            ping_id,
            interaction.user.id,
            e,
        )

    summary_lines = ["**Отчёт о сдаче ресурсов**"]
    if ping_content:
        short = ping_content if len(ping_content) <= 200 else ping_content[:197] + "…"
        summary_lines.append(f"Напоминание #{ping_id}: {short}")

    if aggregated:
        for key, entry in aggregated.items():
            info = assignments.get(key)
            expected = float(info["rate_total"]) * 24 if info else 0.0
            diff = entry["amount"] - expected
            ratio = (entry["amount"] / expected) if expected > 0 else None
            line = f"- {entry['name']}: **{entry['amount']:,.0f}** ед".replace(",", " ")
            if expected > 0:
                line += (
                    f" (≈ {expected:,.0f} за 24ч; Δ {diff:+,.0f} ед"
                ).replace(",", " ")
                if ratio is not None:
                    line += f", {ratio*100:.0f}%"
                line += ")"
            summary_lines.append(line)

        missing = [info["name"] for key, info in assignments.items() if key not in aggregated]
        if missing:
            summary_lines.append("⚠️ В отчёт не попали ресурсы: " + ", ".join(missing))
    else:
        if had_payload:
            summary_lines.append("⚠️ Не удалось распознать ресурсы в таблице — отчёт сохранён без них.")
        else:
            summary_lines.append("Ресурсы не указаны — отчёт сохранён без таблицы.")

    if user_pos_count > 1:
        summary_lines.append(f"У тебя **{user_pos_count}** POS — отчёт отмечен для всех.")
    elif user_pos_count == 1:
        summary_lines.append("Твой POS учтён.")

    if total_pos > 0:
        summary_lines.append(
            f"Покрытие POS по напоминанию: **{submitted_pos}/{total_pos}**."
        )

    summary_lines.append("ℹ️ " + ("Время сдачи обновлено." if already else "Сдача сохранена."))

    await interaction.followup.send("\n".join(summary_lines), ephemeral=should_use_ephemeral(interaction))


@resping_group.command(name="stats", description="Показать статистику отметок за период.")
@app_commands.describe(
    days="Сколько дней учитывать (1-365, по умолчанию 7)",
    ping_id="Опционально: ID конкретного напоминания",
)
async def resping_stats(
    interaction: discord.Interaction,
    days: Optional[int] = 7,
    ping_id: Optional[int] = None,
):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return

    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

    days_val = int(days or 7)
    if days_val < 1:
        days_val = 1
    if days_val > 365:
        days_val = 365

    now_dt = datetime.now(timezone.utc)
    since_dt = now_dt - timedelta(days=days_val)
    since_iso = since_dt.replace(microsecond=0).isoformat()

    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        params: List[object] = [guild.id, since_iso]
        where_extra = ""
        ping_info = None
        if ping_id is not None:
            where_extra = " AND s.ping_id=?"
            params.append(int(ping_id))
            cur.execute(
                "SELECT id, content, created_at FROM resource_ping WHERE id=? AND guild_id=?",
                (int(ping_id), guild.id),
            )
            ping_info = cur.fetchone()
            if not ping_info:
                await interaction.followup.send("Напоминание с таким ID не найдено на этом сервере.", ephemeral=should_use_ephemeral(interaction))
                return

        cur.execute(
            f"""
            SELECT s.user_id, COUNT(*) AS cnt, MIN(s.submitted_at) AS first_at, MAX(s.submitted_at) AS last_at
            FROM resource_ping_submission s
            JOIN resource_ping p ON p.id = s.ping_id
            WHERE p.guild_id=? AND s.submitted_at>=?{where_extra}
            GROUP BY s.user_id
            ORDER BY cnt DESC, last_at DESC
            """,
            params,
        )
        user_rows = cur.fetchall()

        cur.execute(
            f"""
            SELECT s.ping_id, COUNT(*) AS cnt, MAX(s.submitted_at) AS last_at
            FROM resource_ping_submission s
            JOIN resource_ping p ON p.id = s.ping_id
            WHERE p.guild_id=? AND s.submitted_at>=?{where_extra}
            GROUP BY s.ping_id
            ORDER BY last_at DESC
            """,
            params,
        )
        ping_rows = cur.fetchall()
    except Exception as e:
        logger.exception("resping_stats error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
        return
    finally:
        conn.close()

    total_submissions = sum(int(r["cnt"] or 0) for r in ping_rows)
    unique_users = len(user_rows)

    header_lines = [
        f"Период: {format_dt(since_dt)} — {format_dt(now_dt)}",
        f"Всего отметок: **{total_submissions}**",
        f"Участников отметилось: **{unique_users}**",
    ]

    if ping_info:
        header_lines.append(f"Напоминание #{ping_info['id']} создано {format_ts(ping_info['created_at'])}")
        content_text = ping_info["content"] or ""
        if content_text:
            short = content_text if len(content_text) <= 200 else content_text[:197] + "…"
            header_lines.append(f"Текст: {short}")

    embed = discord.Embed(
        title="Статистика сдачи ресурсов",
        description="\n".join(header_lines),
        color=discord.Color.gold(),
        timestamp=now_dt,
    )

    if user_rows:
        user_lines = []
        for row in user_rows[:20]:
            uid = int(row["user_id"])
            member = guild.get_member(uid)
            mention = member.mention if member else f"<@{uid}>"
            count = int(row["cnt"] or 0)
            last_at = format_ts(row["last_at"])
            user_lines.append(f"{mention} — **{count}** (последняя отметка: {last_at})")
        embed.add_field(name="Пользователи", value="\n".join(user_lines), inline=False)
    else:
        embed.add_field(name="Пользователи", value="Нет отметок за выбранный период.", inline=False)

    if ping_rows:
        ping_lines = []
        for row in ping_rows[:10]:
            pid = int(row["ping_id"])
            count = int(row["cnt"] or 0)
            last_at = format_ts(row["last_at"])
            ping_lines.append(f"#{pid} — **{count}** отметок (последняя: {last_at})")
        embed.add_field(name="Напоминания", value="\n".join(ping_lines), inline=False)

    await interaction.followup.send(embed=embed, ephemeral=should_use_ephemeral(interaction))


@resping_group.command(name="audit", description="Анализ сдачи ресурсов между отметками.")
@app_commands.describe(
    days="Сколько дней учитывать (1-90, по умолчанию 7)",
    tolerance="Допустимая погрешность в процентах (0-100, по умолчанию 15)",
)
async def resping_audit(
    interaction: discord.Interaction,
    days: Optional[int] = 7,
    tolerance: Optional[float] = 15.0,
):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return
    if not is_admin_user(interaction):
        await interaction.response.send_message(
            "⛔ Команда доступна только администраторам сервера.", ephemeral=should_use_ephemeral(interaction)
        )
        return

    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

    days_val = int(days or 7)
    if days_val < 1:
        days_val = 1
    if days_val > 90:
        days_val = 90

    tol_val = float(tolerance if tolerance is not None else 15.0)
    if tol_val < 0:
        tol_val = 0.0
    if tol_val > 100:
        tol_val = 100.0
    tol_fraction = tol_val / 100.0

    now_dt = datetime.now(timezone.utc)
    since_dt = now_dt - timedelta(days=days_val)
    since_iso = since_dt.replace(microsecond=0).isoformat()

    conn = ensure_db_ready()
    try:
        stats, intervals, extras = compute_resping_delivery_audit(conn, guild.id, since_iso)
    except Exception as e:
        logger.exception("resping_audit error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
        return
    finally:
        conn.close()

    if not stats:
        await interaction.followup.send(
            "Нет данных о сдаче ресурсов за выбранный период.", ephemeral=should_use_ephemeral(interaction)
        )
        return

    user_entries = []
    for user_id, resources in stats.items():
        total_expected = sum(float(info.get("expected") or 0.0) for info in resources.values())
        if total_expected <= 0:
            continue
        total_actual = sum(float(info.get("actual") or 0.0) for info in resources.values())
        ratio = (total_actual / total_expected) if total_expected > 0 else 0.0
        user_entries.append((ratio, user_id, resources))

    if not user_entries:
        await interaction.followup.send(
            "Недостаточно данных о сдаче ресурсов за выбранный период.", ephemeral=should_use_ephemeral(interaction)
        )
        return

    user_entries.sort(key=lambda item: item[0])

    lines: List[str] = [
        "**Аудит сдачи ресурсов**",
        f"Период: {format_dt(since_dt)} — {format_dt(now_dt)}",
        f"Допустимая погрешность: ±{tol_val:.0f}%",
        "Рассчитано по интервалам между отметками \"Сдал\" (кнопка или /resping submit).",
        "",
    ]

    for ratio, user_id, resources in user_entries:
        member = guild.get_member(user_id)
        if member:
            user_name = f"{member.display_name} ({member.mention})"
        else:
            user_name = f"<@{user_id}>"

        total_expected = sum(float(info.get("expected") or 0.0) for info in resources.values())
        total_actual = sum(float(info.get("actual") or 0.0) for info in resources.values())
        total_ratio = (total_actual / total_expected) if total_expected > 0 else 0.0
        total_delta = total_actual - total_expected
        intervals_count = intervals.get(user_id, 0)

        header = (
            f"{user_name} — интервалов: {intervals_count}, всего {total_actual:,.0f}/{total_expected:,.0f} ед"
            f" ({total_ratio*100:.0f}%, Δ {total_delta:+,.0f})"
        ).replace(",", " ")
        lines.append(header)

        resource_entries = list(resources.values())
        resource_entries.sort(
            key=lambda info: (
                (float(info.get("actual") or 0.0) / float(info.get("expected") or 1.0))
                if float(info.get("expected") or 0.0) > 0
                else 1.0
            )
        )

        for entry in resource_entries:
            expected = float(entry.get("expected") or 0.0)
            if expected <= 0:
                continue
            actual = float(entry.get("actual") or 0.0)
            delta = actual - expected
            ratio_val = (actual / expected) if expected > 0 else 0.0
            if actual <= 0:
                status = "❌"
            elif ratio_val < (1 - tol_fraction):
                status = "⚠️"
            else:
                status = "✅"
            line = (
                f"  {status} {entry.get('name')}: {actual:,.0f}/{expected:,.0f} ед"
                f" ({ratio_val*100:.0f}%, Δ {delta:+,.0f})"
            ).replace(",", " ")
            lines.append(line)

        extra_map = extras.get(user_id, {})
        if extra_map:
            extra_parts = []
            for info in extra_map.values():
                amount = float(info.get("amount") or 0.0)
                if amount <= 0:
                    continue
                extra_parts.append(f"{info.get('name')}: {amount:,.0f} ед".replace(",", " "))
            if extra_parts:
                lines.append("  ➕ Дополнительно (не из назначенных): " + ", ".join(extra_parts))

        lines.append("")

    text = "\n".join(lines).strip()
    await send_long(
        interaction,
        text,
        ephemeral=should_use_ephemeral(interaction),
        title="Аудит сдачи ресурсов",
    )


tree.add_command(resping_group)

async def run_resource_reminder_cycle():
    now = datetime.now(timezone.utc)
    delay = timedelta(hours=max(0, RES_REMINDER_DELAY_HOURS))
    due: List[
        Tuple[int, int, int, int, int, str, datetime, Dict[str, Dict[str, object]]]
    ] = []

    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, guild_id, channel_id, message_id, content, created_at
            FROM resource_ping
            WHERE active=1
            """,
        )
        ping_rows = cur.fetchall()
        for row in ping_rows:
            ping_id = int(row["id"])
            guild_id = int(row["guild_id"])
            channel_id = int(row["channel_id"])
            message_id = int(row["message_id"])
            created_raw = row["created_at"]
            try:
                created_at = datetime.fromisoformat(created_raw) if created_raw else None
            except Exception:
                created_at = None
            if created_at is None:
                continue
            if delay.total_seconds() > 0 and now - created_at < delay:
                continue

            producers = get_guild_resource_producers(conn, guild_id)
            if not producers:
                continue

            for user_id in producers:
                cur.execute(
                    "SELECT 1 FROM resource_ping_submission WHERE ping_id=? AND user_id=?",
                    (ping_id, user_id),
                )
                if cur.fetchone():
                    continue

                cur.execute(
                    "SELECT last_reminded_at FROM resource_ping_reminder WHERE ping_id=? AND user_id=?",
                    (ping_id, user_id),
                )
                remind_row = cur.fetchone()
                last_rem: Optional[datetime] = None
                if remind_row and remind_row["last_reminded_at"]:
                    try:
                        last_rem = datetime.fromisoformat(remind_row["last_reminded_at"])
                    except Exception:
                        last_rem = None
                if last_rem and now - last_rem < delay:
                    continue

                assignments = get_user_resource_assignments(conn, guild_id, user_id)
                if not assignments:
                    continue

                due.append(
                    (
                        ping_id,
                        guild_id,
                        channel_id,
                        message_id,
                        user_id,
                        row["content"] or "",
                        created_at,
                        assignments,
                    )
                )
    except Exception as e:
        logger.exception("run_resource_reminder_cycle gather error: %s", e)
        return
    finally:
        conn.close()

    if not due:
        return

    for (
        ping_id,
        guild_id,
        channel_id,
        message_id,
        user_id,
        content,
        created_at,
        assignments,
    ) in due:
        guild_obj = bot.get_guild(guild_id)
        guild_name = guild_obj.name if guild_obj else f"ID {guild_id}"
        user = bot.get_user(user_id)
        if user is None:
            try:
                user = await bot.fetch_user(user_id)
            except Exception as e:
                logger.warning(
                    "Не удалось получить пользователя %s для напоминания %s: %s",
                    user_id,
                    ping_id,
                    e,
                )
                user = None

        channel = bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await bot.fetch_channel(channel_id)
            except Exception as e:
                logger.warning(
                    "Не удалось получить канал %s для напоминания %s: %s",
                    channel_id,
                    ping_id,
                    e,
                )
                channel = None
        if channel is None or not hasattr(channel, "fetch_message"):
            logger.warning(
                "Канал %s для напоминания %s недоступен или не поддерживает fetch_message",
                channel_id,
                ping_id,
            )
            continue

        try:
            message = await channel.fetch_message(message_id)
        except Exception as e:
            logger.warning(
                "Не удалось получить сообщение %s в канале %s для напоминания %s: %s",
                message_id,
                channel_id,
                ping_id,
                e,
            )
            continue

        assignment_lines = []
        for info in assignments.values():
            rate_total = float(info.get("rate_total", 0.0))
            rate_txt = f" ≈ {rate_total:,.0f}/ч".replace(",", " ") if rate_total > 0 else ""
            assignment_lines.append(f"- {info['name']}{rate_txt}")

        mention_text = getattr(user, "mention", None) or f"<@{user_id}>"
        lines = [
            f"{mention_text}, напоминание о сдаче ресурсов на сервере **{guild_name}**.",
            f"Пинг #{ping_id} создан {format_dt(created_at)}.",
        ]
        if content:
            short = content if len(content) <= 200 else content[:197] + "…"
            lines.append(f"Текст напоминания: {short}")
        if assignment_lines:
            lines.append("")
            lines.append("Твои текущие ресурсы:")
            lines.extend(assignment_lines)
        lines.extend(
            [
                "",
                "Отправь отчёт об отгрузке прямо в этой ветке и отметься под основным сообщением.",
                "Можно также использовать команду `/resping submit` на сервере.",
                "После отметки ветка будет удалена автоматически.",
            ]
        )

        thread, created = await get_or_create_ping_thread(ping_id, user_id, message, user)
        sent = False
        if thread is not None:
            try:
                await thread.send("\n".join(lines))
                sent = True
                logger.info(
                    "Отправлено напоминание пользователю %s по ping %s в ветке %s (created=%s)",
                    user_id,
                    ping_id,
                    thread.id,
                    created,
                )
            except discord.Forbidden:
                logger.info(
                    "Нет прав на отправку сообщений в ветку %s для напоминания %s",
                    getattr(thread, "id", "?"),
                    ping_id,
                )
            except Exception as e:
                logger.exception(
                    "Ошибка при отправке напоминания %s в ветку %s пользователю %s: %s",
                    ping_id,
                    getattr(thread, "id", "?"),
                    user_id,
                    e,
                )

        stamp = now_utc_iso()
        conn_upd = ensure_db_ready()
        try:
            cur_upd = conn_upd.cursor()
            cur_upd.execute(
                """
                INSERT INTO resource_ping_reminder (ping_id, user_id, last_reminded_at)
                VALUES (?, ?, ?)
                ON CONFLICT(ping_id, user_id) DO UPDATE
                  SET last_reminded_at=excluded.last_reminded_at
                """,
                (ping_id, user_id, stamp),
            )
            conn_upd.commit()
        except Exception as e:
            logger.exception(
                "Не удалось обновить время напоминания %s/%s: %s",
                ping_id,
                user_id,
                e,
            )
        finally:
            conn_upd.close()

        if not sent:
            logger.debug(
                "Напоминание %s пользователю %s не доставлено в ветку (см. логи выше)",
                ping_id,
                user_id,
            )


async def resource_reminder_loop():
    await bot.wait_until_ready()
    interval = max(60, RES_REMINDER_CHECK_SECONDS)
    while not bot.is_closed():
        try:
            await run_resource_reminder_cycle()
        except Exception as e:
            logger.exception("resource_reminder_loop error: %s", e)
        await asyncio.sleep(interval)


# ==================== КОМАНДЫ ====================
@tree.command(name="setneed", description="Задать/обновить целевой ОБЪЁМ (единиц) по ресурсу.")
@app_commands.describe(resource="Ресурс", amount="Целевой объём (units)")
@app_commands.autocomplete(resource=resource_autocomplete)
async def setneed_cmd(interaction: discord.Interaction, resource: str, amount: float):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    res = clean_resource_name(resource)
    if amount < 0:
        await interaction.response.send_message("amount должен быть >= 0.", ephemeral=should_use_ephemeral(interaction)); return
    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO guild_needs_row (guild_id, resource, amount_required, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(guild_id, resource) DO UPDATE
              SET amount_required=excluded.amount_required, updated_at=excluded.updated_at
        """, (guild.id, res, float(amount), now_utc_iso()))
        conn.commit()
        await interaction.response.send_message(
            f"Цель обновлена: **{res}** = **{amount:,.0f} ед**".replace(",", " "), ephemeral=False
        )
    except Exception as e:
        logger.exception("setneed error: %s", e)
        await interaction.response.send_message(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@tree.command(name="clearneeds", description="Удалить все цели (units) на этом сервере.")
async def clearneeds_cmd(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM guild_needs_row WHERE guild_id=?", (guild.id,))
        n = cur.rowcount or 0
        conn.commit()
        await interaction.response.send_message(f"Удалено целей: **{n}**.", ephemeral=False)
    except Exception as e:
        logger.exception("clearneeds error: %s", e)
        await interaction.response.send_message(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@tree.command(name="eta", description="ETA по всем целям (units) при текущей суммарной скорости (учитывает склад).")
@app_commands.describe(resource="Опционально: конкретный ресурс", limit="Макс. строк (по умолчанию 20)")
@app_commands.autocomplete(resource=resource_autocomplete)
async def eta_cmd(interaction: discord.Interaction, resource: Optional[str] = None, limit: Optional[int] = 20):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=False)
    conn = ensure_db_ready()
    try:
        needs = load_guild_needs(conn, guild.id)
        have  = load_guild_have(conn, guild.id)
        needs_left = subtract_amounts(needs, have)
        rph = active_production_rates(conn, guild.id)

        def fmt(amount_units: float, rate: float) -> Tuple[str, float]:
            if amount_units <= 0: return "0 ч", 0.0
            if rate <= 0: return "— нет добычи —", float("inf")
            hours = amount_units / rate
            days = hours / 24.0
            if days < 3: return f"{hours:,.1f} ч".replace(",", " "), days
            return f"{days:,.2f} дн".replace(",", " "), days

        lines: List[str] = []
        header = "⏱️ **ETA (учитывая склад) при текущей скорости**"
        if resource:
            res = clean_resource_name(resource)
            amount_total = needs.get(res)
            if amount_total is None:
                await interaction.followup.send(f"Для **{res}** цель не задана.", ephemeral=should_use_ephemeral(interaction)); return
            left = needs_left.get(res, 0.0)
            eta_txt, _ = fmt(left, rph.get(res, 0.0))
            lines.append(
                f"**{res}** — цель {amount_total:,.0f} — на складе {have.get(res,0.0):,.0f} — осталось {left:,.0f} — сейчас {rph.get(res,0.0):,.2f}/ч — ETA: **{eta_txt}**"
                .replace(",", " ")
            )
        else:
            rows=[]
            for res, amount_total in needs.items():
                left = needs_left.get(res, 0.0)
                txt, days = fmt(left, rph.get(res,0.0))
                rows.append((res, amount_total, have.get(res,0.0), left, rph.get(res,0.0), days, txt))
            rows.sort(key=lambda t: (t[5]==float('inf'), t[5], -t[4]), reverse=False)
            shown=0
            for res, amount_total, have_amt, left, rate, _days, txt in rows:
                if shown >= int(limit or 20): break
                lines.append(
                    f"**{res}** — цель {amount_total:,.0f} — склад {have_amt:,.0f} — осталось {left:,.0f} — сейчас {rate:,.2f}/ч — ETA: **{txt}**"
                    .replace(",", " ")
                )
                shown += 1
            if not lines: lines.append("_Цели не заданы. Используйте `/setneed`._")

        await send_long(interaction, header + "\n" + "\n".join(lines), ephemeral=False, title="ETA")
    except Exception as e:
        logger.exception("eta error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@tree.command(name="isk", description="Цены ресурсов (ISK/ед).")
@app_commands.describe(action="set/show/import", resource="Ресурс (для set/show)", price="Цена (для set)", csv_prices="CSV строки: resource,price")
@app_commands.choices(action=[
    app_commands.Choice(name="set", value="set"),
    app_commands.Choice(name="show", value="show"),
    app_commands.Choice(name="import", value="import"),
])
@app_commands.autocomplete(resource=resource_autocomplete)
async def isk_cmd(
    interaction: discord.Interaction,
    action: app_commands.Choice[str],
    resource: Optional[str] = None,
    price: Optional[float] = None,
    csv_prices: Optional[str] = None
):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))
    conn = ensure_db_ready()
    try:
        act = action.value.lower()
        if act == "set":
            if not resource or price is None:
                await interaction.followup.send("Нужно resource и price.", ephemeral=should_use_ephemeral(interaction)); return
            set_price(conn, resource, price)
            await interaction.followup.send(f"OK: **{clean_resource_name(resource)}** = **{price:.2f} ISK/ед**", ephemeral=False)
        elif act == "show":
            if resource:
                p = get_price(conn, resource)
                await interaction.followup.send(
                    f"**{clean_resource_name(resource)}** = **{p:.2f} ISK/ед**" if p is not None else "Цена не задана.",
                    ephemeral=should_use_ephemeral(interaction)
                )
            else:
                prices = get_prices(conn)
                if not prices:
                    await interaction.followup.send("Цены не заданы.", ephemeral=should_use_ephemeral(interaction))
                else:
                    lines = [f"- {r}: {v:.2f} ISK/ед" for r,v in sorted(prices.items())]
                    await interaction.followup.send("**Цены (ISK/ед):**\n" + "\n".join(lines), ephemeral=should_use_ephemeral(interaction))
        elif act == "import":
            if not csv_prices:
                await interaction.followup.send("Передай csv_prices: `resource,price` построчно.", ephemeral=should_use_ephemeral(interaction)); return
            import csv, io
            f = io.StringIO(csv_prices); reader = csv.reader(f); n=0
            for row in reader:
                if not row: continue
                res = clean_resource_name(row[0])
                try: val = float(row[1])
                except: continue
                set_price(conn, res, val); n+=1
            await interaction.followup.send(f"Импортировано: **{n}**.", ephemeral=should_use_ephemeral(interaction))
        else:
            await interaction.followup.send("Неизвестное действие.", ephemeral=should_use_ephemeral(interaction))
    except Exception as e:
        logger.exception("isk error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

# --- парсер склада: TSV/CSV/однострочный пробельный ---
def parse_have_table(text: str) -> List[Tuple[str, float, Optional[float]]]:
    """
    Поддерживает:
    - TSV/CSV (с заголовком/без)
    - Однострочный формат из Discord: 'ID Названия Количество Оценка стоимости 1 Lustering Alloy 13115344 7403...'
      Запись: [ID?] <Название с пробелами> <Количество> <Итоговая_стоимость>
    """
    import csv, io, re
    if not text or not text.strip():
        return []

    def to_float(x: str) -> Optional[float]:
        try:
            s = (x or "").strip().replace("\u00A0", "").replace(" ", "").replace(",", ".")
            return float(s) if s else None
        except Exception:
            return None

    txt = text.strip().replace("\r\n", "\n").replace("\r", "\n")

    # Попытка TSV/CSV по строкам
    if "\n" in txt and any(sep in txt for sep in ("\t", ",")):
        delimiter = "\t" if "\t" in txt else ","
        f = io.StringIO(txt)
        reader = csv.reader(f, delimiter=delimiter)
        rows = [r for r in reader if any((c or "").strip() for c in r)]
        if not rows:
            return []

        header = [c.strip().lower() for c in rows[0]]
        has_header = any(k in " ".join(header) for k in ["назван", "name", "resource", "колич", "amount", "оцен", "стоим", "value", "price"])

        def find_col(cands: List[str], default: int) -> int:
            if has_header:
                for i, c in enumerate(header):
                    if any(x in c for x in cands):
                        return i
            return default

        col_name   = find_col(["назв", "name", "resource", "ресур"], 1 if len(rows[0]) > 1 else 0)
        col_amount = find_col(["колич", "amount", "qty", "объём", "шт"], 2 if len(rows[0]) > 2 else (1 if len(rows[0])>1 else 0))
        col_total  = find_col(["оцен", "стоим", "total", "value", "сумм"], 3 if len(rows[0]) > 3 else -1)

        out: List[Tuple[str, float, Optional[float]]] = []
        start = 1 if has_header else 0
        for r in rows[start:]:
            shift = 1 if r and (r[0] or "").strip().isdigit() else 0
            def get(idx: int) -> Optional[str]:
                j = idx + shift
                return r[j] if 0 <= j < len(r) else None
            name = clean_resource_name(get(col_name) or "")
            if not name: continue
            amount = to_float(get(col_amount) or "")
            if amount is None: continue
            unit_price = None
            if col_total >= 0:
                total = to_float(get(col_total) or "")
                if total is not None and amount != 0:
                    unit_price = total / amount
            out.append((name, float(amount), unit_price))
        return out

    # Однострочный пробельный ввод
    tokens = re.findall(r"\S+", txt)
    header_words = {"ID", "Названия", "Количество", "Оценка", "стоимости", "Name", "Amount", "Total", "Value"}
    if any(t in header_words for t in tokens[:10]):
        tokens = [t for t in tokens if t not in header_words]

    out: List[Tuple[str, float, Optional[float]]] = []
    i, n = 0, len(tokens)
    while i < n:
        if i < n and tokens[i].isdigit():
            i += 1
        j = i; found = False
        while j + 1 < n:
            a = to_float(tokens[j]); b = to_float(tokens[j+1])
            if a is not None and b is not None:
                found = True; break
            j += 1
        if not found:
            # формат с одним числом (только amount)
            k, last_num_idx = i, -1
            while k < n:
                if to_float(tokens[k]) is not None:
                    last_num_idx = k
                k += 1
            if last_num_idx == -1 or last_num_idx == i:
                break
            name_tokens = tokens[i:last_num_idx]
            if name_tokens and name_tokens[0].isdigit():
                name_tokens = name_tokens[1:]
            name = " ".join(name_tokens).strip()
            amount = to_float(tokens[last_num_idx])
            if name and amount is not None:
                out.append((name, float(amount), None))
            break

        name_tokens = tokens[i:j]
        if name_tokens and name_tokens[0].isdigit():
            name_tokens = name_tokens[1:]
        name = " ".join(name_tokens).strip()
        amount = to_float(tokens[j])
        total  = to_float(tokens[j+1])
        unit_price = (total / amount) if (amount not in (None, 0) and total is not None) else None
        if name and amount is not None:
            out.append((name, float(amount), unit_price))
        i = j + 2
    return out

def upsert_have(conn: sqlite3.Connection, guild_id: int, items: List[Tuple[str, float, Optional[float]]]) -> int:
    cur = conn.cursor()
    n = 0
    for resource, amount, unit_price in items:
        resource = clean_resource_name(resource)
        if not resource: continue
        if amount < 0: amount = 0.0
        cur.execute("""
            INSERT INTO guild_have(guild_id, resource, amount_units, unit_price, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(guild_id, resource) DO UPDATE
              SET amount_units=excluded.amount_units,
                  unit_price=COALESCE(excluded.unit_price, guild_have.unit_price),
                  updated_at=excluded.updated_at
        """, (guild_id, resource, float(amount), float(unit_price) if unit_price is not None else None, now_utc_iso()))
        n += 1
    conn.commit()
    return n

@tree.command(name="have", description="Импорт/просмотр склада: цели уменьшаются на эти остатки; цены берутся из оценки/шт.")
@app_commands.describe(action="import/show/clear", data="Вставь таблицу (TSV/CSV) или одно строкой: [ID] Название Количество [Оценка]")
@app_commands.choices(action=[
    app_commands.Choice(name="import", value="import"),
    app_commands.Choice(name="show",   value="show"),
    app_commands.Choice(name="clear",  value="clear"),
])
async def have_cmd(interaction: discord.Interaction, action: app_commands.Choice[str], data: Optional[str] = None):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))
    conn = ensure_db_ready()
    try:
        act = action.value.lower()
        if act == "import":
            if not data or not data.strip():
                await interaction.followup.send("Вставь таблицу в параметр `data` (TSV/CSV или одной строкой).", ephemeral=should_use_ephemeral(interaction)); return
            items = parse_have_table(data)
            if not items:
                await interaction.followup.send("Не удалось распарсить данные. Проверь колонки/формат.", ephemeral=should_use_ephemeral(interaction)); return
            n = upsert_have(conn, guild.id, items)

            # Дополнительно запишем unit_price в isk_price (чтобы /isk show видел эти цены)
            for res, _amt, unit_price in items:
                if unit_price is not None:
                    set_price(conn, res, unit_price)

            total_items = len(items)
            with_prices = sum(1 for _,_,p in items if p is not None)
            await interaction.followup.send(f"✅ Импортировано позиций: **{n}/{total_items}** (с ценой: **{with_prices}**).", ephemeral=False)

        elif act == "show":
            cur = conn.cursor()
            cur.execute("""SELECT resource, amount_units, unit_price FROM guild_have WHERE guild_id=? ORDER BY resource""", (guild.id,))
            rows = cur.fetchall()
            if not rows:
                await interaction.followup.send("Склад пуст. Используй `/have import`.", ephemeral=should_use_ephemeral(interaction)); return
            lines=[]
            for r in rows[:50]:
                res = r["resource"]; amt = float(r["amount_units"]); up = r["unit_price"]
                tail = f" · ≈ {up:.2f} ISK/ед" if up is not None else ""
                lines.append(f"- {res}: **{amt:,.0f}**{tail}".replace(",", " "))
            more = "" if len(rows) <= 50 else f"\n… и ещё {len(rows)-50} строк."
            await interaction.followup.send("**Склад:**\n" + "\n".join(lines) + more, ephemeral=False)

        elif act == "clear":
            cur = conn.cursor()
            cur.execute("DELETE FROM guild_have WHERE guild_id=?", (guild.id,))
            n = cur.rowcount or 0
            conn.commit()
            await interaction.followup.send(f"🧹 Очищено записей склада: **{n}**.", ephemeral=should_use_ephemeral(interaction))
        else:
            await interaction.followup.send("Неизвестное действие.", ephemeral=should_use_ephemeral(interaction))
    except Exception as e:
        logger.exception("have error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

# --- ADDPOS ---
@tree.command(
    name="addpos",
    description="Создать/обновить POS (имя как в игре) и автоматически выставить планеты/буры под цели."
)
@app_commands.describe(
    name="Имя POS (как в игре)",
    system="Система (например, O3L-95)",
    slots="Сколько планет-слотов назначать (по умолчанию 10)",
    drills="Сколько буров ставить на планету (по умолчанию 22)",
)
@app_commands.autocomplete(system=system_autocomplete)
async def addpos_cmd(
    interaction: discord.Interaction,
    name: str,
    system: str,
    slots: Optional[int] = None,
    drills: Optional[int] = None,
):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

    gid = guild.id
    uid = interaction.user.id
    conn = ensure_db_ready()
    try:
        defaults = get_effective_pos_defaults(conn, gid, uid)
        base_slots = int(defaults["slots"])
        base_drills = int(defaults["drills"])
        slots_override = slots is not None
        drills_override = drills is not None
        slots_val = int(slots) if slots_override else base_slots
        drills_val = int(drills) if drills_override else base_drills

        err = _validate_default_bounds(slots_val, "slots") or _validate_default_bounds(drills_val, "drills")
        if err:
            await interaction.followup.send(err, ephemeral=should_use_ephemeral(interaction))
            return

        constellation = find_constellation_by_system(conn, system)
        if not constellation:
            await interaction.followup.send(f"Система **{system}** не найдена в базе.", ephemeral=should_use_ephemeral(interaction)); return

        cur = conn.cursor()
        cur.execute("SELECT id, owner_user_id FROM pos WHERE guild_id=? AND name=? LIMIT 1", (gid, name))
        r = cur.fetchone()
        if r:
            pos_id = int(r["id"])
            owner_id = int(r["owner_user_id"])
            if not (is_admin_user(interaction) or owner_id == uid):
                await interaction.followup.send("⛔ Этот POS принадлежит другому пользователю.", ephemeral=should_use_ephemeral(interaction))
                return
            cur.execute("UPDATE pos SET system=?, constellation=?, updated_at=? WHERE id=?",
                        (system, constellation, now_utc_iso(), pos_id))
            cur.execute("DELETE FROM pos_planet WHERE pos_id=?", (pos_id,))
            conn.commit()
        else:
            cur.execute("""INSERT INTO pos(guild_id, owner_user_id, name, system, constellation, created_at, updated_at)
                           VALUES(?,?,?,?,?,?,?)""",
                        (gid, uid, name, system, constellation, now_utc_iso(), now_utc_iso()))
            pos_id = cur.lastrowid
            conn.commit()

        assignments = compute_pos_assignments(
            conn=conn,
            guild_id=gid,
            constellation=constellation,
            slots=slots_val,
            drills=drills_val,
        )

        upsert_assignments(conn, pos_id, assignments)

        msg = build_pos_assignment_message(
            name=name,
            system=system,
            constellation=constellation,
            assignments=assignments,
            slots_val=slots_val,
            drills_val=drills_val,
            defaults=defaults,
            slots_override=slots_override,
            drills_override=drills_override,
        )
        await interaction.followup.send(msg, ephemeral=should_use_ephemeral(interaction))

    except Exception as e:
        logger.exception("addpos error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@tree.command(name="refreshpos", description="Сбросить все POS и запросить обновление у владельцев.")
async def refreshpos_cmd(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return
    if not is_admin_user(interaction):
        await interaction.response.send_message("⛔ Команда доступна только администраторам сервера.", ephemeral=should_use_ephemeral(interaction))
        return

    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))

    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, owner_user_id FROM pos WHERE guild_id=?", (guild.id,))
        rows = cur.fetchall()
        if not rows:
            await interaction.followup.send("На сервере нет POS для обновления.", ephemeral=should_use_ephemeral(interaction))
            return

        pos_ids = [int(r["id"]) for r in rows]
        owner_ids = sorted({int(r["owner_user_id"]) for r in rows if r["owner_user_id"] is not None})

        cur.execute(
            "DELETE FROM pos_planet WHERE pos_id IN (SELECT id FROM pos WHERE guild_id=?)",
            (guild.id,),
        )
        deleted_assignments = cur.rowcount or 0
        cleared_acks = clear_pos_ack(conn, pos_ids)
        conn.commit()
    except Exception as e:
        logger.exception("refreshpos error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
        return
    finally:
        conn.close()

    notified = 0
    failed: List[int] = []
    for owner_id in owner_ids:
        user = interaction.client.get_user(owner_id)
        if user is None:
            try:
                user = await interaction.client.fetch_user(owner_id)
            except Exception as e:
                logger.warning("Не удалось получить пользователя %s: %s", owner_id, e)
                user = None
        if user is None:
            failed.append(owner_id)
            continue

        view = RefreshPosView(guild.id)
        message = (
            f"👋 Привет! Администратор сервера **{guild.name}** сбросил назначения POS.\n"
            "Нажми кнопку ниже, чтобы выбрать количество слотов и буров и получить новый список планет.\n"
            "Оставь поля пустыми, чтобы взять сохранённые значения по умолчанию."
        )
        try:
            await user.send(message, view=view)
            notified += 1
        except discord.Forbidden:
            failed.append(owner_id)
        except Exception as e:
            logger.exception("Не удалось отправить DM пользователю %s: %s", owner_id, e)
            failed.append(owner_id)

    summary_lines = [
        f"Сброшено назначений: **{deleted_assignments}**.",
        f"Сброшено отметок обновления: **{cleared_acks}**.",
        f"Владельцев POS: **{len(owner_ids)}**.",
        f"Сообщений отправлено: **{notified}**.",
    ]
    if failed:
        failed_mentions = ", ".join(f"<@{uid}>" for uid in failed)
        summary_lines.append("Не удалось отправить: " + failed_mentions)

    await interaction.followup.send("\n".join(summary_lines), ephemeral=should_use_ephemeral(interaction))

@tree.command(name="delpos", description="Удалить один POS (если без id — покажу список).")
@app_commands.describe(pos_id="ID POS")
async def delpos_cmd(interaction: discord.Interaction, pos_id: Optional[int] = None):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))
    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        if pos_id is None:
            cur.execute("SELECT id, name, system, constellation, created_at FROM pos WHERE guild_id=? ORDER BY id DESC", (guild.id,))
            rows = cur.fetchall()
            if not rows:
                await interaction.followup.send("POS-ов нет.", ephemeral=should_use_ephemeral(interaction)); return
            lines = [f"ID **{r['id']}** — {r['name']} ({r['system']}, {r['constellation']}) · создан {r['created_at']}" for r in rows[:25]]
            await interaction.followup.send("Укажи `/delpos pos_id:<ID>`:\n" + "\n".join(lines), ephemeral=should_use_ephemeral(interaction))
            return

        cur.execute("SELECT id FROM pos WHERE id=? AND guild_id=?", (pos_id, guild.id))
        if not cur.fetchone():
            await interaction.followup.send("POS не найден.", ephemeral=should_use_ephemeral(interaction)); return

        if not ensure_owner_or_admin(conn, interaction, pos_id):
            await interaction.followup.send("⛔ Удалять может только владелец POS или админ сервера.", ephemeral=should_use_ephemeral(interaction))
            return

        cur.execute("DELETE FROM pos_planet WHERE pos_id=?", (pos_id,))
        cur.execute("DELETE FROM pos WHERE id=?", (pos_id,))
        conn.commit()
        await interaction.followup.send(f"🗑️ POS **{pos_id}** удалён.", ephemeral=should_use_ephemeral(interaction))
    except Exception as e:
        logger.exception("delpos error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@tree.command(name="delallpos", description="Удалить все POS на сервере.")
async def delallpos_cmd(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))
    if not is_admin_user(interaction):
        await interaction.followup.send("⛔ Команда доступна только администраторам сервера.", ephemeral=should_use_ephemeral(interaction))
        return
    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM pos WHERE guild_id=?", (guild.id,))
        ids = [r["id"] for r in cur.fetchall()]
        if ids:
            cur.execute("DELETE FROM pos_planet WHERE pos_id IN (%s)" % ",".join("?"*len(ids)), ids)
            cur.execute("DELETE FROM pos WHERE guild_id=?", (guild.id,))
            conn.commit()
        await interaction.followup.send(f"🧹 Удалено POS-ов: **{len(ids)}**.", ephemeral=should_use_ephemeral(interaction))
    except Exception as e:
        logger.exception("delallpos error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@tree.command(name="mypos", description="Показать мои POS на сервере.")
async def mypos_cmd(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))
    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, system, constellation, created_at, updated_at
            FROM pos
            WHERE guild_id=? AND owner_user_id=?
            ORDER BY id DESC
        """, (guild.id, interaction.user.id))
        rows = cur.fetchall()
        if not rows:
            await interaction.followup.send("У тебя ещё нет POS-ов.", ephemeral=should_use_ephemeral(interaction)); return
        lines = [
            f"ID **{r['id']}** — {r['name']} ({r['system']}, {r['constellation']}) · создан {r['created_at']} · обновлён {r['updated_at']}"
            for r in rows[:25]
        ]
        await interaction.followup.send("\n".join(lines), ephemeral=should_use_ephemeral(interaction))
    except Exception as e:
        logger.exception("mypos error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

@tree.command(name="posstats", description="Статистика POS на сервере: всего и по владельцам.")
async def posstats_cmd(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=False)
    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM pos WHERE guild_id=?", (guild.id,))
        total = int(cur.fetchone()["c"])
        cur.execute("""
            SELECT owner_user_id, COUNT(*) AS c
            FROM pos
            WHERE guild_id=?
            GROUP BY owner_user_id
            ORDER BY c DESC, owner_user_id ASC
        """, (guild.id,))
        rows = cur.fetchall()
        owners_total = len(rows)
        my_count = 0
        lines = []
        for r in rows[:25]:
            uid = int(r["owner_user_id"])
            cnt = int(r["c"])
            if uid == interaction.user.id:
                my_count = cnt
            member = guild.get_member(uid)
            mention = member.mention if member else f"<@{uid}>"
            lines.append(f"{mention}: **{cnt}**")
        header = f"**POS на сервере:** {total}\n**Владельцев:** {owners_total}\n**У тебя:** {my_count}"
        body = "**Топ по владельцам:**\n" + ("\n".join(lines) if lines else "_нет POS_")
        await interaction.followup.send(header + "\n\n" + body, ephemeral=False)
    except Exception as e:
        logger.exception("posstats error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

# --- Обновление бота из Discord ---
@tree.command(name="updatebot", description="Обновить код бота из GitHub (только для администраторов).")
@app_commands.describe(
    branch="Ветка git для обновления (опционально)",
    reinstall_deps="Переустановить зависимости из requirements.txt после git pull",
)
async def updatebot_cmd(
    interaction: discord.Interaction,
    branch: Optional[str] = None,
    reinstall_deps: bool = False,
):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Команда доступна только в сервере.", ephemeral=should_use_ephemeral(interaction))
        return
    if not is_admin_user(interaction):
        await interaction.response.send_message("Требуются права администратора сервера.", ephemeral=should_use_ephemeral(interaction))
        return

    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))
    try:
        success, log_text = await update_bot_repository(branch, reinstall_deps)
        status = "✅ Обновление завершено успешно." if success else "⚠️ Обновление завершилось с ошибкой."

        extra_lines: List[str] = []
        restart_task: Optional[asyncio.Task] = None
        if success:
            info, restart_task = prepare_service_restart()
            if info:
                extra_lines.append(info)
            if restart_task is None and BOT_SERVICE_NAME:
                logger.warning(
                    "Перезапуск службы %s не был запланирован (проверьте systemctl).",
                    BOT_SERVICE_NAME,
                )
            remember_successful_update(interaction)
        else:
            extra_lines.append("Перезапуск службы пропущен из-за ошибки обновления.")

        body = f"{status}\n\n{log_text}"
        if extra_lines:
            body += "\n\n" + "\n".join(extra_lines)

        await send_long(
            interaction,
            body,
            ephemeral=should_use_ephemeral(interaction),
            title="Обновление бота",
        )

        if restart_task is None and success and not BOT_SERVICE_NAME:
            logger.info("Перезапуск службы после updatebot не настроен (BOT_SERVICE_NAME не задана).")
    except Exception as e:
        logger.exception("updatebot error: %s", e)
        await interaction.followup.send(f"Ошибка при обновлении: {e}", ephemeral=should_use_ephemeral(interaction))


# --- НОВОЕ: мои назначения (планеты/буры) ---
@tree.command(name="myassigns", description="Список планет/буров по моим POS (опционально по одному POS).")
@app_commands.describe(pos_id="Фильтровать по ID POS (опционально)")
async def myassigns_cmd(interaction: discord.Interaction, pos_id: Optional[int] = None):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=should_use_ephemeral(interaction))
    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        if pos_id is not None:
            cur.execute("""
                SELECT id, name, system, constellation
                FROM pos
                WHERE guild_id=? AND id=? AND owner_user_id=?
            """, (guild.id, pos_id, interaction.user.id))
            pos_rows = cur.fetchall()
            if not pos_rows:
                await interaction.followup.send("POS не найден или принадлежит другому пользователю.", ephemeral=should_use_ephemeral(interaction)); return
        else:
            cur.execute("""
                SELECT id, name, system, constellation
                FROM pos
                WHERE guild_id=? AND owner_user_id=?
                ORDER BY id DESC
            """, (guild.id, interaction.user.id))
            pos_rows = cur.fetchall()
            if not pos_rows:
                await interaction.followup.send("У тебя ещё нет POS-ов.", ephemeral=should_use_ephemeral(interaction)); return

        lines = []
        for p in pos_rows:
            pid = int(p["id"])
            lines.append(f"**POS {p['name']}** (ID {pid}) — {p['system']}, {p['constellation']}")
            cur.execute("""
                SELECT
                    pp.planet_id,
                    pp.resource,
                    pp.drills_count,
                    pp.rate,
                    pr.system AS pr_system,
                    pr.planet_name AS pr_planet
                FROM pos_planet pp
                LEFT JOIN planet_resources pr
                  ON (CASE WHEN typeof(pr.planet_id)='integer' AND pr.planet_id>0
                           THEN pr.planet_id ELSE pr.rowid END) = pp.planet_id
                WHERE pp.pos_id=?
                ORDER BY pp.resource, pp.planet_id
            """, (pid,))
            arows = cur.fetchall()
            if not arows:
                lines.append("  _Назначений нет._")
                continue
            for r in arows:
                total_rate = float(r["rate"]) * int(r["drills_count"])
                sys = r["pr_system"] or "?"
                planet = r["pr_planet"] or f"#{r['planet_id']}"
                lines.append(
                    f"- {sys} · {planet} · **{r['resource']}** · drills={r['drills_count']} · "
                    f"base={float(r['rate']):.2f}/h/bore → **{total_rate:,.2f}/ч**".replace(",", " ")
                )
            lines.append("")
        await send_long(interaction, "\n".join(lines), ephemeral=should_use_ephemeral(interaction), title="Мои назначения")
    except Exception as e:
        logger.exception("myassigns error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

# --- НОВОЕ: POS конкретного пользователя ---
@tree.command(name="userpos", description="Список POS указанного пользователя на сервере.")
@app_commands.describe(user="Пользователь (упоминание или выбор из списка)")
async def userpos_cmd(interaction: discord.Interaction, user: discord.User):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Только в сервере.", ephemeral=should_use_ephemeral(interaction)); return
    await interaction.response.defer(ephemeral=False)
    conn = ensure_db_ready()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, system, constellation, created_at, updated_at
            FROM pos
            WHERE guild_id=? AND owner_user_id=?
            ORDER BY id DESC
        """, (guild.id, user.id))
        rows = cur.fetchall()

        member = guild.get_member(user.id)
        mention = member.mention if member else f"<@{user.id}>"

        if not rows:
            await interaction.followup.send(f"У {mention} нет POS-ов.", ephemeral=False)
            return

        lines = [f"**POS пользователя {mention}: {len(rows)}**", ""]
        for r in rows[:50]:
            lines.append(
                f"ID **{r['id']}** — {r['name']} ({r['system']}, {r['constellation']}) · "
                f"создан {r['created_at']} · обновлён {r['updated_at']}"
            )
        if len(rows) > 50:
            lines.append(f"\n… и ещё {len(rows)-50} записей.")

        await send_long(interaction, "\n".join(lines), ephemeral=False, title="POS пользователя")
    except Exception as e:
        logger.exception("userpos error: %s", e)
        await interaction.followup.send(f"Ошибка: {e}", ephemeral=should_use_ephemeral(interaction))
    finally:
        conn.close()

# ==================== СТАРТ ====================
@bot.event
async def on_ready():
    logger.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "?")
    try:
        await tree.sync()
        logger.info("Application commands synced.")
    except Exception as e:
        logger.exception("Failed to sync commands: %s", e)

    conn = None
    try:
        conn = ensure_db_ready()
        cur = conn.cursor()
        cur.execute("SELECT id FROM resource_ping WHERE active=1")
        rows = cur.fetchall()
        for row in rows:
            ping_id = int(row["id"])
            bot.add_view(ResourcePingView(ping_id))
        if rows:
            logger.info("Зарегистрировано напоминаний с кнопкой: %s", len(rows))
    except Exception as e:
        logger.exception("Не удалось зарегистрировать напоминания: %s", e)
    finally:
        if conn is not None:
            conn.close()

    global reminder_task
    if reminder_task is None or reminder_task.done():
        reminder_task = bot.loop.create_task(resource_reminder_loop())
        logger.info("Цикл напоминаний о сдаче ресурсов запущен.")

    await notify_after_restart()

def main():
    token = validate_and_clean_token(DISCORD_TOKEN)
    if not token:
        logger.error(
            "Discord token не задан или указан некорректно. "
            "Укажите валидный токен в константе DISCORD_TOKEN в коде.",
        )
        raise SystemExit(1)

    logger.info("Использую токен, указанный в коде.")
    try:
        bot.run(token)
    except discord.errors.LoginFailure:
        logger.error(
            "LoginFailure: Discord отклонил токен (401 Unauthorized). "
            "Скорее всего токен устарел/сброшен или скопирован не полностью. "
            "Зайдите в Discord Developer Portal → Your App → Bot → Reset Token и обновите переменную окружения."
        )
        raise

if __name__ == "__main__":
    main()

