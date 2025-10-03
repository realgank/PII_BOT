#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Установка/обновление PII BOT.

Использование: sudo ./scripts/install.sh --repo <URL> [--branch main] [--dir /opt/pii_bot] [--data /var/lib/pii_bot]

Параметры:
  --repo    URL git-репозитория (обязательно).
  --branch  Ветка, которую нужно развернуть (по умолчанию main).
  --dir     Каталог установки кода бота (по умолчанию /opt/pii_bot).
  --data    Каталог для данных (planets.db) (по умолчанию /var/lib/pii_bot).
  --no-deps Не устанавливать системные зависимости через apt.
USAGE
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

REPO_URL=""
BRANCH="main"
INSTALL_DIR="/opt/pii_bot"
DATA_DIR="/var/lib/pii_bot"
INSTALL_DEPS=1
APT_UPDATED=0

to_systemd_unit_name() {
    local name="$1"
    name=$(echo "$name" | tr '[:upper:]' '[:lower:]')
    name=$(echo "$name" | sed 's/[^a-z0-9]/-/g')
    name=$(echo "$name" | sed 's/-\{2,\}/-/g')
    name=$(echo "$name" | sed 's/^-\+//; s/-\+$//')
    if [[ -z "$name" ]]; then
        name="pii-bot"
    fi
    printf '%s' "$name"
}

sanitize_env_var() {
    local name="$1"
    name=$(echo "$name" | tr '[:lower:]' '[:upper:]')
    name=$(echo "$name" | sed 's/[^A-Z0-9_]/_/g')
    name=$(echo "$name" | sed 's/_\{2,\}/_/g')
    while [[ "$name" == _* ]]; do
        name="${name#_}"
    done
    if [[ -z "$name" ]]; then
        name="PII_BOT_DISCORD_TOKEN"
    fi
    if [[ $name =~ ^[0-9] ]]; then
        name="_${name}"
    fi
    printf '%s' "$name"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            REPO_URL="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --no-deps)
            INSTALL_DEPS=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Неизвестный параметр: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$REPO_URL" ]]; then
    echo "Параметр --repo обязателен." >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    echo "⚠️  Скрипт рекомендуется запускать от root (sudo), иначе возможны ошибки при установке зависимостей." >&2
fi

ensure_dep() {
    local pkg="$1"
    if ! command -v "$pkg" >/dev/null 2>&1; then
        echo "Устанавливаю пакет $pkg ..."
        if [[ $APT_UPDATED -eq 0 ]]; then
            apt-get update -y
            APT_UPDATED=1
        fi
        apt-get install -y "$pkg"
    fi
}

if [[ $INSTALL_DEPS -eq 1 ]]; then
    ensure_dep git
    ensure_dep python3
    ensure_dep python3-venv
fi

mkdir -p "$INSTALL_DIR"
mkdir -p "$DATA_DIR"

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    echo "Клонирую репозиторий $REPO_URL в $INSTALL_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
else
    echo "Обновляю существующий репозиторий в $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch origin "$BRANCH"
    git -C "$INSTALL_DIR" reset --hard "origin/$BRANCH"
fi

echo "Создаю виртуальное окружение..."
PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip

if [[ -f "$INSTALL_DIR/requirements.txt" ]]; then
    echo "Устанавливаю зависимости из requirements.txt"
    "${INSTALL_DIR}/.venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
else
    echo "Файл requirements.txt не найден, пропускаю установку Python-зависимостей"
fi

DB_SOURCE="$INSTALL_DIR/planets.db"
DB_TARGET="$DATA_DIR/planets.db"

if [[ -f "$DB_TARGET" ]]; then
    echo "База данных уже существует ($DB_TARGET), пропускаю копирование."
else
    if [[ -f "$DB_SOURCE" ]]; then
        echo "Копирую базу данных в $DB_TARGET"
        cp "$DB_SOURCE" "$DB_TARGET"
    else
        echo "⚠️  Файл planets.db не найден в репозитории, пропускаю копирование."
    fi
fi

BASENAME_ENV=$(sanitize_env_var "$(basename "$INSTALL_DIR")")
if [[ -z "$BASENAME_ENV" ]]; then
    BASENAME_ENV="PII_BOT"
fi
DEFAULT_ENV_NAME=$(sanitize_env_var "${BASENAME_ENV}_DISCORD_TOKEN")

echo -n "Введите Discord-токен бота: "
read -r -s DISCORD_TOKEN_VALUE
echo
while [[ -z "${DISCORD_TOKEN_VALUE}" ]]; do
    echo -n "Токен не может быть пустым. Повторите ввод: "
    read -r -s DISCORD_TOKEN_VALUE
    echo
done

read -r -p "Имя переменной окружения для токена [$DEFAULT_ENV_NAME]: " TOKEN_ENV_NAME
if [[ -z "${TOKEN_ENV_NAME}" ]]; then
    TOKEN_ENV_NAME="$DEFAULT_ENV_NAME"
fi

SANITIZED_TOKEN_ENV=$(sanitize_env_var "$TOKEN_ENV_NAME")
if [[ "$SANITIZED_TOKEN_ENV" != "$TOKEN_ENV_NAME" ]]; then
    echo "Имя переменной скорректировано до $SANITIZED_TOKEN_ENV для совместимости со стандартом POSIX."
fi
TOKEN_ENV_NAME="$SANITIZED_TOKEN_ENV"

ENV_FILE="$INSTALL_DIR/.env"
read -r -p "Путь к файлу базы данных [$DB_TARGET]: " DB_PATH_VALUE
if [[ -z "${DB_PATH_VALUE}" ]]; then
    DB_PATH_VALUE="$DB_TARGET"
fi

DEFAULT_LOG_DIR="$INSTALL_DIR/logs"
read -r -p "Каталог для логов [$DEFAULT_LOG_DIR]: " LOG_DIR_VALUE
if [[ -z "${LOG_DIR_VALUE}" ]]; then
    LOG_DIR_VALUE="$DEFAULT_LOG_DIR"
fi

read -r -p "Уровень логирования (DEBUG/INFO/...) [INFO]: " LOG_LEVEL_VALUE
if [[ -z "${LOG_LEVEL_VALUE}" ]]; then
    LOG_LEVEL_VALUE="INFO"
fi

read -r -p "Слоты POS по умолчанию [10]: " DEFAULT_SLOTS_VALUE
if [[ -z "${DEFAULT_SLOTS_VALUE}" ]]; then
    DEFAULT_SLOTS_VALUE="10"
fi

read -r -p "Количество буров по умолчанию [22]: " DEFAULT_DRILLS_VALUE
if [[ -z "${DEFAULT_DRILLS_VALUE}" ]]; then
    DEFAULT_DRILLS_VALUE="22"
fi

read -r -p "Горизонт прогноза (часов) [168]: " DEFAULT_HOURS_VALUE
if [[ -z "${DEFAULT_HOURS_VALUE}" ]]; then
    DEFAULT_HOURS_VALUE="168"
fi

read -r -p "Коэффициент достоверности добычи [0.85]: " DEFAULT_BETA_VALUE
if [[ -z "${DEFAULT_BETA_VALUE}" ]]; then
    DEFAULT_BETA_VALUE="0.85"
fi

read -r -p "Задержка повторного напоминания (часов) [24]: " RES_REMINDER_DELAY_HOURS_VALUE
if [[ -z "${RES_REMINDER_DELAY_HOURS_VALUE}" ]]; then
    RES_REMINDER_DELAY_HOURS_VALUE="24"
fi

read -r -p "Период проверки напоминаний (секунд) [3600]: " RES_REMINDER_CHECK_SECONDS_VALUE
if [[ -z "${RES_REMINDER_CHECK_SECONDS_VALUE}" ]]; then
    RES_REMINDER_CHECK_SECONDS_VALUE="3600"
fi

echo -n "GitHub токен (PAT) для доступа к репозиторию (опционально): "
read -r -s GITHUB_TOKEN_VALUE
echo

DEFAULT_UPDATE_BRANCH="$BRANCH"
read -r -p "Ветка по умолчанию для команды обновления бота [${DEFAULT_UPDATE_BRANCH:-<пусто>}]: " BOT_UPDATE_BRANCH_VALUE
if [[ -z "${BOT_UPDATE_BRANCH_VALUE}" ]]; then
    BOT_UPDATE_BRANCH_VALUE="$DEFAULT_UPDATE_BRANCH"
fi

echo "Записываю переменные окружения в $ENV_FILE"
if [[ -f "$ENV_FILE" ]]; then
    BACKUP_FILE="$ENV_FILE.bak.$(date +%s)"
    cp "$ENV_FILE" "$BACKUP_FILE"
    echo "Существующий файл сохранён как $BACKUP_FILE"
fi

cat >"$ENV_FILE" <<ENVFILE
# Автоматически создано install.sh $(date -u +%Y-%m-%dT%H:%M:%SZ)
export ${TOKEN_ENV_NAME}="${DISCORD_TOKEN_VALUE}"
export PII_BOT_TOKEN_ENV="${TOKEN_ENV_NAME}"
export DB_PATH="${DB_PATH_VALUE}"
export LOG_DIR="${LOG_DIR_VALUE}"
export LOG_LEVEL="${LOG_LEVEL_VALUE}"
export DEFAULT_SLOTS="${DEFAULT_SLOTS_VALUE}"
export DEFAULT_DRILLS="${DEFAULT_DRILLS_VALUE}"
export DEFAULT_HOURS="${DEFAULT_HOURS_VALUE}"
export DEFAULT_BETA="${DEFAULT_BETA_VALUE}"
export RES_REMINDER_DELAY_HOURS="${RES_REMINDER_DELAY_HOURS_VALUE}"
export RES_REMINDER_CHECK_SECONDS="${RES_REMINDER_CHECK_SECONDS_VALUE}"
ENVFILE

if [[ -n "${GITHUB_TOKEN_VALUE}" ]]; then
    ESCAPED_GITHUB_TOKEN=$(printf '%s' "${GITHUB_TOKEN_VALUE}" | sed 's/\\/\\\\/g; s/"/\\"/g')
    printf 'export GITHUB_TOKEN="%s"\n' "${ESCAPED_GITHUB_TOKEN}" >>"$ENV_FILE"
fi
if [[ -n "${BOT_UPDATE_BRANCH_VALUE}" ]]; then
    ESCAPED_UPDATE_BRANCH=$(printf '%s' "${BOT_UPDATE_BRANCH_VALUE}" | sed 's/\\/\\\\/g; s/"/\\"/g')
    printf 'export BOT_UPDATE_BRANCH="%s"\n' "${ESCAPED_UPDATE_BRANCH}" >>"$ENV_FILE"
fi

chmod 600 "$ENV_FILE"

DEFAULT_SERVICE_UNIT_NAME=$(to_systemd_unit_name "$(basename "$INSTALL_DIR")")
DEFAULT_SERVICE_UNIT_FILE="${DEFAULT_SERVICE_UNIT_NAME}.service"
read -r -p "Создать systemd-службу для автозапуска бота? [Y/n]: " CREATE_SERVICE_CHOICE
if [[ -z "${CREATE_SERVICE_CHOICE}" || $CREATE_SERVICE_CHOICE =~ ^[YyАаДд] ]]; then
    read -r -p "Имя сервиса [${DEFAULT_SERVICE_UNIT_FILE}]: " SERVICE_UNIT_INPUT
    if [[ -z "${SERVICE_UNIT_INPUT}" ]]; then
        SERVICE_UNIT_INPUT="$DEFAULT_SERVICE_UNIT_FILE"
    fi
    SERVICE_UNIT_INPUT=$(to_systemd_unit_name "${SERVICE_UNIT_INPUT%.service}")
    SERVICE_UNIT_INPUT="${SERVICE_UNIT_INPUT}.service"

    DEFAULT_SERVICE_USER="${SUDO_USER:-$(logname 2>/dev/null || true)}"
    if [[ -z "$DEFAULT_SERVICE_USER" ]]; then
        DEFAULT_SERVICE_USER="$(whoami)"
    fi
    read -r -p "Пользователь, от имени которого запускать сервис [$DEFAULT_SERVICE_USER]: " SERVICE_USER_INPUT
    if [[ -z "${SERVICE_USER_INPUT}" ]]; then
        SERVICE_USER_INPUT="$DEFAULT_SERVICE_USER"
    fi

    SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_UNIT_INPUT}"
    if [[ $EUID -ne 0 ]]; then
        echo "⚠️  Недостаточно прав для записи в $SERVICE_FILE_PATH. Запустите скрипт с правами root или создайте юнит вручную." >&2
    else
        SERVICE_EXEC="source \"$ENV_FILE\" && exec \"$INSTALL_DIR/.venv/bin/python\" \"$INSTALL_DIR/PII_BOT.py\""
        SERVICE_EXEC_ESCAPED=$(printf '%s' "$SERVICE_EXEC" | sed "s/'/'\\\\''/g")

        echo "Создаю systemd-сервис в $SERVICE_FILE_PATH"
        if [[ -f "$SERVICE_FILE_PATH" ]]; then
            SERVICE_BACKUP="$SERVICE_FILE_PATH.bak.$(date +%s)"
            cp "$SERVICE_FILE_PATH" "$SERVICE_BACKUP"
            echo "Существующий файл сервиса сохранён как $SERVICE_BACKUP"
        fi
        cat >"$SERVICE_FILE_PATH" <<SERVICE_UNIT
[Unit]
Description=PII BOT (${INSTALL_DIR})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER_INPUT}
WorkingDirectory=${INSTALL_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash -lc '${SERVICE_EXEC_ESCAPED}'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE_UNIT

        if command -v systemctl >/dev/null 2>&1; then
            echo "Перезагружаю конфигурацию systemd"
            systemctl daemon-reload
            if systemctl enable --now "$SERVICE_UNIT_INPUT"; then
                echo "Служба $SERVICE_UNIT_INPUT включена и запущена."
            else
                echo "Не удалось запустить или включить $SERVICE_UNIT_INPUT. Проверьте сообщения об ошибках." >&2
            fi
        else
            echo "⚠️  Команда systemctl не найдена. Созданный юнит необходимо активировать вручную."
        fi
    fi
else
    echo "Создание systemd-сервиса пропущено по запросу пользователя."
fi

echo "Готово. Для запуска укажите переменную окружения DB_PATH=$DB_PATH_VALUE"
echo "Пример: DB_PATH=$DB_PATH_VALUE ${INSTALL_DIR}/.venv/bin/python PII_BOT.py"
echo "Можно также загрузить переменные командой: source $ENV_FILE"
