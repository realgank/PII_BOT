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
export DB_PATH="${DB_TARGET}"
ENVFILE

chmod 600 "$ENV_FILE"

echo "Готово. Для запуска укажите переменную окружения DB_PATH=$DB_TARGET"
echo "Пример: DB_PATH=$DB_TARGET ${INSTALL_DIR}/.venv/bin/python PII_BOT.py"
echo "Можно также загрузить переменные командой: source $ENV_FILE"
