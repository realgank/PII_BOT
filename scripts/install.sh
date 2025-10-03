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

echo "Готово. Для запуска укажите переменную окружения DB_PATH=$DB_TARGET"
echo "Пример: DB_PATH=$DB_TARGET ${INSTALL_DIR}/.venv/bin/python PII_BOT.py"
