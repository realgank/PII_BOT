"""Localization and translation helpers for the PII bot."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from collections import OrderedDict
from string import Formatter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==================== ЛОКАЛИЗАЦИЯ ====================
SUPPORTED_LANGUAGES: "OrderedDict[str, Dict[str, str]]" = OrderedDict(
    {
        "ru": {"name": "Русский", "emoji": "🇷🇺"},
        "en": {"name": "English", "emoji": "🇬🇧"},
    }
)
DEFAULT_LANGUAGE = "ru"

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "admin_required": {
        "ru": "Требуются права администратора сервера.",
        "en": "Server administrator permissions are required.",
    },
    "amount_non_negative": {
        "ru": "amount должен быть >= 0.",
        "en": "`amount` must be greater than or equal to 0.",
    },
    "cannot_determine_channel": {
        "ru": "Не удалось определить канал для отправки сообщения.",
        "en": "Failed to determine the channel to send the message to.",
    },
    "cannot_determine_permissions": {
        "ru": "Не удалось определить права бота на сервере. Проверьте, что бот добавлен на сервер.",
        "en": "Could not determine the bot permissions on the server. Make sure the bot is added to the server.",
    },
    "channel_same_server_only": {
        "ru": "Можно выбрать только канал текущего сервера.",
        "en": "You can only choose a channel from this server.",
    },
    "command_admin_only": {
        "ru": "⛔ Команда доступна только администраторам сервера.",
        "en": "⛔ This command is only available to server administrators.",
    },
    "command_server_only": {
        "ru": "Команда доступна только в сервере.",
        "en": "This command can only be used inside a server.",
    },
    "csv_prices_required": {
        "ru": "Передай csv_prices: `resource,price` построчно.",
        "en": "Provide `csv_prices` as lines in the `resource,price` format.",
    },
    "delete_owner_or_admin": {
        "ru": "⛔ Удалять может только владелец POS или админ сервера.",
        "en": "⛔ Only the POS owner or a server admin can delete it.",
    },
    "drills_not_int": {
        "ru": "`drills` должен быть целым числом.",
        "en": "`drills` must be an integer.",
    },
    "error_generic": {
        "ru": "Ошибка: {error}",
        "en": "Error: {error}",
    },
    "import_data_required": {
        "ru": "Вставь таблицу в параметр `data` (TSV/CSV или одной строкой).",
        "en": "Paste the table into the `data` parameter (TSV/CSV or a single line).",
    },
    "imported_count": {
        "ru": "Импортировано: **{count}**.",
        "en": "Imported: **{count}**.",
    },
    "language_changed": {
        "ru": "Язык обновлён на {language_emoji} {language_name}.",
        "en": "Language updated to {language_emoji} {language_name}.",
    },
    "language_current": {
        "ru": "Текущий язык: {language_emoji} {language_name}.",
        "en": "Current language: {language_emoji} {language_name}.",
    },
    "language_invalid": {
        "ru": "Неизвестный язык.",
        "en": "Unknown language.",
    },
    "language_options": {
        "ru": "Доступные языки: {languages}.",
        "en": "Available languages: {languages}.",
    },
    "language_usage": {
        "ru": "Используй `/language <ru|en>` для смены языка.",
        "en": "Use `/language <ru|en>` to change the language.",
    },
    "list_too_long": {
        "ru": "📄 {title}: список слишком большой, приложил файлом.",
        "en": "📄 {title}: the list is too large, attached as a file.",
    },
    "logs_channel_set": {
        "ru": "✅ Логи будут отправляться в {channel}.",
        "en": "✅ Logs will be sent to {channel}.",
    },
    "missing_permissions": {
        "ru": "⛔ У бота нет необходимых прав для отправки напоминания в этом канале: {permissions}.",
        "en": "⛔ The bot lacks the required permissions to send a reminder in this channel: {permissions}.",
    },
    "no_active_reminders": {
        "ru": "Нет активных напоминаний, к которым можно отправить отчёт.",
        "en": "There are no active reminders to report to.",
    },
    "no_assigned_planets": {
        "ru": "У тебя нет назначенных планет на этом сервере.",
        "en": "You have no assigned planets on this server.",
    },
    "no_data_period": {
        "ru": "Нет данных о сдаче ресурсов за выбранный период.",
        "en": "No resource submissions for the selected period.",
    },
    "no_pos": {
        "ru": "POS-ов нет.",
        "en": "No POS found.",
    },
    "no_pos_assignments": {
        "ru": "Назначений POS пока нет.",
        "en": "No POS assignments yet.",
    },
    "no_pos_in_server_for_user": {
        "ru": "Для этого сервера у тебя нет POS.",
        "en": "You have no POS on this server.",
    },
    "no_pos_on_server": {
        "ru": "На сервере ещё нет POS.",
        "en": "There are no POS on the server yet.",
    },
    "no_pos_to_update": {
        "ru": "На сервере нет POS для обновления.",
        "en": "There are no POS to update on this server.",
    },
    "not_enough_data_period": {
        "ru": "Недостаточно данных о сдаче ресурсов за выбранный период.",
        "en": "Not enough resource submission data for the selected period.",
    },
    "parse_failed": {
        "ru": "Не удалось распарсить данные. Проверь колонки/формат.",
        "en": "Failed to parse the data. Check the columns/format.",
    },
    "pos_belongs_other": {
        "ru": "⛔ Этот POS принадлежит другому пользователю.",
        "en": "⛔ This POS belongs to another user.",
    },
    "pos_deleted": {
        "ru": "🗑️ POS **{name}** удалён.",
        "en": "🗑️ POS **{name}** has been deleted.",
    },
    "pos_deleted_count": {
        "ru": "🧹 Удалено POS-ов: **{count}**.",
        "en": "🧹 POS deleted: **{count}**.",
    },
    "pos_not_found": {
        "ru": "POS не найден.",
        "en": "POS not found.",
    },
    "pos_not_found_deleted": {
        "ru": "POS не найден (возможно, удалён).",
        "en": "POS not found (possibly deleted).",
    },
    "pos_not_found_or_not_owned": {
        "ru": "POS не найден или принадлежит другому пользователю.",
        "en": "POS not found or belongs to another user.",
    },
    "positions_imported": {
        "ru": "✅ Импортировано позиций: **{processed}/{total}** (с ценой: **{priced}**).",
        "en": "✅ Imported positions: **{processed}/{total}** (with price: **{priced}**).",
    },
    "prices_not_set": {
        "ru": "Цены не заданы.",
        "en": "No prices configured.",
    },
    "reminder_closed": {
        "ru": "Это напоминание уже закрыто.",
        "en": "This reminder is already closed.",
    },
    "reminder_not_available": {
        "ru": "Напоминание больше не доступно.",
        "en": "The reminder is no longer available.",
    },
    "reminder_not_found": {
        "ru": "Напоминание не найдено.",
        "en": "Reminder not found.",
    },
    "reminder_not_found_id": {
        "ru": "Напоминание с таким ID не найдено на этом сервере.",
        "en": "A reminder with this ID was not found on this server.",
    },
    "reminder_other_server": {
        "ru": "Это напоминание относится к другому серверу.",
        "en": "This reminder belongs to another server.",
    },
    "reminder_text_empty": {
        "ru": "Текст напоминания не должен быть пустым.",
        "en": "Reminder text must not be empty.",
    },
    "reminder_text_too_long": {
        "ru": "Текст напоминания слишком длинный (максимум 2000 символов).",
        "en": "Reminder text is too long (maximum 2000 characters).",
    },
    "resource_price_required": {
        "ru": "Нужно resource и price.",
        "en": "`resource` and `price` are required.",
    },
    "save_submission_error": {
        "ru": "Ошибка при сохранении отметки. Попробуй позже.",
        "en": "Failed to save the submission. Please try again later.",
    },
    "send_failed": {
        "ru": "Не удалось отправить сообщение: {error}",
        "en": "Failed to send the message: {error}",
    },
    "server_only_short": {
        "ru": "Только в сервере.",
        "en": "Server only.",
    },
    "slots_not_int": {
        "ru": "`slots` должен быть целым числом.",
        "en": "`slots` must be an integer.",
    },
    "specify_log_channel": {
        "ru": "Укажи текстовый канал сервера для логов.",
        "en": "Specify a server text channel for logs.",
    },
    "system_not_found": {
        "ru": "Система **{system}** не найдена в базе.",
        "en": "System **{system}** not found in the database.",
    },
    "target_not_set": {
        "ru": "Для **{resource}** цель не задана.",
        "en": "No target is configured for **{resource}**.",
    },
    "targets_deleted": {
        "ru": "Удалено целей: **{count}**.",
        "en": "Targets removed: **{count}**.",
    },
    "unknown_action": {
        "ru": "Неизвестное действие.",
        "en": "Unknown action.",
    },
    "update_error": {
        "ru": "Ошибка при обновлении: {error}",
        "en": "Update failed: {error}",
    },
    "update_owner_only": {
        "ru": "⛔ Отметить обновление может только владелец POS.",
        "en": "⛔ Only the POS owner can acknowledge the update.",
    },
    "user_has_no_pos": {
        "ru": "У {user} нет POS-ов.",
        "en": "{user} has no POS.",
    },
    "warehouse_cleared": {
        "ru": "🧹 Очищено записей склада: **{count}**.",
        "en": "🧹 Warehouse entries cleared: **{count}**.",
    },
    "warehouse_empty": {
        "ru": "Склад пуст. Используй `/have import`.",
        "en": "Warehouse is empty. Use `/have import`.",
    },
    "you_have_no_pos": {
        "ru": "У тебя ещё нет POS-ов.",
        "en": "You don't have any POS yet.",
    },
    "ok_price": {
        "ru": "OK: **{resource}** = **{price} ISK/ед**",
        "en": "OK: **{resource}** = **{price} ISK/unit**",
    },
    "quantity_negative": {
        "ru": "Количество не может быть отрицательным. Проверь: {line}.",
        "en": "Quantity cannot be negative. Check: {line}.",
    },
    "only_text_channels": {
        "ru": "Доступны только текстовые каналы и треды.",
        "en": "Only text channels and threads are supported.",
    },
}


def normalize_language(language: Optional[str]) -> str:
    if not language:
        return DEFAULT_LANGUAGE
    language = language.lower()
    return language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


class TemplateTranslation:
    __slots__ = ("template_ru", "translations", "pattern", "formatter")

    def __init__(self, template_ru: str, translations: Dict[str, str]):
        self.template_ru = template_ru
        self.translations = translations
        self.formatter = Formatter()
        self.pattern = self._build_pattern(template_ru)

    def _build_pattern(self, template: str) -> re.Pattern[str]:
        parsed = list(self.formatter.parse(template))
        parts: List[str] = ["^"]
        seen_fields: Dict[str, str] = {}
        total = len(parsed)
        for idx, (literal, field_name, _format_spec, _conversion) in enumerate(parsed):
            if literal:
                parts.append(re.escape(literal))
            if field_name is None:
                continue
            key = field_name or f"field_{idx}"
            if key in seen_fields:
                parts.append(f"(?P={key})")
                continue
            seen_fields[key] = key
            remaining_literals = any(lit for lit, _, _, _ in parsed[idx + 1 :])
            quantifier = ".*" if not remaining_literals else ".*?"
            parts.append(f"(?P<{key}>{quantifier})")
        parts.append("$")
        return re.compile("".join(parts), re.DOTALL)

    def try_translate(self, text: str, language: str) -> Optional[str]:
        if language == DEFAULT_LANGUAGE:
            return text
        template_target = self.translations.get(language)
        if not template_target:
            return None
        match = self.pattern.fullmatch(text)
        if not match:
            return None
        groups = match.groupdict()
        try:
            return template_target.format(**groups)
        except Exception:
            return template_target


_template_translations: List[TemplateTranslation] = [
    TemplateTranslation(data["ru"], data)
    for data in TRANSLATIONS.values()
    if data.get("ru") and any(lang != DEFAULT_LANGUAGE for lang in data)
]

_machine_translation_cache: Dict[Tuple[str, str], str] = {}
_machine_translation_lock = threading.Lock()
_user_language_cache: Dict[int, str] = {}
_token_language_cache: Dict[str, str] = {}


def _contains_cyrillic(text: str) -> bool:
    return any("а" <= ch.lower() <= "я" or ch in "ёЁ" for ch in text)


def _run_machine_translation(text: str) -> str:
    from urllib import parse, request

    url = (
        "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q="
        + parse.quote(text)
    )
    with request.urlopen(url, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return "".join(part[0] for part in data[0])


async def machine_translate_text(text: str, language: str) -> str:
    key = (text, language)
    with _machine_translation_lock:
        cached = _machine_translation_cache.get(key)
    if cached is not None:
        return cached

    translated = text
    if language == "en" and text:
        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                translated = await loop.run_in_executor(None, _run_machine_translation, text)
            else:
                translated = _run_machine_translation(text)
        except Exception as exc:
            logger.debug("Machine translation failed for %s: %s", text[:60], exc)

    with _machine_translation_lock:
        _machine_translation_cache[key] = translated
    return translated


async def translate_text(text: Optional[str], language: str) -> Optional[str]:
    if not text or language == DEFAULT_LANGUAGE:
        return text
    if not isinstance(text, str):
        return text
    if not _contains_cyrillic(text):
        return text

    for tpl in _template_translations:
        result = tpl.try_translate(text, language)
        if result is not None and result != text:
            return result
    return await machine_translate_text(text, language)


def translate_for_language(language: str, key: str, **kwargs) -> str:
    entry = TRANSLATIONS.get(key, {})
    template = entry.get(language) or entry.get(DEFAULT_LANGUAGE) or ""
    return template.format(**kwargs)


__all__ = [
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "TRANSLATIONS",
    "TemplateTranslation",
    "normalize_language",
    "machine_translate_text",
    "translate_for_language",
    "translate_text",
]

