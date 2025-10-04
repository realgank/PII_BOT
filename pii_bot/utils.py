"""Shared utility helpers for the PII bot."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def now_utc_iso() -> str:
    """Return the current UTC time in ISO format without microseconds."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_dt(dt: datetime) -> str:
    """Format a datetime object as a human readable UTC string."""

    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_ts(ts: Optional[str]) -> str:
    """Format an ISO timestamp string into a human readable form."""

    if not ts:
        return "?"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    return format_dt(dt)


def parse_iso_dt(ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO formatted timestamp into a timezone-aware datetime."""

    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clean_resource_name(resource: str) -> str:
    """Normalise resource names by trimming whitespace and quotes."""

    return (resource or "").strip().strip('"').strip("'").strip()


__all__ = [
    "clean_resource_name",
    "format_dt",
    "format_ts",
    "now_utc_iso",
    "parse_iso_dt",
]
