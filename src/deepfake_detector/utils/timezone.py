from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


if ZoneInfo is not None:
    IST = ZoneInfo("Asia/Kolkata")
else:  # pragma: no cover
    IST = timezone(timedelta(hours=5, minutes=30))


def now_ist_iso() -> str:
    return datetime.now(IST).replace(microsecond=0).isoformat()
