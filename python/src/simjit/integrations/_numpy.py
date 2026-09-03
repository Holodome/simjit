# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import datetime as _datetime
from typing import TYPE_CHECKING, SupportsInt, TypeAlias, cast

try:
    import numpy as np
except ImportError:
    np = None

if TYPE_CHECKING:
    import numpy as _np

    TimestampValue: TypeAlias = str | _datetime.datetime | _np.datetime64 | int
else:
    TimestampValue: TypeAlias = object

_UNIT_NS = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}


def is_available() -> bool:
    return np is not None


def require_numpy(feature: str):
    if np is None:
        raise RuntimeError(f"{feature} requires numpy to be installed")
    return np


def datetime64_to_int(value: object, unit: str) -> int | None:
    if np is None or not isinstance(value, np.datetime64):
        return None
    normalized = cast("_np.datetime64", value.astype(f"datetime64[{unit}]"))
    raw = cast(SupportsInt, normalized.astype(np.int64))
    return int(raw)


def _datetime_to_unit(value: _datetime.datetime, unit: str) -> int:
    epoch = _datetime.datetime(1970, 1, 1, tzinfo=_datetime.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=_datetime.timezone.utc)
    else:
        value = value.astimezone(_datetime.timezone.utc)
    delta = value - epoch
    ns = (
        delta.days * 86_400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )
    return ns // _UNIT_NS[unit]


def normalize_timestamp_literal(value: TimestampValue, unit: str) -> tuple[int, str | None]:
    if unit not in _UNIT_NS:
        raise ValueError(f"unsupported timestamp unit {unit!r}")

    if isinstance(value, int):
        return value, None

    converted = datetime64_to_int(value, unit)
    if converted is not None:
        return converted, None

    if isinstance(value, str):
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        value = _datetime.datetime.fromisoformat(value)

    if isinstance(value, _datetime.datetime):
        tz = None if value.tzinfo is None else "UTC"
        return _datetime_to_unit(value, unit), tz

    raise TypeError(f"unsupported timestamp literal type {type(value).__name__}")
