from __future__ import annotations

from typing import Iterable, Optional


def find_column(candidates: Iterable[str], available: Iterable[str]) -> Optional[str]:
    """Return the first matching column name (case-insensitive) from candidates.

    Args:
        candidates: List of preferred column names to look for.
        available: Iterable of available column names in the dataset.

    Returns:
        The actual column name if found, else None.
    """
    lower_map = {c.lower(): c for c in available}
    for name in candidates:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    return None

