from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src import config


_GENERIC_CATEGORY_TAGS = {
    "single player",
    "multiplayer",
    "offline",
    "online",
    "stylized",
    "competitive multiplayer",
    "casual",
    "abstract",
    "abstract graphics",
    "business tools",
    "budgeting tools",
    "offline play",
}


def ensure_directories() -> None:
    """Create project directories if they do not exist."""
    for path in [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.ARTIFACT_DIR,
        config.AGGREGATES_DIR,
        config.FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def standardize_column_name(column_name: str) -> str:
    column_name = column_name.strip().lower()
    column_name = re.sub(r"[^a-z0-9]+", "_", column_name)
    return re.sub(r"_+", "_", column_name).strip("_")


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: standardize_column_name(str(column)) for column in dataframe.columns}
    return dataframe.rename(columns=renamed)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_review_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_word_count(text: str) -> int:
    return len([token for token in text.split(" ") if token])


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def safe_divide(numerator: Any, denominator: Any) -> Any:
    if isinstance(numerator, pd.Series):
        if denominator in (0, 0.0) or pd.isna(denominator):
            return pd.Series(np.zeros(len(numerator)), index=numerator.index)
        return numerator / float(denominator)
    return safe_ratio(float(numerator), float(denominator))


def growth_rate(current_value: float, previous_value: float) -> float:
    if previous_value <= 0:
        return 1.0 if current_value > 0 else 0.0
    return (float(current_value) - float(previous_value)) / float(previous_value)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def json_ready_records(dataframe: pd.DataFrame, limit: int = 3) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    preview = dataframe.head(limit).copy()
    for record in preview.to_dict(orient="records"):
        clean_record: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (pd.Timestamp, datetime)):
                clean_record[key] = value.isoformat()
            elif isinstance(value, (np.integer, np.floating)):
                clean_record[key] = value.item()
            else:
                clean_record[key] = clean_text(value)[:240]
        records.append(clean_record)
    return records


def flatten_categories(raw_value: Any) -> list[str]:
    text = clean_text(raw_value)
    if not text:
        return []
    categories = []
    for item in text.split(","):
        item = clean_text(item)
        if not item:
            continue
        if item.startswith("#"):
            continue
        categories.append(item)
    return categories


def extract_primary_category(raw_value: Any) -> str:
    categories = flatten_categories(raw_value)
    if not categories:
        return "Unknown"
    for category in categories:
        lowered = category.lower()
        if lowered in _GENERIC_CATEGORY_TAGS:
            continue
        return category
    return categories[0]


def min_max_dates(series: pd.Series) -> tuple[str | None, str | None]:
    valid = series.dropna()
    if valid.empty:
        return None, None
    return valid.min().strftime("%Y-%m-%d"), valid.max().strftime("%Y-%m-%d")


def percentage(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def score_to_points(value: float, digits: int = 2) -> float:
    return round(float(value) * 100, digits)


def safe_mode(values: Iterable[Any], default: Any = "") -> Any:
    series = pd.Series(list(values))
    if series.empty:
        return default
    modes = series.mode(dropna=True)
    if modes.empty:
        return default
    return modes.iloc[0]


def compact_number(value: float | int) -> str:
    numeric = float(value)
    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:.1f}M"
    if abs(numeric) >= 1_000:
        return f"{numeric / 1_000:.1f}K"
    if math.isclose(numeric, round(numeric)):
        return str(int(round(numeric)))
    return f"{numeric:.2f}"
