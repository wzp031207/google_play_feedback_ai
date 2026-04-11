from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.data_loader import describe_available_fields, inspect_data_files, load_google_play_sources
from src.utils import (
    clean_text,
    ensure_directories,
    extract_primary_category,
    min_max_dates,
    normalize_review_text,
    safe_mode,
    safe_to_datetime,
    safe_to_numeric,
    text_word_count,
    write_json,
)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None  # type: ignore[assignment]


UNIFIED_INFO_COLUMNS = {
    "entity_id": "item_id",
    "title": "title",
    "description": "description",
    "store_score": "store_score",
    "ratings_count": "ratings_count",
    "downloads": "downloads",
    "content_rating": "content_rating",
    "section": "section",
    "categories": "categories",
    "version": "version",
    "country": "country",
    "language": "language",
    "price": "price",
    "developer": "developer",
    "updated_at": "updated_at",
}

UNIFIED_REVIEW_COLUMNS = {
    "entity_id": "item_id",
    "review_text": "review_text",
    "review_score": "review_score",
    "review_date": "review_date",
    "helpful_count": "helpful_count",
    "language": "review_language",
    "country": "review_country",
}


def _rename_from_field_map(dataframe: pd.DataFrame, field_map: dict[str, str | None], rename_map: dict[str, str]) -> pd.DataFrame:
    renamed = dataframe.copy()
    columns_to_rename = {
        actual_name: rename_map[logical_name]
        for logical_name, actual_name in field_map.items()
        if actual_name and logical_name in rename_map and actual_name in renamed.columns
    }
    return renamed.rename(columns=columns_to_rename)


def _rating_bucket(score: float) -> str:
    if pd.isna(score):
        return "未知"
    if score <= config.LOW_SCORE_MAX:
        return "低分(1-2星)"
    if score >= config.HIGH_SCORE_MIN:
        return "高分(4-5星)"
    return "中性(3星)"


def _rating_sentiment(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score <= config.LOW_SCORE_MAX:
        return "negative"
    if score >= config.HIGH_SCORE_MIN:
        return "positive"
    return "neutral"


def _analyze_text_sentiment(text_series: pd.Series) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None

    def single(text: str) -> tuple[float, str]:
        if not text:
            return 0.0, "unknown"
        if analyzer is None:
            return 0.0, "unknown"
        compound = analyzer.polarity_scores(text)["compound"]
        if compound >= 0.2:
            return compound, "positive"
        if compound <= -0.2:
            return compound, "negative"
        return compound, "neutral"

    scores = text_series.fillna("").map(single)
    return pd.DataFrame(scores.tolist(), columns=["text_sentiment_score", "text_sentiment_label"])


def _merge_sources() -> tuple[pd.DataFrame, dict[str, Any]]:
    inspection = inspect_data_files(config.RAW_DATA_DIR)
    schema_report = inspection["schema_report"]
    source_tables = load_google_play_sources(config.RAW_DATA_DIR)

    info_tables = {table.source_key: table for table in source_tables if table.role == "info"}
    review_tables = {table.source_key: table for table in source_tables if table.role == "reviews"}

    merged_frames: list[pd.DataFrame] = []
    source_summary: list[dict[str, Any]] = []

    for source_key, review_table in review_tables.items():
        info_table = info_tables.get(source_key)
        if info_table is None:
            continue

        info_frame = _rename_from_field_map(info_table.dataframe, info_table.field_map, UNIFIED_INFO_COLUMNS)
        review_frame = _rename_from_field_map(review_table.dataframe, review_table.field_map, UNIFIED_REVIEW_COLUMNS)

        if "item_id" not in info_frame.columns or "item_id" not in review_frame.columns:
            continue

        info_columns = [
            "item_id",
            "title",
            "description",
            "store_score",
            "ratings_count",
            "downloads",
            "content_rating",
            "section",
            "categories",
            "version",
            "country",
            "language",
            "price",
            "developer",
            "updated_at",
        ]
        available_info_columns = [column for column in info_columns if column in info_frame.columns]
        merged = review_frame.merge(
            info_frame[available_info_columns].drop_duplicates(subset=["item_id"]),
            on="item_id",
            how="left",
        )
        merged["entity_type"] = info_table.entity_type
        merged["source_key"] = source_key
        merged_frames.append(merged)
        source_summary.append(
            {
                "source_key": source_key,
                "entity_type": info_table.entity_type,
                "info_rows": int(len(info_table.dataframe)),
                "review_rows": int(len(review_table.dataframe)),
            }
        )

    if not merged_frames:
        raise ValueError("No valid info/review table pairs were found. Please check the raw data schema.")

    unified = pd.concat(merged_frames, ignore_index=True)
    return unified, {
        "schema_report": schema_report,
        "source_summary": source_summary,
    }


def build_analysis_base() -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_directories()
    raw_frame, metadata = _merge_sources()
    base = raw_frame.copy()

    empty_series = pd.Series([""] * len(base), index=base.index)
    base["title"] = base.get("title", empty_series).map(clean_text)
    base["description"] = base.get("description", empty_series).map(clean_text)
    base["review_text"] = base.get("review_text", empty_series).map(clean_text)
    base["review_text_clean"] = base["review_text"].map(normalize_review_text)
    base["review_text_length"] = base["review_text"].str.len().fillna(0).astype(int)
    base["review_word_count"] = base["review_text_clean"].map(text_word_count)
    base["is_empty_review"] = base["review_text_clean"].eq("")
    base["review_score"] = safe_to_numeric(base.get("review_score", pd.Series(index=base.index, dtype=float))).clip(1, 5)
    base["store_score"] = safe_to_numeric(base.get("store_score", pd.Series(index=base.index, dtype=float))).clip(0, 5)
    base["ratings_count"] = safe_to_numeric(base.get("ratings_count", pd.Series(index=base.index, dtype=float))).fillna(0).astype(int)
    base["downloads"] = safe_to_numeric(base.get("downloads", pd.Series(index=base.index, dtype=float))).fillna(0).astype(int)
    base["helpful_count"] = safe_to_numeric(base.get("helpful_count", pd.Series(index=base.index, dtype=float))).fillna(0).astype(int)
    base["review_date"] = safe_to_datetime(base.get("review_date", pd.Series(index=base.index, dtype="object")))
    base["updated_at"] = safe_to_datetime(base.get("updated_at", pd.Series(index=base.index, dtype="object")))
    base["categories"] = base.get("categories", empty_series).map(clean_text)
    base["primary_category"] = base["categories"].map(extract_primary_category)
    base["content_rating"] = base.get("content_rating", empty_series).map(clean_text)
    base["section"] = base.get("section", empty_series).map(clean_text)
    base["country"] = base.get("country", empty_series).map(clean_text)
    base["language"] = base.get("language", empty_series).map(clean_text)
    base["review_language"] = base.get("review_language", empty_series).map(clean_text)
    base["review_country"] = base.get("review_country", empty_series).map(clean_text)

    non_empty_lengths = base.loc[~base["is_empty_review"], "review_text_length"]
    max_reasonable_length = int(non_empty_lengths.quantile(0.995)) if not non_empty_lengths.empty else 0
    base["is_abnormal_text"] = (
        base["is_empty_review"]
        | (base["review_text_length"] < 3)
        | (base["review_text_length"] > max_reasonable_length)
    )
    base["rating_bucket"] = base["review_score"].map(_rating_bucket)
    base["rating_sentiment"] = base["review_score"].map(_rating_sentiment)
    base["is_low_score"] = base["review_score"] <= config.LOW_SCORE_MAX
    base["is_high_score"] = base["review_score"] >= config.HIGH_SCORE_MIN
    base["review_month"] = base["review_date"].dt.to_period("M").astype(str)
    base["review_week"] = base["review_date"].dt.to_period("W").astype(str)
    base["review_day"] = base["review_date"].dt.strftime("%Y-%m-%d")

    sentiment_frame = _analyze_text_sentiment(base["review_text"])
    base = pd.concat([base.reset_index(drop=True), sentiment_frame], axis=1)
    base["sentiment_mismatch_flag"] = (
        (base["rating_sentiment"] == "positive") & (base["text_sentiment_label"] == "negative")
    ) | (
        (base["rating_sentiment"] == "negative") & (base["text_sentiment_label"] == "positive")
    )

    dedupe_columns = ["source_key", "item_id", "review_date", "review_score", "review_text_clean"]
    base["is_duplicate_review"] = base.duplicated(subset=dedupe_columns, keep="first")
    duplicate_count = int(base["is_duplicate_review"].sum())
    base = base.loc[~base["is_duplicate_review"]].copy()

    base["review_id"] = (
        base["source_key"].astype(str)
        + "_"
        + base["item_id"].astype(str)
        + "_"
        + base["review_date"].fillna(pd.Timestamp("1970-01-01")).dt.strftime("%Y%m%d")
        + "_"
        + base.index.astype(str)
    )
    base = base.sort_values(["review_date", "helpful_count"], ascending=[False, False]).reset_index(drop=True)

    base.to_parquet(config.ANALYSIS_BASE_PATH, index=False)

    date_min, date_max = min_max_dates(base["review_date"])
    data_profile = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "project_name": "基于数据分析与 LLM 的 Google Play 用户反馈洞察及运营建议助手",
        "raw_sources": metadata["source_summary"],
        "total_rows_after_merge": int(len(raw_frame)),
        "analysis_rows": int(len(base)),
        "duplicate_reviews_removed": duplicate_count,
        "date_range": {"min": date_min, "max": date_max},
        "entity_type_distribution": base["entity_type"].value_counts().to_dict(),
        "rating_distribution": base["review_score"].value_counts(dropna=False).sort_index().to_dict(),
        "missing_ratio": base.isna().mean().round(4).to_dict(),
        "empty_review_ratio": round(float(base["is_empty_review"].mean()), 4),
        "abnormal_text_ratio": round(float(base["is_abnormal_text"].mean()), 4),
        "text_sentiment_available": SentimentIntensityAnalyzer is not None,
        "available_field_mapping": describe_available_fields(metadata["schema_report"]),
        "unsupported_expected_fields": {
            "version": not bool(describe_available_fields(metadata["schema_report"]).get("版本号")),
            "country": not bool(describe_available_fields(metadata["schema_report"]).get("国家字段")),
            "language": not bool(describe_available_fields(metadata["schema_report"]).get("语言字段")),
            "developer": not bool(describe_available_fields(metadata["schema_report"]).get("开发者")),
            "updated_at": not bool(describe_available_fields(metadata["schema_report"]).get("更新时间")),
            "price": not bool(describe_available_fields(metadata["schema_report"]).get("价格")),
        },
        "adaptation_notes": [
            "当前数据包含应用和游戏两类实体，已统一映射为 title / primary_category / entity_type 分析口径。",
            "当前数据未提供版本号，因此版本维度分析会输出空结果并在应用中提示字段缺失。",
            "当前数据未提供国家和语言字段，项目默认按全量评论做统一口径分析。",
        ],
        "key_columns": [
            "review_id",
            "entity_type",
            "item_id",
            "title",
            "primary_category",
            "review_text",
            "review_score",
            "review_date",
            "helpful_count",
            "downloads",
            "ratings_count",
            "text_sentiment_label",
        ],
    }
    write_json(config.DATA_PROFILE_PATH, data_profile)
    return base, data_profile
