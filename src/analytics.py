from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src import config
from src.utils import (
    clean_text,
    compact_number,
    ensure_directories,
    growth_rate,
    safe_divide,
    safe_mode,
)


def _save_aggregate(dataframe: pd.DataFrame, file_name: str) -> Path:
    output_path = config.AGGREGATES_DIR / file_name
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def _period_trend(base: pd.DataFrame, freq: str, label: str) -> pd.DataFrame:
    trend = base.dropna(subset=["review_date"]).copy()
    trend["period_start"] = trend["review_date"].dt.to_period(freq).dt.start_time
    aggregated = (
        trend.groupby("period_start", as_index=False)
        .agg(
            review_count=("review_id", "count"),
            average_rating=("review_score", "mean"),
            low_score_ratio=("is_low_score", "mean"),
            high_score_ratio=("is_high_score", "mean"),
        )
        .sort_values("period_start")
    )
    aggregated["period_type"] = label
    aggregated["average_rating"] = aggregated["average_rating"].round(4)
    aggregated["low_score_ratio"] = aggregated["low_score_ratio"].round(4)
    aggregated["high_score_ratio"] = aggregated["high_score_ratio"].round(4)
    return aggregated


def build_time_trends(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "review_volume_trend_daily.csv": _period_trend(base, "D", "day"),
        "review_volume_trend_weekly.csv": _period_trend(base, "W", "week"),
        "review_volume_trend_monthly.csv": _period_trend(base, "M", "month"),
    }


def build_category_summary(base: pd.DataFrame) -> pd.DataFrame:
    category_summary = (
        base.groupby(["entity_type", "primary_category"], as_index=False)
        .agg(
            review_count=("review_id", "count"),
            average_rating=("review_score", "mean"),
            low_score_ratio=("is_low_score", "mean"),
            high_score_ratio=("is_high_score", "mean"),
            average_helpful_count=("helpful_count", "mean"),
            title_count=("title", "nunique"),
        )
        .sort_values(["review_count", "average_rating"], ascending=[False, True])
    )
    category_summary["review_share"] = safe_divide(
        category_summary["review_count"],
        float(category_summary["review_count"].sum()),
    )
    category_summary["average_rating"] = category_summary["average_rating"].round(4)
    category_summary["low_score_ratio"] = category_summary["low_score_ratio"].round(4)
    category_summary["high_score_ratio"] = category_summary["high_score_ratio"].round(4)
    category_summary["average_helpful_count"] = category_summary["average_helpful_count"].round(2)
    category_summary["review_share"] = category_summary["review_share"].round(4)
    return category_summary


def build_title_summary(base: pd.DataFrame) -> pd.DataFrame:
    latest_date = base["review_date"].max()
    recent_cutoff = latest_date - pd.Timedelta(days=config.RECENT_LONG_DAYS - 1)
    title_summary = (
        base.groupby(["entity_type", "title", "primary_category"], as_index=False)
        .agg(
            review_count=("review_id", "count"),
            average_rating=("review_score", "mean"),
            low_score_ratio=("is_low_score", "mean"),
            high_score_ratio=("is_high_score", "mean"),
            average_helpful_count=("helpful_count", "mean"),
            latest_review_date=("review_date", "max"),
            recent_30d_reviews=("review_date", lambda values: int((values >= recent_cutoff).sum())),
        )
        .sort_values(["review_count", "average_rating"], ascending=[False, True])
    )
    title_summary["average_rating"] = title_summary["average_rating"].round(4)
    title_summary["low_score_ratio"] = title_summary["low_score_ratio"].round(4)
    title_summary["high_score_ratio"] = title_summary["high_score_ratio"].round(4)
    title_summary["average_helpful_count"] = title_summary["average_helpful_count"].round(2)
    title_summary["latest_review_date"] = title_summary["latest_review_date"].dt.strftime("%Y-%m-%d")
    return title_summary


def build_low_high_share_trends(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    weekly = base.dropna(subset=["review_date"]).copy()
    weekly["period_start"] = weekly["review_date"].dt.to_period("W").dt.start_time
    low_high = (
        weekly.groupby("period_start", as_index=False)
        .agg(
            review_count=("review_id", "count"),
            low_score_ratio=("is_low_score", "mean"),
            high_score_ratio=("is_high_score", "mean"),
            average_rating=("review_score", "mean"),
        )
        .sort_values("period_start")
    )
    return {
        "low_score_trend_weekly.csv": low_high[["period_start", "review_count", "low_score_ratio", "average_rating"]],
        "high_score_trend_weekly.csv": low_high[["period_start", "review_count", "high_score_ratio", "average_rating"]],
    }


def build_version_rating_change(base: pd.DataFrame) -> pd.DataFrame:
    if "version" not in base.columns or base["version"].fillna("").eq("").all():
        return pd.DataFrame(
            [
                {
                    "status": "unavailable",
                    "message": "当前数据不包含版本号字段，已跳过版本维度评分变化分析。",
                }
            ]
        )

    version_frame = base.loc[base["version"].fillna("").ne("")].copy()
    version_frame["period_start"] = version_frame["review_date"].dt.to_period("M").dt.start_time
    return (
        version_frame.groupby(["title", "version", "period_start"], as_index=False)
        .agg(review_count=("review_id", "count"), average_rating=("review_score", "mean"))
        .sort_values(["title", "period_start", "version"])
    )


def build_recent_rating_drop(base: pd.DataFrame) -> pd.DataFrame:
    dated = base.dropna(subset=["review_date"]).copy()
    latest_date = dated["review_date"].max()
    recent_start = latest_date - pd.Timedelta(days=config.RATING_DROP_WINDOW_DAYS - 1)
    previous_start = recent_start - pd.Timedelta(days=config.RATING_DROP_WINDOW_DAYS)

    recent = dated.loc[dated["review_date"] >= recent_start].copy()
    previous = dated.loc[
        (dated["review_date"] >= previous_start) & (dated["review_date"] < recent_start)
    ].copy()

    recent_group = (
        recent.groupby(["entity_type", "title", "primary_category"], as_index=False)
        .agg(recent_review_count=("review_id", "count"), recent_average_rating=("review_score", "mean"))
    )
    previous_group = (
        previous.groupby(["entity_type", "title", "primary_category"], as_index=False)
        .agg(previous_review_count=("review_id", "count"), previous_average_rating=("review_score", "mean"))
    )
    merged = recent_group.merge(
        previous_group,
        on=["entity_type", "title", "primary_category"],
        how="outer",
    ).fillna(0)
    merged["rating_drop"] = merged["previous_average_rating"] - merged["recent_average_rating"]
    merged["review_volume_change_rate"] = merged.apply(
        lambda row: growth_rate(row["recent_review_count"], row["previous_review_count"]),
        axis=1,
    )
    filtered = merged.loc[
        (merged["recent_review_count"] >= config.MIN_TITLE_REVIEWS_FOR_ALERT)
        | (merged["previous_review_count"] >= config.MIN_TITLE_REVIEWS_FOR_ALERT)
    ].copy()
    filtered = filtered.sort_values(
        ["rating_drop", "recent_review_count"],
        ascending=[False, False],
    )
    for column in ["recent_average_rating", "previous_average_rating", "rating_drop", "review_volume_change_rate"]:
        filtered[column] = filtered[column].round(4)
    return filtered


def _top_ngrams(texts: pd.Series, top_n: int = 20) -> list[tuple[str, int]]:
    clean_texts = texts.fillna("").astype(str)
    clean_texts = clean_texts.loc[clean_texts.str.len() > 0]
    if clean_texts.empty:
        return []
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=max(3, int(len(clean_texts) * 0.001)),
        max_features=3_000,
    )
    matrix = vectorizer.fit_transform(clean_texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    ranked_indexes = np.argsort(counts)[::-1][:top_n]
    return [(terms[index], int(counts[index])) for index in ranked_indexes if counts[index] > 0]


def build_keyword_outputs(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    low_score = base.loc[base["is_low_score"]].copy()
    low_score_keywords = pd.DataFrame(
        [{"keyword": keyword, "count": count} for keyword, count in _top_ngrams(low_score["review_text_clean"], top_n=config.MAX_KEYWORDS)]
    )

    rating_bucket_records: list[dict[str, Any]] = []
    for bucket, bucket_frame in base.groupby("rating_bucket"):
        for keyword, count in _top_ngrams(bucket_frame["review_text_clean"], top_n=12):
            rating_bucket_records.append(
                {
                    "rating_bucket": bucket,
                    "keyword": keyword,
                    "count": count,
                }
            )
    rating_bucket_keywords = pd.DataFrame(rating_bucket_records)

    low_score_length = (
        base.groupby("rating_bucket", as_index=False)
        .agg(
            average_text_length=("review_text_length", "mean"),
            median_text_length=("review_text_length", "median"),
            average_word_count=("review_word_count", "mean"),
            review_count=("review_id", "count"),
        )
        .sort_values("review_count", ascending=False)
    )
    low_score_length[["average_text_length", "median_text_length", "average_word_count"]] = low_score_length[
        ["average_text_length", "median_text_length", "average_word_count"]
    ].round(2)

    return {
        "low_score_keywords.csv": low_score_keywords,
        "keywords_by_rating_bucket.csv": rating_bucket_keywords,
        "text_length_by_rating_bucket.csv": low_score_length,
    }


def build_sentiment_outputs(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sentiment_distribution = (
        base.groupby(["rating_sentiment", "text_sentiment_label"], as_index=False)
        .agg(review_count=("review_id", "count"))
        .sort_values("review_count", ascending=False)
    )
    sentiment_distribution["share"] = sentiment_distribution["review_count"] / max(len(base), 1)
    sentiment_distribution["share"] = sentiment_distribution["share"].round(4)

    trend = base.dropna(subset=["review_date"]).copy()
    trend["period_start"] = trend["review_date"].dt.to_period("W").dt.start_time
    sentiment_trend = (
        trend.groupby(["period_start", "text_sentiment_label"], as_index=False)
        .agg(review_count=("review_id", "count"), average_rating=("review_score", "mean"))
        .sort_values(["period_start", "review_count"], ascending=[True, False])
    )
    sentiment_trend["average_rating"] = sentiment_trend["average_rating"].round(4)

    category_sentiment = (
        base.groupby(["entity_type", "primary_category"], as_index=False)
        .agg(
            review_count=("review_id", "count"),
            negative_text_share=("text_sentiment_label", lambda values: float((values == "negative").mean())),
            positive_text_share=("text_sentiment_label", lambda values: float((values == "positive").mean())),
            mismatch_share=("sentiment_mismatch_flag", "mean"),
            average_rating=("review_score", "mean"),
        )
        .sort_values(["negative_text_share", "review_count"], ascending=[False, False])
    )
    for column in ["negative_text_share", "positive_text_share", "mismatch_share", "average_rating"]:
        category_sentiment[column] = category_sentiment[column].round(4)

    title_sentiment = (
        base.groupby(["entity_type", "title"], as_index=False)
        .agg(
            review_count=("review_id", "count"),
            negative_text_share=("text_sentiment_label", lambda values: float((values == "negative").mean())),
            positive_text_share=("text_sentiment_label", lambda values: float((values == "positive").mean())),
            mismatch_share=("sentiment_mismatch_flag", "mean"),
            average_rating=("review_score", "mean"),
        )
        .sort_values(["negative_text_share", "review_count"], ascending=[False, False])
    )
    for column in ["negative_text_share", "positive_text_share", "mismatch_share", "average_rating"]:
        title_sentiment[column] = title_sentiment[column].round(4)

    anomalies = base.loc[base["sentiment_mismatch_flag"]].copy()
    anomalies = anomalies.sort_values(["helpful_count", "review_date"], ascending=[False, False]).head(300)
    anomalies = anomalies[
        [
            "review_id",
            "entity_type",
            "title",
            "primary_category",
            "review_date",
            "review_score",
            "rating_sentiment",
            "text_sentiment_label",
            "text_sentiment_score",
            "helpful_count",
            "review_text",
        ]
    ]
    anomalies["review_date"] = anomalies["review_date"].dt.strftime("%Y-%m-%d")
    anomalies["review_text"] = anomalies["review_text"].map(lambda value: clean_text(value)[:220])

    return {
        "sentiment_distribution.csv": sentiment_distribution,
        "sentiment_trend_weekly.csv": sentiment_trend,
        "category_sentiment_comparison.csv": category_sentiment,
        "title_sentiment_comparison.csv": title_sentiment,
        "sentiment_anomalies.csv": anomalies,
    }


def run_analytics(base: pd.DataFrame) -> dict[str, Path]:
    ensure_directories()
    outputs: dict[str, Path] = {}

    for file_name, dataframe in build_time_trends(base).items():
        outputs[file_name] = _save_aggregate(dataframe, file_name)

    outputs["category_summary.csv"] = _save_aggregate(build_category_summary(base), "category_summary.csv")
    outputs["title_summary.csv"] = _save_aggregate(build_title_summary(base), "title_summary.csv")
    outputs["version_rating_change.csv"] = _save_aggregate(build_version_rating_change(base), "version_rating_change.csv")
    outputs["recent_rating_drop.csv"] = _save_aggregate(build_recent_rating_drop(base), "recent_rating_drop.csv")

    for file_name, dataframe in build_low_high_share_trends(base).items():
        outputs[file_name] = _save_aggregate(dataframe, file_name)

    for file_name, dataframe in build_keyword_outputs(base).items():
        outputs[file_name] = _save_aggregate(dataframe, file_name)

    for file_name, dataframe in build_sentiment_outputs(base).items():
        outputs[file_name] = _save_aggregate(dataframe, file_name)

    return outputs


def summarize_aggregate_snapshot(base: pd.DataFrame) -> dict[str, Any]:
    latest_date = base["review_date"].max()
    recent_30d_start = latest_date - pd.Timedelta(days=config.RECENT_LONG_DAYS - 1)
    previous_30d_start = recent_30d_start - pd.Timedelta(days=config.RECENT_LONG_DAYS)

    recent_30d = base.loc[base["review_date"] >= recent_30d_start]
    previous_30d = base.loc[
        (base["review_date"] >= previous_30d_start) & (base["review_date"] < recent_30d_start)
    ]

    top_category_row = safe_mode(
        build_category_summary(base).head(1).to_dict(orient="records"),
        default={},
    )
    top_title_row = safe_mode(
        build_title_summary(base).head(1).to_dict(orient="records"),
        default={},
    )

    return {
        "latest_review_date": latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else None,
        "total_review_count": int(len(base)),
        "average_rating": round(float(base["review_score"].mean()), 4),
        "low_score_ratio": round(float(base["is_low_score"].mean()), 4),
        "high_score_ratio": round(float(base["is_high_score"].mean()), 4),
        "recent_30d_review_count": int(len(recent_30d)),
        "recent_30d_review_growth_rate": round(growth_rate(len(recent_30d), len(previous_30d)), 4),
        "recent_30d_average_rating": round(float(recent_30d["review_score"].mean()), 4) if not recent_30d.empty else 0.0,
        "top_category": top_category_row,
        "top_title": top_title_row,
        "entity_type_distribution": base["entity_type"].value_counts().to_dict(),
        "review_count_by_rating_bucket": base["rating_bucket"].value_counts().to_dict(),
        "downloads_by_entity_type": {
            key: compact_number(value)
            for key, value in base.groupby("entity_type")["downloads"].median().to_dict().items()
        },
    }
