from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.utils import clamp


def _prepare_topic_reviews(base: pd.DataFrame) -> pd.DataFrame:
    low_score = base.loc[
        (base["is_low_score"])
        & (~base["is_empty_review"])
        & (~base["is_abnormal_text"])
        & (base["review_text_clean"].str.len() >= 8)
    ].copy()
    if len(low_score) > config.MAX_TOPIC_REVIEWS:
        low_score = (
            low_score.sort_values(["review_date", "helpful_count"], ascending=[False, False])
            .head(config.MAX_TOPIC_REVIEWS)
            .copy()
        )
    return low_score.reset_index(drop=True)


def _choose_topic_count(review_count: int) -> int:
    if review_count < 200:
        return min(4, max(2, review_count // 40))
    estimated = int(np.sqrt(review_count / 300))
    return int(clamp(estimated, 6, config.DEFAULT_TOPIC_COUNT))


def _build_topic_name(keywords: list[str]) -> str:
    selected = [keyword for keyword in keywords if keyword][:3]
    if not selected:
        return "其他问题"
    return " / ".join(selected)


def _growth_label(growth_value: float) -> str:
    if growth_value >= 0.4:
        return "快速上升"
    if growth_value >= 0.1:
        return "持续上升"
    if growth_value <= -0.2:
        return "明显回落"
    return "相对平稳"


def discover_topics(base: pd.DataFrame) -> pd.DataFrame:
    topic_reviews = _prepare_topic_reviews(base)
    if topic_reviews.empty:
        empty = pd.DataFrame(
            columns=[
                "topic_id",
                "topic_name",
                "core_keywords",
                "representative_review",
                "representative_title",
                "review_count",
                "average_rating",
                "unique_titles",
                "top_categories",
                "top_titles",
                "recent_7d_count",
                "recent_7d_growth_rate",
                "recent_30d_count",
                "recent_30d_growth_rate",
                "priority_score",
                "priority_level",
                "trend_label",
                "analysis_method",
            ]
        )
        empty.to_csv(config.TOPIC_SUMMARY_PATH, index=False, encoding="utf-8-sig")
        return empty

    topic_count = _choose_topic_count(len(topic_reviews))
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=max(5, int(len(topic_reviews) * 0.001)),
        max_features=5_000,
    )
    matrix = vectorizer.fit_transform(topic_reviews["review_text_clean"])
    model = NMF(
        n_components=min(topic_count, matrix.shape[1] if matrix.shape[1] > 0 else topic_count),
        init="nndsvda",
        random_state=config.DEFAULT_RANDOM_STATE,
        max_iter=400,
    )
    topic_weights = model.fit_transform(matrix)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    labels = topic_weights.argmax(axis=1)
    topic_reviews["topic_id"] = labels
    topic_reviews["topic_strength"] = topic_weights.max(axis=1)

    latest_date = topic_reviews["review_date"].max()
    recent_7d_start = latest_date - pd.Timedelta(days=config.RECENT_SHORT_DAYS - 1)
    previous_7d_start = recent_7d_start - pd.Timedelta(days=config.RECENT_SHORT_DAYS)
    recent_30d_start = latest_date - pd.Timedelta(days=config.RECENT_LONG_DAYS - 1)
    previous_30d_start = recent_30d_start - pd.Timedelta(days=config.RECENT_LONG_DAYS)

    topic_rows: list[dict[str, Any]] = []
    max_unique_titles = max(int(topic_reviews["title"].nunique()), 1)

    for topic_id in sorted(topic_reviews["topic_id"].unique()):
        topic_frame = topic_reviews.loc[topic_reviews["topic_id"] == topic_id].copy()
        if len(topic_frame) < config.MIN_REVIEWS_PER_TOPIC:
            continue

        term_weights = model.components_[topic_id]
        top_keyword_indexes = np.argsort(term_weights)[::-1][:8]
        keywords = feature_names[top_keyword_indexes].tolist()

        representative_row = topic_frame.sort_values(
            ["topic_strength", "helpful_count", "review_date"],
            ascending=[False, False, False],
        ).iloc[0]

        recent_7d_count = int((topic_frame["review_date"] >= recent_7d_start).sum())
        previous_7d_count = int(
            ((topic_frame["review_date"] >= previous_7d_start) & (topic_frame["review_date"] < recent_7d_start)).sum()
        )
        recent_30d_count = int((topic_frame["review_date"] >= recent_30d_start).sum())
        previous_30d_count = int(
            ((topic_frame["review_date"] >= previous_30d_start) & (topic_frame["review_date"] < recent_30d_start)).sum()
        )
        unique_titles = int(topic_frame["title"].nunique())
        review_count = int(len(topic_frame))
        average_rating = float(topic_frame["review_score"].mean())

        review_count_norm = review_count / max(len(topic_reviews), 1)
        severity_score = clamp((5 - average_rating) / 4, 0, 1)
        recent_30d_growth = 1.0 if previous_30d_count == 0 and recent_30d_count > 0 else (
            (recent_30d_count - previous_30d_count) / max(previous_30d_count, 1)
        )
        growth_norm = clamp((recent_30d_growth + 1) / 2, 0, 1)
        title_breadth_norm = unique_titles / max_unique_titles

        priority_score = round(
            (
                0.40 * review_count_norm
                + 0.25 * severity_score
                + 0.20 * growth_norm
                + 0.15 * title_breadth_norm
            )
            * 100,
            2,
        )
        if priority_score >= 65:
            priority_level = "高"
        elif priority_score >= 40:
            priority_level = "中"
        else:
            priority_level = "低"

        topic_rows.append(
            {
                "topic_id": int(topic_id),
                "topic_name": _build_topic_name(keywords),
                "core_keywords": " | ".join(keywords[:6]),
                "representative_review": representative_row["review_text"][:260],
                "representative_title": representative_row["title"],
                "review_count": review_count,
                "average_rating": round(average_rating, 4),
                "unique_titles": unique_titles,
                "top_categories": " | ".join(topic_frame["primary_category"].value_counts().head(3).index.tolist()),
                "top_titles": " | ".join(topic_frame["title"].value_counts().head(3).index.tolist()),
                "recent_7d_count": recent_7d_count,
                "recent_7d_growth_rate": round(
                    1.0 if previous_7d_count == 0 and recent_7d_count > 0 else (
                        (recent_7d_count - previous_7d_count) / max(previous_7d_count, 1)
                    ),
                    4,
                ),
                "recent_30d_count": recent_30d_count,
                "recent_30d_growth_rate": round(recent_30d_growth, 4),
                "priority_score": priority_score,
                "priority_level": priority_level,
                "trend_label": _growth_label(recent_30d_growth),
                "analysis_method": "tfidf_nmf",
            }
        )

    topic_summary = pd.DataFrame(topic_rows).sort_values(
        ["priority_score", "review_count"],
        ascending=[False, False],
    )
    topic_summary.to_csv(config.TOPIC_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return topic_summary
