from __future__ import annotations

from typing import Any

import pandas as pd

from src import config
from src.analytics import run_analytics, summarize_aggregate_snapshot
from src.preprocess import build_analysis_base
from src.topic_modeling import discover_topics
from src.llm_engine import generate_insight_cards
from src.utils import ensure_directories, write_json, write_markdown


def _load_top_rows(path: str, top_n: int = 5) -> list[dict[str, Any]]:
    dataframe = pd.read_csv(config.AGGREGATES_DIR / path)
    return dataframe.head(top_n).to_dict(orient="records")


def _build_insight_context(base: pd.DataFrame, topic_summary: pd.DataFrame) -> dict[str, Any]:
    snapshot = summarize_aggregate_snapshot(base)
    rating_drop_top = _load_top_rows("recent_rating_drop.csv", top_n=8)
    category_issues = _load_top_rows("category_summary.csv", top_n=8)
    title_summary = _load_top_rows("title_summary.csv", top_n=8)
    low_score_keywords = _load_top_rows("low_score_keywords.csv", top_n=12)
    sentiment_summary = _load_top_rows("sentiment_distribution.csv", top_n=12)
    return {
        "snapshot": snapshot,
        "top_topics": topic_summary.head(8).to_dict(orient="records"),
        "rating_drop_top": rating_drop_top,
        "category_issues": category_issues,
        "title_summary_top": title_summary,
        "low_score_keywords": low_score_keywords,
        "sentiment_summary": sentiment_summary,
    }


def _build_weekly_markdown(insight_cards: dict[str, Any]) -> str:
    def bullet_block(values: list[str]) -> str:
        return "\n".join(f"- {value}" for value in values if str(value).strip()) or "- 暂无"

    return f"""
# Google Play 用户反馈周报摘要

## 周报摘要
{insight_cards.get("weekly_summary", "暂无摘要。")}

## 重点问题总结
{bullet_block(insight_cards.get("key_issues", []))}

## 产品建议
{bullet_block(insight_cards.get("product_recommendations", []))}

## 运营建议
{bullet_block(insight_cards.get("operations_recommendations", []))}

## 风险预警
{bullet_block(insight_cards.get("risk_alerts", []))}

## 生成方式
- 来源：{insight_cards.get("source", "unknown")}
- 模型：{insight_cards.get("model", "rule_fallback")}
""".strip()


def run_full_pipeline() -> dict[str, Any]:
    ensure_directories()

    base, data_profile = build_analysis_base()
    aggregate_outputs = run_analytics(base)
    topic_summary = discover_topics(base)

    insight_context = _build_insight_context(base, topic_summary)
    insight_cards = generate_insight_cards(insight_context)
    write_json(config.INSIGHT_CARDS_PATH, insight_cards)
    write_markdown(config.WEEKLY_SUMMARY_PATH, _build_weekly_markdown(insight_cards))

    return {
        "analysis_base_path": str(config.ANALYSIS_BASE_PATH.relative_to(config.PROJECT_ROOT)),
        "data_profile_path": str(config.DATA_PROFILE_PATH.relative_to(config.PROJECT_ROOT)),
        "schema_report_path": str(config.SCHEMA_REPORT_PATH.relative_to(config.PROJECT_ROOT)),
        "topic_summary_path": str(config.TOPIC_SUMMARY_PATH.relative_to(config.PROJECT_ROOT)),
        "weekly_summary_path": str(config.WEEKLY_SUMMARY_PATH.relative_to(config.PROJECT_ROOT)),
        "insight_cards_path": str(config.INSIGHT_CARDS_PATH.relative_to(config.PROJECT_ROOT)),
        "aggregate_files": [str(path.relative_to(config.PROJECT_ROOT)) for path in aggregate_outputs.values()],
        "analysis_rows": int(len(base)),
        "data_profile": data_profile,
    }
