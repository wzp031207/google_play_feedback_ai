from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.llm_engine import answer_question_with_context
from src.utils import clean_text


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _intent_based_chunks(question: str) -> list[dict[str, Any]]:
    question_text = clean_text(question).lower()
    chunks: list[dict[str, Any]] = []

    if any(keyword in question_text for keyword in ["主题", "问题", "差评", "低分"]):
        topic_path = config.ARTIFACT_DIR / "topic_summary.csv"
        if topic_path.exists():
            topic_df = pd.read_csv(topic_path).sort_values(
                ["recent_30d_count", "priority_score", "review_count"],
                ascending=[False, False, False],
            )
            for _, row in topic_df.head(8).iterrows():
                chunks.append(_row_to_chunk("topic_summary.csv", row.to_dict()))

    if any(keyword in question_text for keyword in ["下降", "下滑", "恶化", "变差"]):
        drop_path = config.AGGREGATES_DIR / "recent_rating_drop.csv"
        if drop_path.exists():
            drop_df = pd.read_csv(drop_path).sort_values(["rating_drop", "recent_review_count"], ascending=[False, False])
            for _, row in drop_df.head(8).iterrows():
                chunks.append(_row_to_chunk("recent_rating_drop.csv", row.to_dict()))

    if any(keyword in question_text for keyword in ["分类", "品类", "category"]):
        category_path = config.AGGREGATES_DIR / "category_summary.csv"
        if category_path.exists():
            category_df = pd.read_csv(category_path).sort_values(
                ["low_score_ratio", "review_count"], ascending=[False, False]
            )
            for _, row in category_df.head(8).iterrows():
                chunks.append(_row_to_chunk("category_summary.csv", row.to_dict()))

    if any(keyword in question_text for keyword in ["标题", "app", "game", "应用", "游戏"]):
        title_path = config.AGGREGATES_DIR / "title_summary.csv"
        if title_path.exists():
            title_df = pd.read_csv(title_path).sort_values(
                ["low_score_ratio", "review_count"], ascending=[False, False]
            )
            for _, row in title_df.head(8).iterrows():
                chunks.append(_row_to_chunk("title_summary.csv", row.to_dict()))

    if any(keyword in question_text for keyword in ["建议", "周报", "预警", "风险"]):
        cards_path = config.ARTIFACT_DIR / "insight_cards.json"
        if cards_path.exists():
            payload = json.loads(cards_path.read_text(encoding="utf-8"))
            for key, value in payload.items():
                text = "；".join(str(item) for item in value) if isinstance(value, list) else str(value)
                chunks.append({"source": "insight_cards.json", "text": f"{key}：{text}", "metadata": {"field": key}})

    return chunks


def _row_to_chunk(source_name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if source_name == "topic_summary.csv":
        text = (
            f"问题主题 {row.get('topic_name')}，核心关键词 {row.get('core_keywords')}，"
            f"评论量 {row.get('review_count')}，平均评分 {row.get('average_rating')}，"
            f"近7天评论量 {row.get('recent_7d_count')}，近30天评论量 {row.get('recent_30d_count')}，"
            f"近30天增长率 {row.get('recent_30d_growth_rate')}，优先级分数 {row.get('priority_score')}，"
            f"优先级 {row.get('priority_level')}，代表标题 {row.get('representative_title')}，"
            f"涉及品类 {row.get('top_categories')}。"
        )
    elif source_name == "recent_rating_drop.csv":
        text = (
            f"评分下滑标题 {row.get('title')}，品类 {row.get('primary_category')}，"
            f"近30天评论量 {row.get('recent_review_count')}，近30天平均评分 {row.get('recent_average_rating')}，"
            f"前30天平均评分 {row.get('previous_average_rating')}，评分下降 {row.get('rating_drop')}，"
            f"评论量变化率 {row.get('review_volume_change_rate')}。"
        )
    elif source_name == "category_summary.csv":
        text = (
            f"分类 {row.get('primary_category')}，实体类型 {row.get('entity_type')}，评论量 {row.get('review_count')}，"
            f"平均评分 {row.get('average_rating')}，低分评论占比 {row.get('low_score_ratio')}，"
            f"高分评论占比 {row.get('high_score_ratio')}，标题数 {row.get('title_count')}。"
        )
    elif source_name == "title_summary.csv":
        text = (
            f"标题 {row.get('title')}，实体类型 {row.get('entity_type')}，分类 {row.get('primary_category')}，"
            f"评论量 {row.get('review_count')}，平均评分 {row.get('average_rating')}，"
            f"低分评论占比 {row.get('low_score_ratio')}，最近30天评论量 {row.get('recent_30d_reviews')}。"
        )
    elif source_name in {"low_score_trend_weekly.csv", "high_score_trend_weekly.csv"}:
        focus = "低分评论占比" if "low_score" in source_name else "高分评论占比"
        text = (
            f"周度趋势日期 {row.get('period_start')}，评论量 {row.get('review_count')}，"
            f"{focus} {row.get('low_score_ratio', row.get('high_score_ratio'))}，平均评分 {row.get('average_rating')}。"
        )
    else:
        fragments = [f"{key}={value}" for key, value in row.items() if value not in ("", None)]
        text = "；".join(fragments)

    return {
        "source": source_name,
        "text": text,
        "metadata": dict(row),
    }


def build_retrieval_corpus(artifact_dir: Path | None = None) -> list[dict[str, Any]]:
    artifact_root = artifact_dir or config.ARTIFACT_DIR
    chunks: list[dict[str, Any]] = []

    topic_path = artifact_root / "topic_summary.csv"
    if topic_path.exists():
        topic_df = pd.read_csv(topic_path)
        for _, row in topic_df.head(50).iterrows():
            chunks.append(_row_to_chunk("topic_summary.csv", row.to_dict()))

    aggregate_dir = artifact_root / "aggregates"
    if aggregate_dir.exists():
        for path in sorted(aggregate_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            for _, row in frame.head(80).iterrows():
                chunks.append(_row_to_chunk(path.name, row.to_dict()))

    summary_path = artifact_root / "weekly_summary.md"
    if summary_path.exists():
        content = _load_text(summary_path)
        for index, block in enumerate([part.strip() for part in content.split("\n\n") if part.strip()], start=1):
            chunks.append(
                {
                    "source": "weekly_summary.md",
                    "text": block,
                    "metadata": {"section_index": index},
                }
            )

    cards_path = artifact_root / "insight_cards.json"
    if cards_path.exists():
        payload = json.loads(cards_path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            if isinstance(value, list):
                text = "；".join(str(item) for item in value)
            else:
                text = str(value)
            chunks.append(
                {
                    "source": "insight_cards.json",
                    "text": f"{key}：{text}",
                    "metadata": {"field": key},
                }
            )
    return chunks


def retrieve_context(question: str, top_k: int = 8) -> list[dict[str, Any]]:
    intent_chunks = _intent_based_chunks(question)
    if intent_chunks:
        unique_rows: list[dict[str, Any]] = []
        seen = set()
        for chunk in intent_chunks:
            key = (chunk["source"], chunk["text"])
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(
                {
                    "source": chunk["source"],
                    "score": 1.0,
                    "text": clean_text(chunk["text"])[:500],
                    "metadata": chunk.get("metadata", {}),
                }
            )
        return unique_rows[:top_k]

    corpus = build_retrieval_corpus()
    if not corpus:
        return []

    documents = [chunk["text"] for chunk in corpus]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(documents + [question])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked_indexes = similarities.argsort()[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for index in ranked_indexes:
        chunk = corpus[index]
        results.append(
            {
                "source": chunk["source"],
                "score": round(float(similarities[index]), 4),
                "text": clean_text(chunk["text"])[:500],
                "metadata": chunk.get("metadata", {}),
            }
        )
    return results


def _fallback_answer(question: str, retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return "基于当前检索结果无法直接下结论，因为还没有可用的分析产物。请先运行数据流水线。"

    summary_lines = [f"问题：{question}", "基于当前检索结果，优先可见的信息有："]
    for chunk in retrieved[:4]:
        summary_lines.append(f"- 来源 {chunk['source']}：{chunk['text']}")
    summary_lines.append("如果你想要更准确的回答，可以继续限定时间范围、品类或标题。")
    return "\n".join(summary_lines)


def answer_question(question: str) -> dict[str, Any]:
    retrieved = retrieve_context(question)
    fallback = _fallback_answer(question, retrieved)
    result = answer_question_with_context(question, retrieved, fallback)
    if result.get("source") == "llm":
        bullets = result.get("bullets", [])
        bullet_text = "\n".join(f"- {item}" for item in bullets if clean_text(item))
        cited = result.get("cited_sources", [])
        cited_text = "、".join(str(item) for item in cited if clean_text(item))
        answer_parts = [clean_text(result.get("answer", ""))]
        if bullet_text:
            answer_parts.append(bullet_text)
        if cited_text:
            answer_parts.append(f"引用来源：{cited_text}")
        result["text"] = "\n\n".join(part for part in answer_parts if part)
    else:
        result["text"] = fallback
    result["retrieved_context"] = retrieved
    return result
