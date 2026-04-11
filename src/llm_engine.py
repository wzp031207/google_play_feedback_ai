from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency
    OpenAI = None  # type: ignore[assignment]


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
DEFAULT_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "none")
DEFAULT_TEXT_VERBOSITY = os.getenv("OPENAI_TEXT_VERBOSITY", "medium")


def llm_status() -> dict[str, Any]:
    if OpenAI is None:
        return {
            "available": False,
            "message": "未安装 openai 依赖，将使用规则化摘要。",
            "model": None,
        }
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "available": False,
            "message": "未检测到 OPENAI_API_KEY，将使用规则化摘要。",
            "model": DEFAULT_MODEL,
        }
    return {
        "available": True,
        "message": f"已启用 OpenAI 模型：{DEFAULT_MODEL}",
        "model": DEFAULT_MODEL,
    }


def _make_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai 依赖不可用。")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY。")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def call_llm_json(instructions: str, prompt: str, max_output_tokens: int = 700) -> dict[str, Any]:
    status = llm_status()
    if not status["available"]:
        return {"ok": False, "error": status["message"], "data": None, "model": status["model"]}

    try:
        client = _make_client()
        response = client.responses.create(
            model=DEFAULT_MODEL,
            instructions=instructions,
            input=prompt,
            reasoning={"effort": DEFAULT_REASONING_EFFORT},
            text={"verbosity": DEFAULT_TEXT_VERBOSITY},
            max_output_tokens=max_output_tokens,
        )
        output_text = (getattr(response, "output_text", "") or "").strip()
        if not output_text:
            raise RuntimeError("LLM 未返回有效文本。")
        return {
            "ok": True,
            "data": _extract_json_object(output_text),
            "model": DEFAULT_MODEL,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        return {"ok": False, "data": None, "model": DEFAULT_MODEL, "error": str(exc)}


def build_rule_summary(context: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = context["snapshot"]
    top_topics = context.get("top_topics", [])
    rating_drop = context.get("rating_drop_top", [])
    category_issues = context.get("category_issues", [])

    recent_growth = snapshot.get("recent_30d_review_growth_rate", 0.0)
    if recent_growth >= 0.15:
        growth_text = "近30天评论量持续增长，用户反馈活跃度明显提升。"
    elif recent_growth <= -0.10:
        growth_text = "近30天评论量回落，近期反馈热度有所下降。"
    else:
        growth_text = "近30天评论量整体相对平稳。"

    top_topic_text = "；".join(
        f"{topic['topic_name']}（量级{topic['review_count']}，优先级{topic['priority_score']}）"
        for topic in top_topics[:3]
    ) or "暂无明显集中问题。"
    drop_text = "；".join(
        f"{row['title']}（近30天评分{row['recent_average_rating']}，较前期下降{row['rating_drop']}）"
        for row in rating_drop[:3]
    ) or "近期未发现显著评分下滑标题。"
    category_text = "；".join(
        f"{row['primary_category']}（均分{row['average_rating']}，低分占比{row['low_score_ratio']:.1%}）"
        for row in category_issues[:3]
    ) or "暂无明显低评分品类。"

    return {
        "weekly_summary": (
            f"当前累计评论量为 {snapshot['total_review_count']:,}，平均评分 {snapshot['average_rating']:.2f}，"
            f"低分评论占比 {snapshot['low_score_ratio']:.1%}。{growth_text}"
        ),
        "key_issues": [
            f"低分主题主要集中在：{top_topic_text}",
            f"近期评分下滑标题主要包括：{drop_text}",
            f"低评分品类重点关注：{category_text}",
        ],
        "product_recommendations": [
            "优先处理高优先级主题对应的核心体验问题，尤其是最近30天增长较快的差评主题。",
            "针对评分下滑最明显的标题，结合代表评论回看最近版本或功能改动。",
            "将高 helpful_count 的差评样本纳入缺陷复盘，减少重复问题持续累积。",
        ],
        "operations_recommendations": [
            "针对高潜在舆情风险标题，优先安排评论区回应和版本说明补充。",
            "对低分主题集中品类建立周度监控，跟踪差评占比和主题增长率。",
            "围绕高分评论中的正向关键词提炼宣传文案，平衡产品口碑表达。",
        ],
        "risk_alerts": [
            "如果高优先级主题持续扩散，可能进一步拖累标题均分与自然转化。",
            "如果评分下滑标题集中在同一品类，可能意味着该品类存在共性体验问题。",
        ],
        "source": "rule_fallback",
    }


def generate_insight_cards(context: Mapping[str, Any]) -> dict[str, Any]:
    instructions = (
        "你是一名 Google Play 用户反馈分析师。"
        "只能基于给定统计结果、主题结果、评分变化结果和摘要上下文输出业务洞察。"
        "不要虚构任何未提供的数据。"
        "请仅返回 JSON，对应字段必须为：weekly_summary, key_issues, product_recommendations, operations_recommendations, risk_alerts。"
        "其中 key_issues, product_recommendations, operations_recommendations, risk_alerts 必须为中文字符串数组。"
    )
    prompt = "请基于以下真实分析结果生成业务周报摘要与建议：\n" + _json_text(context)
    result = call_llm_json(instructions, prompt, max_output_tokens=900)
    if result["ok"]:
        payload = result["data"]
        payload["source"] = "llm"
        payload["model"] = result["model"]
        return payload
    fallback = build_rule_summary(context)
    fallback["llm_error"] = result["error"]
    fallback["model"] = result["model"]
    return fallback


def answer_question_with_context(
    question: str,
    retrieved_chunks: Sequence[Mapping[str, Any]],
    fallback_answer: str,
) -> dict[str, Any]:
    instructions = (
        "你是一名数据分析助手。"
        "只能基于提供的检索结果回答问题，不要编造结论。"
        "回答风格像数据分析师，优先引用真实统计结果。"
        "如果上下文不足，必须明确说明“基于当前检索结果无法直接下结论”。"
        "请仅返回 JSON，字段必须为：answer, bullets, cited_sources。"
        "其中 bullets 与 cited_sources 必须是数组。"
    )
    prompt = (
        f"用户问题：{question}\n\n"
        f"检索结果：\n{_json_text(retrieved_chunks)}"
    )
    result = call_llm_json(instructions, prompt, max_output_tokens=700)
    if result["ok"]:
        data = result["data"]
        data["source"] = "llm"
        data["model"] = result["model"]
        return data
    return {
        "answer": fallback_answer,
        "bullets": [],
        "cited_sources": [chunk.get("source", "") for chunk in retrieved_chunks[:3]],
        "source": "rule_fallback",
        "model": result["model"],
        "llm_error": result["error"],
    }
