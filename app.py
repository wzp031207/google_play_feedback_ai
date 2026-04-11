from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.llm_engine import llm_status
from src.pipeline import run_full_pipeline
from src.qa_engine import answer_question
from src.utils import compact_number, percentage, score_to_points


st.set_page_config(
    page_title="Google Play 用户反馈洞察及运营建议助手",
    page_icon="📱",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_topic_summary() -> pd.DataFrame:
    return pd.read_csv(config.TOPIC_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_base_light() -> pd.DataFrame:
    columns = [
        "review_id",
        "entity_type",
        "item_id",
        "title",
        "primary_category",
        "review_date",
        "review_score",
        "is_low_score",
        "is_high_score",
        "helpful_count",
        "rating_bucket",
        "text_sentiment_label",
    ]
    frame = pd.read_parquet(config.ANALYSIS_BASE_PATH, columns=columns)
    frame["review_date"] = pd.to_datetime(frame["review_date"])
    return frame


@st.cache_data(show_spinner=False)
def load_keywords_by_rating_bucket() -> pd.DataFrame:
    path = config.AGGREGATES_DIR / "keywords_by_rating_bucket.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_recent_rating_drop() -> pd.DataFrame:
    path = config.AGGREGATES_DIR / "recent_rating_drop.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_sentiment_distribution() -> pd.DataFrame:
    path = config.AGGREGATES_DIR / "sentiment_distribution.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def clear_all_caches() -> None:
    st.cache_data.clear()


def artifacts_ready() -> bool:
    return all(path.exists() for path in config.REQUIRED_ARTIFACTS)


def ensure_pipeline_ready() -> None:
    if artifacts_ready():
        return

    st.title("Google Play 用户反馈洞察及运营建议助手")
    st.warning("当前还没有分析产物。请先构建数据底表、聚合结果、主题结果和 AI 洞察。")
    if st.button("开始构建项目分析产物", type="primary", use_container_width=True):
        with st.spinner("正在读取真实数据并生成分析结果，请稍候..."):
            run_full_pipeline()
        clear_all_caches()
        st.success("分析产物已生成完成。")
        st.rerun()
    st.stop()


def format_score(value: float) -> str:
    return f"{score_to_points(value):.2f}"


def compute_recent_growth(filtered_base: pd.DataFrame, days: int = 30) -> float:
    if filtered_base.empty:
        return 0.0
    latest_date = filtered_base["review_date"].max()
    recent_start = latest_date - pd.Timedelta(days=days - 1)
    previous_start = recent_start - pd.Timedelta(days=days)
    recent_count = int((filtered_base["review_date"] >= recent_start).sum())
    previous_count = int(
        (
            (filtered_base["review_date"] >= previous_start)
            & (filtered_base["review_date"] < recent_start)
        ).sum()
    )
    if previous_count == 0:
        return 1.0 if recent_count > 0 else 0.0
    return (recent_count - previous_count) / previous_count


def build_trend_frames(filtered_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    review_trend = (
        filtered_base.groupby(filtered_base["review_date"].dt.date, as_index=False)
        .agg(评论量=("review_id", "count"))
        .rename(columns={"review_date": "日期"})
    )
    rating_trend = (
        filtered_base.groupby(filtered_base["review_date"].dt.date, as_index=False)
        .agg(平均评分=("review_score", "mean"))
        .rename(columns={"review_date": "日期"})
    )
    rating_trend["平均评分"] = rating_trend["平均评分"].round(4)
    return review_trend, rating_trend


def build_category_frame(filtered_base: pd.DataFrame) -> pd.DataFrame:
    frame = (
        filtered_base.groupby(["entity_type", "primary_category"], as_index=False)
        .agg(
            评论量=("review_id", "count"),
            平均评分=("review_score", "mean"),
            低分评论占比=("is_low_score", "mean"),
            高分评论占比=("is_high_score", "mean"),
        )
        .sort_values(["评论量", "平均评分"], ascending=[False, True])
    )
    frame["平均评分"] = frame["平均评分"].round(4)
    frame["低分评论占比"] = frame["低分评论占比"].round(4)
    frame["高分评论占比"] = frame["高分评论占比"].round(4)
    return frame


def build_title_frame(filtered_base: pd.DataFrame) -> pd.DataFrame:
    frame = (
        filtered_base.groupby(["entity_type", "title", "primary_category"], as_index=False)
        .agg(
            评论量=("review_id", "count"),
            平均评分=("review_score", "mean"),
            低分评论占比=("is_low_score", "mean"),
            helpful均值=("helpful_count", "mean"),
        )
        .sort_values(["评论量", "平均评分"], ascending=[False, True])
    )
    frame["平均评分"] = frame["平均评分"].round(4)
    frame["低分评论占比"] = frame["低分评论占比"].round(4)
    frame["helpful均值"] = frame["helpful均值"].round(2)
    return frame


def filter_topics(topic_summary: pd.DataFrame, categories: list[str], titles: list[str]) -> pd.DataFrame:
    filtered = topic_summary.copy()
    if categories:
        pattern = "|".join(re.escape(item) for item in categories)
        filtered = filtered.loc[
            filtered["top_categories"].fillna("").str.contains(pattern, case=False, regex=True)
        ]
    if titles:
        pattern = "|".join(re.escape(item) for item in titles)
        filtered = filtered.loc[
            filtered["top_titles"].fillna("").str.contains(pattern, case=False, regex=True)
            | filtered["representative_title"].fillna("").str.contains(pattern, case=False, regex=True)
        ]
    return filtered


def plot_review_volume(trend_frame: pd.DataFrame) -> None:
    figure = px.line(
        trend_frame,
        x="日期",
        y="评论量",
        markers=True,
        color_discrete_sequence=["#0f766e"],
    )
    figure.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figure, use_container_width=True)


def plot_rating_trend(trend_frame: pd.DataFrame) -> None:
    figure = px.line(
        trend_frame,
        x="日期",
        y="平均评分",
        markers=True,
        color_discrete_sequence=["#1d4ed8"],
    )
    figure.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figure, use_container_width=True)


def plot_category_compare(category_frame: pd.DataFrame) -> None:
    display = category_frame.head(12).copy()
    figure = px.bar(
        display,
        x="primary_category",
        y="评论量",
        color="平均评分",
        hover_data=["低分评论占比", "高分评论占比", "entity_type"],
        color_continuous_scale="RdYlGn",
    )
    figure.update_layout(
        height=360,
        xaxis_title="分类",
        yaxis_title="评论量",
        margin=dict(l=10, r=10, t=30, b=80),
    )
    st.plotly_chart(figure, use_container_width=True)


def plot_title_compare(title_frame: pd.DataFrame) -> None:
    display = title_frame.head(25).copy()
    figure = px.scatter(
        display,
        x="评论量",
        y="平均评分",
        size="低分评论占比",
        color="entity_type",
        hover_name="title",
        hover_data=["primary_category", "helpful均值"],
        color_discrete_sequence=["#0f766e", "#dc2626"],
    )
    figure.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figure, use_container_width=True)


def render_insight_cards(insight_cards: dict[str, Any]) -> None:
    summary_col, product_col, ops_col, risk_col = st.columns(4)
    with summary_col:
        with st.container(border=True):
            st.markdown("#### 本周核心问题")
            st.write(insight_cards.get("weekly_summary", "暂无摘要"))
    with product_col:
        with st.container(border=True):
            st.markdown("#### 产品建议")
            for item in insight_cards.get("product_recommendations", []):
                st.markdown(f"- {item}")
    with ops_col:
        with st.container(border=True):
            st.markdown("#### 运营建议")
            for item in insight_cards.get("operations_recommendations", []):
                st.markdown(f"- {item}")
    with risk_col:
        with st.container(border=True):
            st.markdown("#### 风险预警")
            for item in insight_cards.get("risk_alerts", []):
                st.markdown(f"- {item}")


def render_overview_page(
    filtered_base: pd.DataFrame,
    topic_summary: pd.DataFrame,
    insight_cards: dict[str, Any],
    weekly_summary: str,
    data_profile: dict[str, Any],
    keywords_by_bucket: pd.DataFrame,
    recent_drop: pd.DataFrame,
) -> None:
    review_trend, rating_trend = build_trend_frames_safe(filtered_base)
    category_frame = build_category_frame(filtered_base)
    title_frame = build_title_frame(filtered_base)
    filtered_topics = filter_topics(topic_summary, st.session_state.get("selected_categories", []), st.session_state.get("selected_titles", []))

    total_reviews = int(len(filtered_base))
    average_rating = float(filtered_base["review_score"].mean()) if total_reviews else 0.0
    low_score_ratio = float(filtered_base["is_low_score"].mean()) if total_reviews else 0.0
    recent_growth = compute_recent_growth(filtered_base)
    high_priority_topic_count = int((filtered_topics["priority_level"] == "高").sum()) if not filtered_topics.empty else 0

    st.title("基于数据分析与 LLM 的 Google Play 用户反馈洞察及运营建议助手")
    st.caption(
        "围绕 Google Play 用户评论与评分数据，构建数据分析、问题主题发现、AI 总结、策略推荐与检索式问答一体化助手。"
    )

    with st.expander("项目说明与数据适配说明", expanded=False):
        st.markdown(
            f"""
- 当前底表评论总量：`{data_profile['analysis_rows']:,}`
- 数据实体类型：`{', '.join(data_profile['entity_type_distribution'].keys())}`
- 当前数据未提供版本号、国家、语言、开发者、更新时间等部分字段时，系统会自动跳过对应分析或给出字段缺失说明
- LLM 仅用于总结、建议与问答，不参与主统计计算
"""
        )

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
    kpi_1.metric("评论总量", f"{total_reviews:,}")
    kpi_2.metric("平均评分", f"{average_rating:.2f}")
    kpi_3.metric("低分评论占比", percentage(low_score_ratio, 1))
    kpi_4.metric("近30天评论增速", percentage(recent_growth, 1))
    kpi_5.metric("高优先级问题数", f"{high_priority_topic_count}")

    trend_col_1, trend_col_2 = st.columns(2)
    with trend_col_1:
        st.markdown("### 评论量趋势")
        plot_review_volume_safe(review_trend)
    with trend_col_2:
        st.markdown("### 平均评分趋势")
        plot_rating_trend_safe(rating_trend)

    compare_col_1, compare_col_2 = st.columns(2)
    with compare_col_1:
        st.markdown("### 分类对比")
        plot_category_compare(category_frame)
        st.dataframe(category_frame.head(15), hide_index=True, use_container_width=True)
    with compare_col_2:
        st.markdown("### 标题对比")
        plot_title_compare(title_frame)
        st.dataframe(title_frame.head(15), hide_index=True, use_container_width=True)

    st.markdown("### 低分问题主题")
    if filtered_topics.empty:
        st.info("当前筛选条件下没有匹配的主题结果，已建议扩大时间或标题范围。")
    else:
        topic_display = filtered_topics[
            [
                "topic_name",
                "core_keywords",
                "review_count",
                "average_rating",
                "priority_score",
                "priority_level",
                "trend_label",
                "top_categories",
                "representative_title",
            ]
        ].rename(
            columns={
                "topic_name": "主题名",
                "core_keywords": "核心关键词",
                "review_count": "评论量",
                "average_rating": "平均评分",
                "priority_score": "优先级分数",
                "priority_level": "优先级",
                "trend_label": "近30天趋势",
                "top_categories": "涉及品类",
                "representative_title": "代表标题",
            }
        )
        st.dataframe(topic_display.head(12), hide_index=True, use_container_width=True)

    insight_col_1, insight_col_2 = st.columns([1.3, 0.7])
    with insight_col_1:
        st.markdown("### AI 洞察卡片")
        render_insight_cards(insight_cards)
        with st.expander("查看周报摘要原文", expanded=False):
            st.markdown(weekly_summary)
    with insight_col_2:
        st.markdown("### 评分下滑预警")
        if recent_drop.empty:
            st.info("暂无评分下滑结果。")
        else:
            st.dataframe(
                recent_drop.head(12)[
                    [
                        "title",
                        "primary_category",
                        "recent_review_count",
                        "recent_average_rating",
                        "previous_average_rating",
                        "rating_drop",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("### 评分区间关键词差异")
        if keywords_by_bucket.empty:
            st.info("暂无关键词结果。")
        else:
            bucket_options = keywords_by_bucket["rating_bucket"].dropna().unique().tolist()
            selected_bucket = st.selectbox("评分区间", bucket_options)
            bucket_frame = keywords_by_bucket.loc[keywords_by_bucket["rating_bucket"] == selected_bucket].head(12)
            st.dataframe(bucket_frame, hide_index=True, use_container_width=True)


def render_topic_page(filtered_topics: pd.DataFrame) -> None:
    st.markdown("### 主题洞察")
    if filtered_topics.empty:
        st.info("当前筛选条件下没有匹配主题。")
        return

    selected_topic = st.selectbox(
        "选择主题",
        options=filtered_topics["topic_name"].tolist(),
    )
    topic_row = filtered_topics.loc[filtered_topics["topic_name"] == selected_topic].iloc[0]

    left_col, right_col = st.columns([0.45, 0.55])
    with left_col:
        st.metric("优先级分数", f"{topic_row['priority_score']:.2f}")
        st.metric("评论量", f"{int(topic_row['review_count']):,}")
        st.metric("平均评分", f"{float(topic_row['average_rating']):.2f}")
        st.metric("近30天评论数", f"{int(topic_row['recent_30d_count']):,}")
        st.metric("近30天增长率", percentage(float(topic_row['recent_30d_growth_rate']), 1))
    with right_col:
        st.markdown(f"**主题名**：{topic_row['topic_name']}")
        st.markdown(f"**核心关键词**：{topic_row['core_keywords']}")
        st.markdown(f"**涉及标题**：{topic_row['top_titles']}")
        st.markdown(f"**涉及品类**：{topic_row['top_categories']}")
        st.markdown(f"**趋势判断**：{topic_row['trend_label']}")
        st.markdown("**代表评论**")
        st.write(topic_row["representative_review"])

    st.dataframe(filtered_topics.head(20), hide_index=True, use_container_width=True)


def render_qa_page() -> None:
    st.markdown("### AI 分析问答助手")
    st.caption("问答基于已生成的聚合结果、主题结果、周报摘要和洞察卡片做检索，不直接对原始数据胡乱作答。")

    default_questions = [
        "最近30天差评最多的问题是什么？",
        "哪个 category 的评分下降最明显？",
        "给出本周优先处理建议",
        "高评分但负面文本的异常样本说明了什么？",
    ]
    selected_template = st.selectbox("示例问题", ["自定义问题"] + default_questions)
    question = st.text_area(
        "输入你的问题",
        value="" if selected_template == "自定义问题" else selected_template,
        height=120,
        placeholder="例如：最近30天差评最多的问题是什么？",
    )

    if st.button("生成回答", type="primary"):
        if not question.strip():
            st.warning("请先输入问题。")
            return
        with st.spinner("正在检索分析结果并生成回答..."):
            result = answer_question(question)
        st.markdown("#### 回答")
        st.write(result["text"])
        with st.expander("查看检索到的上下文", expanded=False):
            st.json(result.get("retrieved_context", []))


def render_quality_page(schema_report: dict[str, Any], data_profile: dict[str, Any], sentiment_distribution: pd.DataFrame) -> None:
    st.markdown("### 数据质量与字段适配")

    quality_col_1, quality_col_2 = st.columns(2)
    with quality_col_1:
        st.markdown("#### 数据质量摘要")
        quality_frame = pd.DataFrame(
            {
                "指标": [
                    "分析底表行数",
                    "去重删除评论数",
                    "空评论占比",
                    "异常文本占比",
                    "文本情绪分析可用",
                ],
                "值": [
                    data_profile["analysis_rows"],
                    data_profile["duplicate_reviews_removed"],
                    percentage(data_profile["empty_review_ratio"], 2),
                    percentage(data_profile["abnormal_text_ratio"], 2),
                    "是" if data_profile["text_sentiment_available"] else "否",
                ],
            }
        )
        st.dataframe(quality_frame, hide_index=True, use_container_width=True)

        st.markdown("#### 字段可用性")
        availability_frame = pd.DataFrame(
            [
                {"字段用途": key, "检测结果": "、".join(value) if value else "未发现"}
                for key, value in data_profile["available_field_mapping"].items()
            ]
        )
        st.dataframe(availability_frame, hide_index=True, use_container_width=True)
    with quality_col_2:
        st.markdown("#### 原始文件 Schema")
        schema_rows = []
        for file_info in schema_report["files"]:
            schema_rows.append(
                {
                    "文件": file_info["file_name"],
                    "行数": file_info["row_count"],
                    "列数": file_info["column_count"],
                    "角色": file_info["role"],
                    "实体类型": file_info["entity_type"],
                    "列名": ", ".join(file_info["columns"]),
                }
            )
        st.dataframe(pd.DataFrame(schema_rows), hide_index=True, use_container_width=True)

        st.markdown("#### 情绪分布")
        if sentiment_distribution.empty:
            st.info("暂无情绪分析结果。")
        else:
            st.dataframe(sentiment_distribution, hide_index=True, use_container_width=True)


def main() -> None:
    ensure_pipeline_ready()

    schema_report = load_json(config.SCHEMA_REPORT_PATH)
    data_profile = load_json(config.DATA_PROFILE_PATH)
    insight_cards = load_json(config.INSIGHT_CARDS_PATH)
    weekly_summary = load_markdown(config.WEEKLY_SUMMARY_PATH)
    base = load_base_light()
    topic_summary = load_topic_summary()
    keywords_by_bucket = load_keywords_by_rating_bucket()
    recent_drop = load_recent_rating_drop()
    sentiment_distribution = load_sentiment_distribution()

    with st.sidebar:
        st.markdown("## 过滤器")
        status = llm_status()
        if status["available"]:
            st.success(status["message"])
        else:
            st.warning(status["message"])

        if st.button("重新构建分析产物", use_container_width=True):
            with st.spinner("正在重新生成全部分析产物..."):
                run_full_pipeline()
            clear_all_caches()
            st.success("分析产物已更新。")
            st.rerun()

        entity_options = sorted(base["entity_type"].dropna().unique().tolist())
        selected_entities = st.multiselect("实体类型", entity_options, default=entity_options, key="selected_entities")

        filtered_for_category = base.loc[base["entity_type"].isin(selected_entities)] if selected_entities else base.iloc[0:0]
        category_options = sorted(filtered_for_category["primary_category"].dropna().unique().tolist())
        selected_categories = st.multiselect("分类", category_options, default=category_options, key="selected_categories")

        filtered_for_title = filtered_for_category.loc[
            filtered_for_category["primary_category"].isin(selected_categories)
        ] if selected_categories else filtered_for_category
        top_titles = (
            filtered_for_title["title"].value_counts().head(80).index.tolist()
            if not filtered_for_title.empty
            else []
        )
        selected_titles = st.multiselect("标题", top_titles, default=[], key="selected_titles")

        min_date = base["review_date"].min().date()
        max_date = base["review_date"].max().date()
        default_start = max(min_date, max_date - timedelta(days=180))
        selected_date_range = st.slider(
            "评论时间范围",
            min_value=min_date,
            max_value=max_date,
            value=(default_start, max_date),
        )

        page = st.radio("页面", ["首页总览", "问题主题", "问答助手", "数据质量"], label_visibility="collapsed")

    filtered_base = base.loc[base["entity_type"].isin(selected_entities)].copy()
    if selected_categories:
        filtered_base = filtered_base.loc[filtered_base["primary_category"].isin(selected_categories)].copy()
    if selected_titles:
        filtered_base = filtered_base.loc[filtered_base["title"].isin(selected_titles)].copy()
    start_date, end_date = selected_date_range
    filtered_base = filtered_base.loc[
        (filtered_base["review_date"].dt.date >= start_date)
        & (filtered_base["review_date"].dt.date <= end_date)
    ].copy()

    filtered_topics = filter_topics(topic_summary, selected_categories, selected_titles)

    if page == "首页总览":
        render_overview_page(
            filtered_base=filtered_base,
            topic_summary=topic_summary,
            insight_cards=insight_cards,
            weekly_summary=weekly_summary,
            data_profile=data_profile,
            keywords_by_bucket=keywords_by_bucket,
            recent_drop=recent_drop,
        )
    elif page == "问题主题":
        render_topic_page(filtered_topics)
    elif page == "问答助手":
        render_qa_page()
    else:
        render_quality_page(schema_report, data_profile, sentiment_distribution)

def build_trend_frames_safe(filtered_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_with_trend_date = filtered_base.assign(trend_date=filtered_base["review_date"].dt.date)
    review_trend = (
        base_with_trend_date.groupby("trend_date", as_index=False)
        .agg(review_count=("review_id", "count"))
        .sort_values("trend_date")
    )
    rating_trend = (
        base_with_trend_date.groupby("trend_date", as_index=False)
        .agg(avg_rating=("review_score", "mean"))
        .sort_values("trend_date")
    )
    rating_trend["avg_rating"] = rating_trend["avg_rating"].round(4)
    return review_trend, rating_trend


def plot_review_volume_safe(trend_frame: pd.DataFrame) -> None:
    figure = px.line(
        trend_frame,
        x="trend_date",
        y="review_count",
        markers=True,
        color_discrete_sequence=["#0f766e"],
    )
    figure.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figure, use_container_width=True)


def plot_rating_trend_safe(trend_frame: pd.DataFrame) -> None:
    figure = px.line(
        trend_frame,
        x="trend_date",
        y="avg_rating",
        markers=True,
        color_discrete_sequence=["#1d4ed8"],
    )
    figure.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figure, use_container_width=True)


if __name__ == "__main__":
    main()
