# 基于数据分析与 LLM 的 Google Play 用户反馈洞察及运营建议助手

这是一个围绕 Google Play 应用与游戏评论数据构建的作品集项目。系统会自动读取真实评论与元数据文件，完成数据清洗、反馈分析、低分主题发现、AI 业务总结、运营建议生成与检索式问答，适合面试演示和简历项目展示。

结构化统计与主题结果由数据分析流水线生成，LLM 不参与主统计计算，只负责：
- 周报摘要生成
- 产品与运营建议生成
- 基于分析结果的自然语言问答

## 项目目标

项目帮助业务方快速识别：
- 哪些 app / game / category 的用户反馈在恶化
- 差评的核心问题是什么
- 不同时间、品类、标题、评分区间下，用户关注点有什么差异
- 哪些问题优先级最高，应该先处理
- AI 是否可以自动生成业务摘要、风险提示和运营建议

## 真实数据检查结果

当前数据包中已识别出 4 个主文件：

- `data/raw/apps_info.csv`：217 行，9 列
- `data/raw/apps_reviews.csv`：466,700 行，5 列
- `data/raw/games_info.csv`：335 行，9 列
- `data/raw/games_reviews.csv`：583,170 行，5 列

主要字段适配如下：

- 评论文本：`review_text`
- 星级评分：`review_score`
- 评论时间：`review_date`
- 应用/游戏标题：`app_name` / `game_name`
- 分类：`categories`
- 元数据：`score`、`ratings_count`、`downloads`、`content_rating`、`section`

当前数据中未发现以下字段，因此系统会自动跳过或降级对应分析：

- 版本号
- 国家
- 语言
- 开发者
- 更新时间
- 价格

完整 schema 检查结果会自动写入：

- `artifacts/schema_report.json`
- `artifacts/data_profile.json`

## 项目结构

```text
recharge_ai_project/
├─ app.py
├─ run_pipeline.py
├─ requirements.txt
├─ README.md
├─ 简历项目描述.md
├─ 面试讲解版项目说明.md
├─ data/
│  └─ raw/
│     ├─ apps_info.csv
│     ├─ apps_reviews.csv
│     ├─ games_info.csv
│     └─ games_reviews.csv
├─ artifacts/
│  ├─ analysis_base.parquet
│  ├─ data_profile.json
│  ├─ schema_report.json
│  ├─ topic_summary.csv
│  ├─ weekly_summary.md
│  ├─ insight_cards.json
│  └─ aggregates/
└─ src/
   ├─ config.py
   ├─ data_loader.py
   ├─ preprocess.py
   ├─ analytics.py
   ├─ topic_modeling.py
   ├─ llm_engine.py
   ├─ qa_engine.py
   ├─ pipeline.py
   └─ utils.py
```

## 功能模块

### 1. 数据清洗与预处理

`src/preprocess.py` 会完成：

- 读取原始 `info` / `reviews` 表
- 自动识别字段角色并统一映射
- 缺失值处理
- 重复评论处理
- 时间字段解析
- 评分字段规范化
- 文本长度、空文本、异常文本检测
- 基于评分的满意度标签
- 可选的文本情绪分析（安装 `vaderSentiment` 时自动启用）

底表输出：

- `artifacts/analysis_base.parquet`

质量报告输出：

- `artifacts/data_profile.json`

### 2. 评论分析与经营洞察

`src/analytics.py` 会自动生成：

- 评论量趋势（日/周/月）
- 平均评分趋势
- 各 category 的评论量与评分分布
- 各 title 的评论量、平均评分、低分评论占比
- 低分评论占比趋势
- 高分评论占比趋势
- 最近评分显著下滑的标题排行
- 高频低分关键词
- 不同评分区间的文本长度与关键词差异
- 情绪分布、情绪趋势、品类情绪对比、标题情绪对比
- 情绪与评分不一致的异常样本

聚合结果输出到：

- `artifacts/aggregates/*.csv`

### 3. 问题主题发现

`src/topic_modeling.py` 基于低分评论执行主题归因，当前默认采用稳妥的 `TF-IDF + NMF` 方案。

每个主题会输出：

- 主题名
- 核心关键词
- 代表评论
- 评论量
- 平均评分
- 最近 7 天 / 30 天变化
- 涉及标题数
- 优先级分数

输出文件：

- `artifacts/topic_summary.csv`

### 4. LLM 周报与策略建议

`src/llm_engine.py` 会根据真实聚合结果、主题结果与评分变化结果生成：

- 周报摘要
- 重点问题总结
- 产品建议
- 运营建议
- 风险预警

当存在 `OPENAI_API_KEY` 时：

- 使用 OpenAI 生成更自然的业务总结

当不存在 `OPENAI_API_KEY` 时：

- 自动回退到模板化规则摘要
- 保证项目依然可运行

输出文件：

- `artifacts/weekly_summary.md`
- `artifacts/insight_cards.json`

### 5. 检索式问答助手

`src/qa_engine.py` 不是普通聊天机器人，而是先检索以下分析结果，再组织回答：

- `topic_summary.csv`
- `artifacts/aggregates/*.csv`
- `weekly_summary.md`
- `insight_cards.json`

因此问题回答会尽量引用真实统计结果，而不是空泛生成。

### 6. Streamlit 可视化看板

`app.py` 提供中文交互界面，包含：

- 项目说明与数据适配说明
- KPI 卡片
- 评论量与评分趋势图
- category / title 分析
- 低分问题主题表
- AI 洞察卡片
- 评分下滑预警
- 评分区间关键词差异
- 问答助手
- 数据质量与 schema 检查页

## 安装与运行

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2. 运行分析流水线

```bash
python run_pipeline.py
```

首次运行会生成全部产物。由于底表评论量超过 100 万条，完整流水线可能需要几分钟。

### 3. 启动可视化应用

```bash
streamlit run app.py
```

## 主要输出文件

### 核心产物

- `artifacts/analysis_base.parquet`
- `artifacts/data_profile.json`
- `artifacts/schema_report.json`
- `artifacts/topic_summary.csv`
- `artifacts/weekly_summary.md`
- `artifacts/insight_cards.json`

### 聚合结果

- `artifacts/aggregates/review_volume_trend_daily.csv`
- `artifacts/aggregates/review_volume_trend_weekly.csv`
- `artifacts/aggregates/review_volume_trend_monthly.csv`
- `artifacts/aggregates/category_summary.csv`
- `artifacts/aggregates/title_summary.csv`
- `artifacts/aggregates/recent_rating_drop.csv`
- `artifacts/aggregates/low_score_keywords.csv`
- `artifacts/aggregates/keywords_by_rating_bucket.csv`
- `artifacts/aggregates/sentiment_distribution.csv`
- `artifacts/aggregates/sentiment_anomalies.csv`

## 当前实现的字段适配说明

这份 Google Play 数据与常见产品运营数据相比，存在几个典型差异：

- 没有版本字段，因此版本评分变化分析自动降级为“字段不可用说明”
- 没有国家与语言字段，因此不做地域维度对比
- 没有开发者、价格、更新时间，因此相关元数据分析自动跳过
- 同时包含 app 与 game 两类实体，因此项目统一抽象为 `entity_type + title + primary_category`

这也是项目的亮点之一：不是强行套模板，而是先检查真实 schema，再按真实字段自动适配。

## 适合写进简历的能力点

- 真实 CSV 多表数据自动识别与字段适配
- 百万级评论数据清洗、去重、画像化建底
- 基于低分评论的主题发现与优先级排序
- 将统计聚合、主题结果与评分下滑预警串成业务洞察链路
- 构建带 fallback 的 LLM 摘要与检索式问答助手
- 用 Streamlit 搭建中文可交互的作品集级分析看板
