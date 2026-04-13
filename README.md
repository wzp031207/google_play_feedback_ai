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


## 项目结构

```text
recharge_ai_project/
├─ app.py
├─ run_pipeline.py
├─ requirements.txt
├─ README.md
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


### 4. LLM 周报与策略建议

`src/llm_engine.py` 会根据真实聚合结果、主题结果与评分变化结果生成：

- 周报摘要
- 重点问题总结
- 产品建议
- 运营建议
- 风险预警


### 5. 检索式问答助手

`src/qa_engine.py` 先检索以下分析结果，再组织回答：

- `topic_summary.csv`
- `artifacts/aggregates/*.csv`
- `weekly_summary.md`
- `insight_cards.json`


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

