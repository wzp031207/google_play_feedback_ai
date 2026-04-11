from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.utils import (
    clean_text,
    json_ready_records,
    standardize_columns,
    write_json,
)


SUPPORTED_SUFFIXES = {".csv", ".parquet", ".json"}

FIELD_CANDIDATES = {
    "entity_id": ["app_id", "game_id", "id"],
    "title": ["app_name", "game_name", "title", "name"],
    "description": ["description", "summary", "about"],
    "store_score": ["score", "rating", "avg_rating"],
    "ratings_count": ["ratings_count", "rating_count", "reviews_count"],
    "downloads": ["downloads", "installs", "install_count"],
    "content_rating": ["content_rating", "age_rating"],
    "section": ["section", "genre_section"],
    "categories": ["categories", "category", "genres", "genre"],
    "review_text": ["review_text", "review", "content", "text", "comment"],
    "review_score": ["review_score", "score", "rating", "stars"],
    "review_date": ["review_date", "date", "created_at", "timestamp", "time"],
    "helpful_count": ["helpful_count", "likes", "thumbs_up_count", "helpful_votes"],
    "version": ["version", "app_version"],
    "country": ["country", "region"],
    "language": ["language", "lang"],
    "price": ["price"],
    "developer": ["developer", "developer_name"],
    "updated_at": ["updated_at", "last_updated"],
}


@dataclass(slots=True)
class SourceTable:
    source_key: str
    entity_type: str
    role: str
    path: Path
    dataframe: pd.DataFrame
    field_map: dict[str, str | None]


def discover_data_files(raw_dir: Path | None = None) -> list[Path]:
    root = raw_dir or config.RAW_DATA_DIR
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        dataframe = pd.read_csv(path)
    elif suffix == ".parquet":
        dataframe = pd.read_parquet(path)
    elif suffix == ".json":
        dataframe = pd.read_json(path)
    else:  # pragma: no cover - guarded by discover_data_files
        raise ValueError(f"Unsupported data file type: {path.suffix}")
    return standardize_columns(dataframe)


def infer_field_map(columns: list[str]) -> dict[str, str | None]:
    available = set(columns)
    mapping: dict[str, str | None] = {}
    for logical_name, candidates in FIELD_CANDIDATES.items():
        mapping[logical_name] = next((candidate for candidate in candidates if candidate in available), None)
    return mapping


def infer_table_role(field_map: dict[str, str | None]) -> str:
    if field_map.get("review_text") and field_map.get("review_score"):
        return "reviews"
    if field_map.get("title") and field_map.get("entity_id"):
        return "info"
    return "unknown"


def infer_source_key(path: Path) -> str:
    stem = path.stem.lower()
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def infer_entity_type(path: Path, field_map: dict[str, str | None]) -> str:
    stem = path.stem.lower()
    if "game" in stem or (field_map.get("entity_id") or "").startswith("game_"):
        return "game"
    if "app" in stem or (field_map.get("entity_id") or "").startswith("app_"):
        return "app"
    return "unknown"


def inspect_data_files(raw_dir: Path | None = None) -> dict[str, Any]:
    files = discover_data_files(raw_dir)
    inspections: list[dict[str, Any]] = []
    field_availability: dict[str, list[str]] = {
        key: [] for key in FIELD_CANDIDATES
    }
    source_tables: list[SourceTable] = []

    for path in files:
        dataframe = read_table(path)
        field_map = infer_field_map(list(dataframe.columns))
        role = infer_table_role(field_map)
        source_key = infer_source_key(path)
        entity_type = infer_entity_type(path, field_map)
        source_tables.append(
            SourceTable(
                source_key=source_key,
                entity_type=entity_type,
                role=role,
                path=path,
                dataframe=dataframe,
                field_map=field_map,
            )
        )
        for logical_name, column_name in field_map.items():
            if column_name:
                field_availability[logical_name].append(f"{path.name}:{column_name}")

        inspections.append(
            {
                "file_name": path.name,
                "relative_path": path.relative_to(raw_dir or config.RAW_DATA_DIR).as_posix()
                if (raw_dir or config.RAW_DATA_DIR) in path.parents or path == (raw_dir or config.RAW_DATA_DIR)
                else path.name,
                "role": role,
                "entity_type": entity_type,
                "source_key": source_key,
                "row_count": int(len(dataframe)),
                "column_count": int(len(dataframe.columns)),
                "columns": list(dataframe.columns),
                "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
                "field_map": field_map,
                "sample_rows": json_ready_records(dataframe, limit=3),
            }
        )

    schema_report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "raw_dir": str((raw_dir or config.RAW_DATA_DIR).relative_to(config.PROJECT_ROOT)),
        "files": inspections,
        "field_availability": field_availability,
    }
    write_json(config.SCHEMA_REPORT_PATH, schema_report)
    return {
        "schema_report": schema_report,
        "source_tables": source_tables,
    }


def load_google_play_sources(raw_dir: Path | None = None) -> list[SourceTable]:
    inspection = inspect_data_files(raw_dir)
    source_tables: list[SourceTable] = inspection["source_tables"]
    valid_tables = [table for table in source_tables if table.role in {"info", "reviews"}]
    if not valid_tables:
        raise ValueError("No valid Google Play info/reviews tables found in the raw data directory.")
    return valid_tables


def describe_available_fields(schema_report: dict[str, Any]) -> dict[str, Any]:
    availability = schema_report["field_availability"]
    return {
        "评论文本": availability.get("review_text", []),
        "星级评分": availability.get("review_score", []),
        "评论时间": availability.get("review_date", []),
        "应用或游戏标题": availability.get("title", []),
        "分类字段": availability.get("categories", []),
        "版本号": availability.get("version", []),
        "国家字段": availability.get("country", []),
        "语言字段": availability.get("language", []),
        "安装量": availability.get("downloads", []),
        "开发者": availability.get("developer", []),
        "更新时间": availability.get("updated_at", []),
        "价格": availability.get("price", []),
        "内容分级": availability.get("content_rating", []),
    }


def schema_summary_text(schema_report: dict[str, Any]) -> str:
    lines = ["数据文件检查结果："]
    for file_info in schema_report["files"]:
        lines.append(
            f"- {file_info['file_name']}：{file_info['row_count']:,} 行，{file_info['column_count']} 列，角色={file_info['role']}"
        )
        lines.append(f"  列名：{', '.join(file_info['columns'])}")
        if file_info["sample_rows"]:
            lines.append(f"  样例：{clean_text(file_info['sample_rows'][0])}")
    return "\n".join(lines)
