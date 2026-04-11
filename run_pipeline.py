from __future__ import annotations

import json

from src.pipeline import run_full_pipeline


def main() -> None:
    result = run_full_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
