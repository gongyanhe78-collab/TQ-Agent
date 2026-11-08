"""便捷运行离线评测并输出 JSON 与 Markdown 报告。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evals.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
