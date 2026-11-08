"""校验评测报告是否满足 CI 质量门禁。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.evals.quality_gate import 从负载检查门禁


def main(argv: list[str] | None = None) -> int:
    """读取报告并输出门禁结果。"""

    parser = argparse.ArgumentParser(description="校验黄金评测报告质量门禁")
    parser.add_argument("--report", required=True, help="EvaluationReport JSON 路径")
    parser.add_argument("--mode", choices=("offline", "real"), required=True, help="门禁模式")
    parser.add_argument(
        "--expected-dataset-version",
        default=None,
        help="期望数据集版本，默认读取 golden_v1_manifest.json",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "backend" / "evals" / "datasets" / "golden_v1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_version = args.expected_dataset_version or str(manifest["dataset_version"])
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = 从负载检查门禁(
        payload,
        mode=args.mode,
        expected_dataset_version=expected_version,
    )
    if result.passed:
        print(f"评测质量门禁通过：mode={args.mode}, dataset={expected_version}")
        return 0
    print(f"评测质量门禁失败：mode={args.mode}, dataset={expected_version}")
    for failure in result.failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
