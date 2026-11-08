"""本地压测脚本的指标解析单元测试。"""
from __future__ import annotations

import unittest

from scripts.load_test import aggregate_metrics, label_total, percentile


SAMPLE_METRICS = """# HELP model_calls_total 模型调用次数
# TYPE model_calls_total counter
model_calls_total{kind="chat",status="success"} 12.0
model_calls_total{kind="chat",status="error"} 3.0
model_calls_total{kind="intent",status="success"} 5.0
cache_fallbacks_total{reason="redis_down"} 2.0
vector_fallbacks_total{reason="milvus_down"} 1.0
app_requests_total 100.0
"""


class MetricParsingTests(unittest.TestCase):
    """验证 Prometheus 文本解析能正确汇总总数与按标签统计。"""

    def test_aggregate_metrics_sums_all_samples_per_metric(self) -> None:
        totals = aggregate_metrics(SAMPLE_METRICS)
        self.assertEqual(totals["model_calls_total"], 20.0)
        self.assertEqual(totals["cache_fallbacks_total"], 2.0)
        self.assertEqual(totals["vector_fallbacks_total"], 1.0)
        self.assertEqual(totals["app_requests_total"], 100.0)

    def test_label_total_filters_by_label_value(self) -> None:
        self.assertEqual(
            label_total(SAMPLE_METRICS, "model_calls_total", "status", "success"),
            17.0,
        )
        self.assertEqual(
            label_total(SAMPLE_METRICS, "model_calls_total", "status", "error"),
            3.0,
        )

    def test_label_total_returns_zero_for_missing_label(self) -> None:
        self.assertEqual(
            label_total(SAMPLE_METRICS, "model_calls_total", "status", "cancel"),
            0.0,
        )

    def test_percentile_returns_expected_quantiles(self) -> None:
        values = [float(index) for index in range(1, 101)]
        self.assertEqual(percentile(values, 0.5), 50.0)
        self.assertEqual(percentile(values, 0.95), 95.0)


if __name__ == "__main__":
    unittest.main()
