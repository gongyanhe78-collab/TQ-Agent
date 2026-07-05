import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.case_splitter import CaseSplitter


SAMPLE_TEXT = """
一、1-3日雨雪天气过程
3月1日08时至3日08时，全省出现雨雪天气过程。

二、14-15日暴雪天气过程
14日08时至15日08时，北部出现暴雪。

三、3月25～28日沙尘、寒潮、雨雪天气过程
25日至28日出现大风沙尘、寒潮和雨雪天气。

四、3月预报服务情况
本节总结本月预报服务工作，不属于灾害个例。
"""


class CaseSplitterTests(unittest.TestCase):
    def test_splitter_keeps_only_case_sections(self):
        splitter = CaseSplitter()

        sections = splitter.split_cases(SAMPLE_TEXT, "FST2025-3.pdf")

        self.assertEqual(3, len(sections))
        self.assertEqual(
            ["FST2025-3-case-01", "FST2025-3-case-02", "FST2025-3-case-03"],
            [section.case_id for section in sections],
        )
        self.assertEqual("1-3日雨雪天气过程", sections[0].title)
        self.assertEqual("14-15日暴雪天气过程", sections[1].title)
        self.assertIn("沙尘", sections[2].title)
        self.assertTrue(all("预报服务情况" not in section.title for section in sections))

    def test_splitter_extracts_date_range_from_title(self):
        splitter = CaseSplitter()

        sections = splitter.split_cases(SAMPLE_TEXT, "FST2025-3.pdf")

        self.assertEqual("1-3日", sections[0].date_range)
        self.assertEqual("14-15日", sections[1].date_range)
        self.assertEqual("3月25～28日", sections[2].date_range)

    def test_splitter_keeps_single_day_and_extended_disaster_titles(self):
        text = """
一、5月2日雷暴大风天气过程
全省出现雷暴大风。

二、5月16日大风天气过程
全省出现强风沙尘。

三、5月19～21日高温天气
全省出现高温。

四、6月13-14日大范围降水过程
全省出现大范围降水。

五、7月28日强对流天气过程
北中部出现强对流。
"""
        splitter = CaseSplitter()

        sections = splitter.split_cases(text, "FST2025-5.pdf")

        self.assertEqual(5, len(sections))
        self.assertEqual(
            [
                "FST2025-5-case-01",
                "FST2025-5-case-02",
                "FST2025-5-case-03",
                "FST2025-5-case-04",
                "FST2025-5-case-05",
            ],
            [section.case_id for section in sections],
        )
        self.assertEqual("5月2日", sections[0].date_range)
        self.assertEqual("5月16日", sections[1].date_range)
        self.assertEqual("5月19～21日", sections[2].date_range)
        self.assertEqual("6月13-14日", sections[3].date_range)
        self.assertEqual("7月28日", sections[4].date_range)


if __name__ == "__main__":
    unittest.main()
