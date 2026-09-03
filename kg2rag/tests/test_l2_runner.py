from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openie_parser import parse_triples


class OpenIEParserTests(unittest.TestCase):
    def test_json_object(self):
        value = '{"triples":[{"subject":"王家卫","relation":"执导","object":"2046"}]}'
        self.assertEqual(parse_triples(value), [["王家卫", "执导", "2046"]])

    def test_fenced_json_array(self):
        value = '```json\n[["张曼玉", "出演", "花样年华"]]\n```'
        self.assertEqual(parse_triples(value), [["张曼玉", "出演", "花样年华"]])

    def test_xml_fields(self):
        value = '<triples><triple><subject>梁朝伟</subject><predicate>出演</predicate><object>2046</object></triple></triples>'
        self.assertEqual(parse_triples(value), [["梁朝伟", "出演", "2046"]])

    def test_xml_shorthand(self):
        value = '<triple>宫崎骏 ## 执导 ## 千与千寻</triple>'
        self.assertEqual(parse_triples(value), [["宫崎骏", "执导", "千与千寻"]])


if __name__ == "__main__":
    unittest.main()
