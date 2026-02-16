"""
Simple report logging utility for nanochat_mlx.
Collects evaluation results and saves them to disk as JSON.
"""

import json
import os

from nanochat_mlx.common import get_base_dir


class Report:
    def __init__(self):
        self.sections = []

    def log(self, section, data):
        self.sections.append({"section": section, "data": data})
        # Also save to disk
        base_dir = get_base_dir()
        report_path = os.path.join(base_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(self.sections, f, indent=2, default=str)


_report = None


def get_report():
    global _report
    if _report is None:
        _report = Report()
    return _report
