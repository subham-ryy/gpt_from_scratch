"""Machine-readable metrics logger for CSV and JSONL outputs."""
import csv
import json
from pathlib import Path


class MetricsLogger:
    """Logs numerical step metrics to both CSV and JSONL formats simultaneously."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "metrics.csv"
        self.jsonl_path = self.output_dir / "metrics.jsonl"

        self._csv_file = None
        self._csv_writer = None
        self._fieldnames = []

    def log(self, metrics: dict) -> None:
        """Write a dictionary of metrics for the current step."""
        # 1. Append to JSONL (always handles flexible keys)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        # 2. Append to CSV
        keys = list(metrics.keys())
        if self._csv_writer is None:
            self._fieldnames = keys
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()

        # If new keys appear, recreate header or write subset
        row = {k: metrics.get(k, "") for k in self._fieldnames}
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()

