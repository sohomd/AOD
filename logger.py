"""
logger.py — CSV + console logging for AoD experiments.

Usage:
    with Logger(save_dir) as logger:
        logger.log_args(args)
        for update in range(n_updates):
            logger.log({"step": update, "loss": 0.5,...})
    # File is automatically closed and flushed on exit.
"""

import os
import csv
import json
from datetime import datetime


class Logger:
    """
    Logs metrics to a CSV file, one row per call to log().

    Handles schema evolution: if a log() call introduces new keys that
    were not present in earlier calls, the CSV is rewritten with the
    expanded column set and empty strings for missing earlier values.
    """

    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, "metrics.csv")
        self.fieldnames = []
        self._rows = []  # Buffer for schema-evolution rewrites
        self._writer = None
        self._file = None

    # ── Context manager ──

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Do not suppress exceptions

    # ── Args logging ──

    def log_args(self, args):
        """
        Save experiment arguments as JSON.

        Handles non-serializable values by converting them to strings.
        Also records the experiment start timestamp.
        """
        path = os.path.join(self.save_dir, "args.json")

        if hasattr(args, "__dict__"):
            args_dict = vars(args)
        elif isinstance(args, dict):
            args_dict = args
        else:
            args_dict = {"args": str(args)}

        args_dict = {k: _make_serializable(v) for k, v in args_dict.items()}
        args_dict["_logged_at"] = datetime.now().isoformat()

        with open(path, "w") as f:
            json.dump(args_dict, f, indent=2)

    # ── Metrics logging ──

    def log(self, metrics: dict):
        """
        Log a single row of metrics to the CSV file.

        If metrics contains keys not seen in previous calls, the CSV
        is rewritten with the expanded schema.
        """
        metrics = dict(metrics)
        metrics["timestamp"] = datetime.now().isoformat()

        new_keys = [k for k in metrics if k not in self.fieldnames]

        if self._writer is None:
            # First call: initialize CSV
            self.fieldnames = list(metrics.keys())
            self._open_csv()
        elif new_keys:
            # Schema evolution: new columns appeared
            self.fieldnames.extend(new_keys)
            self._rewrite_csv()

        row = {k: metrics.get(k, "") for k in self.fieldnames}
        self._rows.append(row)
        self._writer.writerow(row)
        self._file.flush()

    # ── Internal helpers ──

    def _open_csv(self):
        """Open the CSV file and write the header."""
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self.fieldnames
        )
        self._writer.writeheader()

    def _rewrite_csv(self):
        """Rewrite the entire CSV with an expanded column set."""
        if self._file is not None:
            self._file.close()

        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self.fieldnames
        )
        self._writer.writeheader()

        # Re-emit all previous rows with the new schema
        for old_row in self._rows:
            padded = {k: old_row.get(k, "") for k in self.fieldnames}
            self._writer.writerow(padded)

        self._file.flush()

    def close(self):
        """Flush and close the CSV file."""
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


def _make_serializable(v):
    """Convert a value to a JSON-serializable type."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_make_serializable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _make_serializable(val) for k, val in v.items()}
    # Fallback: torch.device, Path, custom objects, etc.
    return str(v)