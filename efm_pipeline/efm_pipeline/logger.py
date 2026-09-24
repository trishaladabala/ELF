"""Lightweight experiment logger — CSV + JSON, no WandB dependency."""

import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from efm_pipeline.utils import ensure_dir


class ExperimentLogger:
    """Log training metrics to CSV and experiment results to JSON.

    Usage::

        logger = ExperimentLogger("results", "stage0_baseline")
        logger.log_step({"step": 0, "loss": 2.3, "lr": 1e-4})
        logger.log_step({"step": 100, "loss": 1.8, "lr": 9e-5})
        logger.save_result({"gen_ppl": 22.1, "entropy": 5.3})
    """

    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = ensure_dir(output_dir)
        self.experiment_name = experiment_name
        self._csv_path = os.path.join(output_dir, f"{experiment_name}_steps.csv")
        self._json_path = os.path.join(output_dir, f"{experiment_name}_result.json")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None
        self._csv_fields: Optional[List[str]] = None
        self._step_count = 0
        self._start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Per-step CSV logging (training metrics)
    # ------------------------------------------------------------------

    def log_step(self, metrics: Dict[str, Any], print_every: int = 100) -> None:
        """Append one row of metrics to the CSV log.

        The first call determines the column headers.  Subsequent calls must
        provide the same keys (extra keys are silently ignored).
        """
        if self._csv_writer is None:
            self._csv_fields = list(metrics.keys())
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_fields
            )
            self._csv_writer.writeheader()

        # Filter to known fields (ignore extras).
        row = {k: metrics.get(k, "") for k in self._csv_fields}
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._step_count += 1

        # Console output.
        if self._step_count % print_every == 0 or self._step_count == 1:
            elapsed = time.perf_counter() - self._start_time
            parts = [f"[{self.experiment_name}]"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4g}")
                else:
                    parts.append(f"{k}={v}")
            parts.append(f"({elapsed:.0f}s)")
            print(" | ".join(parts))

    # ------------------------------------------------------------------
    # Final result JSON (experiment outcomes)
    # ------------------------------------------------------------------

    def save_result(self, result: Dict[str, Any]) -> str:
        """Save the final experiment result to a JSON file.

        Adds metadata (experiment name, timestamp, elapsed time).
        Returns the path to the saved file.
        """
        elapsed = time.perf_counter() - self._start_time
        output = {
            "experiment": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            **result,
        }
        with open(self._json_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"[{self.experiment_name}] Result saved → {self._json_path}")
        return self._json_path

    def load_result(self) -> Optional[Dict[str, Any]]:
        """Load a previously saved result JSON, or None if it doesn't exist."""
        if os.path.exists(self._json_path):
            with open(self._json_path) as f:
                return json.load(f)
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def __del__(self):
        self.close()


def load_all_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all *_result.json files in a directory into a dict keyed by experiment name."""
    results = {}
    if not os.path.isdir(results_dir):
        return results
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_result.json"):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            name = data.get("experiment", fname.replace("_result.json", ""))
            results[name] = data
    return results
