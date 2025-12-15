"""CSV database manager for storing evaluation results."""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from src.relation_comparator import ComparisonResult


class CSVManager:
    """Manage persistent CSV results database."""

    DEFAULT_PATH = Path("results.csv")

    FIELDNAMES = [
        "model_name",
        "doc_id",
        "timestamp",
        "total_ground_truth",
        "total_extracted",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f_score",
        "matched_relations",
        "missed_relations",
        "spurious_relations"
    ]

    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize CSV manager.

        Args:
            csv_path: Path to CSV file. If None, uses default path.
        """
        self.csv_path = csv_path or self.DEFAULT_PATH
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create CSV with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def save_result(self, result: ComparisonResult) -> None:
        """
        Save or update a comparison result.

        Overwrites existing row with same (model_name, doc_id).

        Args:
            result: ComparisonResult to save
        """
        # Read existing data
        existing_rows = self._read_all()

        # Create new row
        new_row = self._result_to_row(result)

        # Find and replace or append
        key = (result.model_name, result.doc_id)
        updated = False
        for i, row in enumerate(existing_rows):
            if (row["model_name"], row["doc_id"]) == key:
                existing_rows[i] = new_row
                updated = True
                break

        if not updated:
            existing_rows.append(new_row)

        # Write all data back
        self._write_all(existing_rows)

    def _result_to_row(self, result: ComparisonResult) -> dict:
        """
        Convert ComparisonResult to CSV row dict.

        Args:
            result: ComparisonResult to convert

        Returns:
            Dictionary with CSV row data
        """
        return {
            "model_name": result.model_name,
            "doc_id": result.doc_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_ground_truth": result.total_ground_truth,
            "total_extracted": result.total_extracted,
            "true_positives": result.true_positives,
            "false_positives": result.false_positives,
            "false_negatives": result.false_negatives,
            "precision": f"{result.precision:.4f}",
            "recall": f"{result.recall:.4f}",
            "f_score": f"{result.f_score:.4f}",
            "matched_relations": json.dumps(result.matched_relations),
            "missed_relations": json.dumps(result.missed_relations),
            "spurious_relations": json.dumps(result.spurious_relations)
        }

    def _read_all(self) -> list:
        """
        Read all rows from CSV.

        Returns:
            List of row dictionaries
        """
        rows = []
        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows

    def _write_all(self, rows: list) -> None:
        """
        Write all rows to CSV.

        Args:
            rows: List of row dictionaries to write
        """
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def get_aggregate_stats(self, model_name: Optional[str] = None) -> dict:
        """
        Calculate aggregate statistics across all documents.

        Optionally filter by model name.

        Args:
            model_name: Optional model name to filter by

        Returns:
            Dictionary with aggregate statistics including:
            - count: number of documents
            - total_true_positives: sum of all TPs
            - total_false_positives: sum of all FPs
            - total_false_negatives: sum of all FNs
            - micro_precision: micro-averaged precision
            - micro_recall: micro-averaged recall
            - micro_f_score: micro-averaged F-score
        """
        rows = self._read_all()

        if model_name:
            rows = [r for r in rows if r["model_name"] == model_name]

        if not rows:
            return {"count": 0}

        total_tp = sum(int(r["true_positives"]) for r in rows)
        total_fp = sum(int(r["false_positives"]) for r in rows)
        total_fn = sum(int(r["false_negatives"]) for r in rows)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        return {
            "count": len(rows),
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f_score": f_score
        }
