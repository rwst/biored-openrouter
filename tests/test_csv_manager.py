"""Tests for CSV manager."""

import pytest
import csv
import json
import tempfile
import os
from pathlib import Path
from src.csv_manager import CSVManager
from src.relation_comparator import ComparisonResult


class TestCSVManager:

    @pytest.fixture
    def temp_csv_path(self):
        """Create a temporary CSV file path."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
        # Delete the file so CSVManager can create it
        if temp_path.exists():
            temp_path.unlink()
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def manager(self, temp_csv_path):
        """Create manager with temporary CSV."""
        return CSVManager(temp_csv_path)

    @pytest.fixture
    def sample_result(self):
        """Create a sample ComparisonResult."""
        return ComparisonResult(
            doc_id="12345",
            model_name="test-model",
            total_ground_truth=5,
            total_extracted=4,
            true_positives=3,
            false_positives=1,
            false_negatives=2,
            precision=0.75,
            recall=0.60,
            f_score=0.6667,
            matched_relations=["<gene a, disease b, Association>"],
            missed_relations=["<chem x, disease y, Bind>"],
            spurious_relations=["<gene c, disease d, Cotreatment>"]
        )

    def test_creates_file_with_headers(self, temp_csv_path):
        """Verify CSV file is created with correct headers."""
        manager = CSVManager(temp_csv_path)

        assert temp_csv_path.exists()
        with open(temp_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            assert "model_name" in reader.fieldnames
            assert "doc_id" in reader.fieldnames
            assert "precision" in reader.fieldnames
            assert "recall" in reader.fieldnames
            assert "f_score" in reader.fieldnames

    def test_save_result_appends_row(self, manager, sample_result, temp_csv_path):
        """Verify save_result adds new row."""
        manager.save_result(sample_result)

        rows = manager._read_all()
        assert len(rows) == 1
        assert rows[0]["doc_id"] == "12345"
        assert rows[0]["model_name"] == "test-model"

    def test_save_result_overwrites_existing(self, manager, sample_result):
        """Verify same (model, doc_id) overwrites existing row."""
        manager.save_result(sample_result)

        # Modify and save again
        updated_result = ComparisonResult(
            doc_id="12345",
            model_name="test-model",
            total_ground_truth=10,  # Changed
            total_extracted=8,
            true_positives=6,
            false_positives=2,
            false_negatives=4,
            precision=0.75,
            recall=0.60,
            f_score=0.6667,
            matched_relations=[],
            missed_relations=[],
            spurious_relations=[]
        )
        manager.save_result(updated_result)

        rows = manager._read_all()
        assert len(rows) == 1  # Still only one row
        assert rows[0]["total_ground_truth"] == "10"  # Updated value

    def test_different_models_separate_rows(self, manager, sample_result):
        """Verify different models create separate rows."""
        manager.save_result(sample_result)

        result2 = ComparisonResult(
            doc_id="12345",  # Same doc
            model_name="different-model",  # Different model
            total_ground_truth=5,
            total_extracted=4,
            true_positives=2,
            false_positives=2,
            false_negatives=3,
            precision=0.50,
            recall=0.40,
            f_score=0.4444,
            matched_relations=[],
            missed_relations=[],
            spurious_relations=[]
        )
        manager.save_result(result2)

        rows = manager._read_all()
        assert len(rows) == 2

    def test_different_docs_separate_rows(self, manager, sample_result):
        """Verify different doc_ids create separate rows."""
        manager.save_result(sample_result)

        result2 = ComparisonResult(
            doc_id="67890",  # Different doc
            model_name="test-model",  # Same model
            total_ground_truth=3,
            total_extracted=3,
            true_positives=3,
            false_positives=0,
            false_negatives=0,
            precision=1.0,
            recall=1.0,
            f_score=1.0,
            matched_relations=[],
            missed_relations=[],
            spurious_relations=[]
        )
        manager.save_result(result2)

        rows = manager._read_all()
        assert len(rows) == 2

    def test_precision_formatting(self, manager, sample_result, temp_csv_path):
        """Verify numeric precision in CSV output."""
        manager.save_result(sample_result)

        rows = manager._read_all()
        assert rows[0]["precision"] == "0.7500"
        assert rows[0]["f_score"] == "0.6667"

    def test_json_encoded_lists(self, manager, sample_result):
        """Verify relation lists are JSON-encoded."""
        manager.save_result(sample_result)

        rows = manager._read_all()
        matched = json.loads(rows[0]["matched_relations"])
        assert isinstance(matched, list)
        assert len(matched) == 1

    def test_aggregate_stats_single_model(self, manager):
        """Verify aggregate statistics calculation."""
        # Add multiple results
        results = [
            ComparisonResult("doc1", "model1", 5, 4, 3, 1, 2, 0.75, 0.60, 0.67, [], [], []),
            ComparisonResult("doc2", "model1", 5, 5, 4, 1, 1, 0.80, 0.80, 0.80, [], [], []),
        ]
        for r in results:
            manager.save_result(r)

        stats = manager.get_aggregate_stats("model1")

        assert stats["count"] == 2
        assert stats["total_true_positives"] == 7  # 3 + 4
        assert stats["total_false_positives"] == 2  # 1 + 1
        assert stats["total_false_negatives"] == 3  # 2 + 1

    def test_aggregate_stats_filter_by_model(self, manager):
        """Verify filtering by model name."""
        results = [
            ComparisonResult("doc1", "model1", 5, 4, 3, 1, 2, 0.75, 0.60, 0.67, [], [], []),
            ComparisonResult("doc1", "model2", 5, 5, 5, 0, 0, 1.0, 1.0, 1.0, [], [], []),
        ]
        for r in results:
            manager.save_result(r)

        stats = manager.get_aggregate_stats("model1")
        assert stats["count"] == 1
        assert stats["total_true_positives"] == 3

    def test_aggregate_stats_all_models(self, manager):
        """Verify aggregate stats without model filter."""
        results = [
            ComparisonResult("doc1", "model1", 5, 4, 3, 1, 2, 0.75, 0.60, 0.67, [], [], []),
            ComparisonResult("doc1", "model2", 5, 5, 5, 0, 0, 1.0, 1.0, 1.0, [], [], []),
        ]
        for r in results:
            manager.save_result(r)

        stats = manager.get_aggregate_stats()  # No filter
        assert stats["count"] == 2
        assert stats["total_true_positives"] == 8  # 3 + 5

    def test_aggregate_stats_empty_csv(self, manager):
        """Verify aggregate stats for empty CSV."""
        stats = manager.get_aggregate_stats()
        assert stats["count"] == 0
        assert "total_true_positives" not in stats

    def test_aggregate_stats_micro_precision(self, manager):
        """Verify micro-precision calculation."""
        results = [
            ComparisonResult("doc1", "model1", 5, 4, 3, 1, 2, 0.75, 0.60, 0.67, [], [], []),
            ComparisonResult("doc2", "model1", 5, 5, 4, 1, 1, 0.80, 0.80, 0.80, [], [], []),
        ]
        for r in results:
            manager.save_result(r)

        stats = manager.get_aggregate_stats("model1")
        # micro_precision = TP / (TP + FP) = 7 / (7 + 2) = 7/9 = 0.777...
        assert stats["micro_precision"] == pytest.approx(7/9, rel=0.01)

    def test_aggregate_stats_micro_recall(self, manager):
        """Verify micro-recall calculation."""
        results = [
            ComparisonResult("doc1", "model1", 5, 4, 3, 1, 2, 0.75, 0.60, 0.67, [], [], []),
            ComparisonResult("doc2", "model1", 5, 5, 4, 1, 1, 0.80, 0.80, 0.80, [], [], []),
        ]
        for r in results:
            manager.save_result(r)

        stats = manager.get_aggregate_stats("model1")
        # micro_recall = TP / (TP + FN) = 7 / (7 + 3) = 7/10 = 0.7
        assert stats["micro_recall"] == pytest.approx(0.7, rel=0.01)

    def test_aggregate_stats_micro_f_score(self, manager):
        """Verify micro-F-score calculation."""
        results = [
            ComparisonResult("doc1", "model1", 5, 4, 3, 1, 2, 0.75, 0.60, 0.67, [], [], []),
            ComparisonResult("doc2", "model1", 5, 5, 4, 1, 1, 0.80, 0.80, 0.80, [], [], []),
        ]
        for r in results:
            manager.save_result(r)

        stats = manager.get_aggregate_stats("model1")
        # precision = 7/9, recall = 7/10
        # f_score = 2 * (7/9 * 7/10) / (7/9 + 7/10) = 2 * 49/90 / (70/90 + 63/90) = 98/90 / 133/90 = 98/133
        expected_f = 2 * (7/9 * 7/10) / (7/9 + 7/10)
        assert stats["micro_f_score"] == pytest.approx(expected_f, rel=0.01)

    def test_timestamp_field_present(self, manager, sample_result):
        """Verify timestamp is included in saved results."""
        manager.save_result(sample_result)

        rows = manager._read_all()
        assert "timestamp" in rows[0]
        assert len(rows[0]["timestamp"]) > 0  # Non-empty

    def test_timestamp_iso_format(self, manager, sample_result):
        """Verify timestamp is in ISO format."""
        manager.save_result(sample_result)

        rows = manager._read_all()
        timestamp = rows[0]["timestamp"]
        # Should be ISO format like "2023-12-15T10:30:45.123456"
        assert "T" in timestamp
        # Should be parseable
        from datetime import datetime
        datetime.fromisoformat(timestamp)  # Should not raise

    def test_all_fieldnames_present(self, manager, sample_result):
        """Verify all expected field names are in the CSV."""
        manager.save_result(sample_result)

        rows = manager._read_all()
        row = rows[0]

        expected_fields = [
            "model_name", "doc_id", "timestamp",
            "total_ground_truth", "total_extracted",
            "true_positives", "false_positives", "false_negatives",
            "precision", "recall", "f_score",
            "matched_relations", "missed_relations", "spurious_relations"
        ]

        for field in expected_fields:
            assert field in row

    def test_integer_fields_formatted_correctly(self, manager, sample_result):
        """Verify integer fields are formatted correctly."""
        manager.save_result(sample_result)

        rows = manager._read_all()
        row = rows[0]

        assert row["total_ground_truth"] == "5"
        assert row["total_extracted"] == "4"
        assert row["true_positives"] == "3"
        assert row["false_positives"] == "1"
        assert row["false_negatives"] == "2"

    def test_multiple_updates_to_same_doc(self, manager, sample_result):
        """Verify multiple updates to same (model, doc) work correctly."""
        # Save 3 times with different values
        for i in range(3):
            result = ComparisonResult(
                doc_id="12345",
                model_name="test-model",
                total_ground_truth=i,
                total_extracted=i,
                true_positives=i,
                false_positives=0,
                false_negatives=0,
                precision=1.0,
                recall=1.0,
                f_score=1.0,
                matched_relations=[],
                missed_relations=[],
                spurious_relations=[]
            )
            manager.save_result(result)

        rows = manager._read_all()
        assert len(rows) == 1
        assert rows[0]["total_ground_truth"] == "2"  # Last value

    def test_empty_relation_lists_encoded(self, manager):
        """Verify empty relation lists are properly JSON-encoded."""
        result = ComparisonResult(
            doc_id="12345",
            model_name="test-model",
            total_ground_truth=0,
            total_extracted=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f_score=0.0,
            matched_relations=[],
            missed_relations=[],
            spurious_relations=[]
        )
        manager.save_result(result)

        rows = manager._read_all()
        matched = json.loads(rows[0]["matched_relations"])
        missed = json.loads(rows[0]["missed_relations"])
        spurious = json.loads(rows[0]["spurious_relations"])

        assert matched == []
        assert missed == []
        assert spurious == []

    def test_aggregate_stats_no_matching_model(self, manager):
        """Verify aggregate stats for non-existent model."""
        result = ComparisonResult(
            "doc1", "model1", 5, 4, 3, 1, 2, 0.75, 0.60, 0.67, [], [], []
        )
        manager.save_result(result)

        stats = manager.get_aggregate_stats("nonexistent-model")
        assert stats["count"] == 0
