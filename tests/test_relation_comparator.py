"""Tests for relation comparator."""

import pytest
from src.relation_comparator import RelationComparator, ComparisonResult
from dataclasses import dataclass


# Mock data classes for testing
@dataclass
class MockGroundTruthRelation:
    """Mock ground truth relation for testing."""
    id: str
    entity1_text: str
    entity2_text: str
    relation_type: str
    entity1_id: str = ""
    entity2_id: str = ""
    novel: str = "No"


@dataclass
class MockExtractedRelation:
    """Mock extracted relation for testing."""
    entity1_text: str
    entity1_type: str
    entity2_text: str
    entity2_type: str
    relation_type: str


class TestRelationComparator:

    @pytest.fixture
    def comparator(self):
        """Create a RelationComparator instance."""
        return RelationComparator()

    def test_perfect_match(self, comparator):
        """Verify 100% precision and recall for perfect match."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f_score == 1.0
        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_no_matches(self, comparator):
        """Verify 0 metrics when no matches."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [MockExtractedRelation("Gene X", "Gene", "Disease Y", "Disease", "Bind")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f_score == 0.0
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_case_insensitive_matching(self, comparator):
        """Verify case-insensitive entity matching."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [MockExtractedRelation("gene a", "Gene", "disease b", "Disease", "Association")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.true_positives == 1
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_symmetric_entity_matching(self, comparator):
        """Verify entity order doesn't matter (symmetric relations)."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [MockExtractedRelation("Disease B", "Disease", "Gene A", "Gene", "Association")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.true_positives == 1
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_whitespace_normalization(self, comparator):
        """Verify whitespace is normalized in entity names."""
        gt = [MockGroundTruthRelation("R0", "Gene  A", "Disease   B", "Association")]
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.true_positives == 1
        assert result.precision == 1.0

    def test_relation_type_must_match(self, comparator):
        """Verify relation type must match exactly."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Positive_Correlation")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_partial_extraction(self, comparator):
        """Verify correct metrics for partial extraction."""
        gt = [
            MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association"),
            MockGroundTruthRelation("R1", "Gene C", "Disease D", "Bind"),
            MockGroundTruthRelation("R2", "Chem X", "Disease Y", "Negative_Correlation")
        ]
        ext = [
            MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association"),
            MockExtractedRelation("Gene E", "Gene", "Disease F", "Disease", "Cotreatment")  # Spurious
        ]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.true_positives == 1
        assert result.false_positives == 1
        assert result.false_negatives == 2
        assert result.precision == 0.5  # 1/2
        assert result.recall == pytest.approx(1/3, rel=0.01)  # 1/3
        # F-score = 2 * (0.5 * 0.333) / (0.5 + 0.333) = 0.4
        assert result.f_score == pytest.approx(0.4, rel=0.01)

    def test_empty_ground_truth(self, comparator):
        """Verify handling of empty ground truth."""
        gt = []
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.recall == 0.0  # No ground truth to match
        assert result.precision == 0.0  # All extractions are false positives when no GT
        assert result.false_positives == 1
        assert result.false_negatives == 0

    def test_empty_extraction(self, comparator):
        """Verify handling of empty extraction."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = []

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.precision == 0.0  # No extractions
        assert result.recall == 0.0
        assert result.false_negatives == 1
        assert result.false_positives == 0

    def test_duplicate_relations_counted_once(self, comparator):
        """Verify duplicate relations are not double-counted."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [
            MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association"),
            MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")  # Duplicate
        ]

        result = comparator.compare("doc1", "model1", gt, ext)

        # Duplicates should be deduplicated by set operations
        assert result.total_extracted == 1  # After dedup
        assert result.true_positives == 1
        assert result.precision == 1.0

    def test_both_empty(self, comparator):
        """Verify handling when both GT and extracted are empty."""
        gt = []
        ext = []

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f_score == 0.0
        assert result.true_positives == 0
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_comparison_result_contains_doc_id(self, comparator):
        """Verify ComparisonResult stores document ID."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")]

        result = comparator.compare("doc12345", "model1", gt, ext)

        assert result.doc_id == "doc12345"

    def test_comparison_result_contains_model_name(self, comparator):
        """Verify ComparisonResult stores model name."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")]

        result = comparator.compare("doc1", "test-model-xyz", gt, ext)

        assert result.model_name == "test-model-xyz"

    def test_matched_relations_formatted(self, comparator):
        """Verify matched relations are formatted correctly."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert len(result.matched_relations) == 1
        # Should be formatted as <entity1, entity2, type> with normalized, sorted entities
        matched = result.matched_relations[0]
        assert matched.startswith("<")
        assert matched.endswith(">")
        assert "Association" in matched

    def test_missed_relations_formatted(self, comparator):
        """Verify missed relations are formatted correctly."""
        gt = [
            MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association"),
            MockGroundTruthRelation("R1", "Gene C", "Disease D", "Bind")
        ]
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert len(result.missed_relations) == 1
        missed = result.missed_relations[0]
        assert "Bind" in missed

    def test_spurious_relations_formatted(self, comparator):
        """Verify spurious relations are formatted correctly."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [
            MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association"),
            MockExtractedRelation("Gene X", "Gene", "Disease Y", "Disease", "Bind")
        ]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert len(result.spurious_relations) == 1
        spurious = result.spurious_relations[0]
        assert "Bind" in spurious

    def test_multiple_correct_extractions(self, comparator):
        """Verify multiple correct extractions are all counted."""
        gt = [
            MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association"),
            MockGroundTruthRelation("R1", "Gene C", "Disease D", "Bind"),
            MockGroundTruthRelation("R2", "Chem X", "Gene Y", "Positive_Correlation")
        ]
        ext = [
            MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association"),
            MockExtractedRelation("Gene C", "Gene", "Disease D", "Disease", "Bind"),
            MockExtractedRelation("Chem X", "Chemical", "Gene Y", "Gene", "Positive_Correlation")
        ]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.true_positives == 3
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f_score == 1.0

    def test_text_normalization_preserves_meaning(self, comparator):
        """Verify text normalization handles various formats."""
        # Test various whitespace and case combinations
        gt = [MockGroundTruthRelation("R0", "IL-6", "COVID-19", "Association")]
        ext = [MockExtractedRelation("il-6", "Gene", "covid-19", "Disease", "Association")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.true_positives == 1

    def test_special_characters_in_entity_names(self, comparator):
        """Verify special characters in entity names are preserved."""
        gt = [MockGroundTruthRelation("R0", "TNF-α", "Alzheimer's disease", "Association")]
        ext = [MockExtractedRelation("tnf-α", "Gene", "alzheimer's disease", "Disease", "Association")]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.true_positives == 1

    def test_counts_match_relations_lists(self, comparator):
        """Verify counts match the length of relation lists."""
        gt = [
            MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association"),
            MockGroundTruthRelation("R1", "Gene C", "Disease D", "Bind")
        ]
        ext = [
            MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association"),
            MockExtractedRelation("Gene X", "Gene", "Disease Y", "Disease", "Cotreatment")
        ]

        result = comparator.compare("doc1", "model1", gt, ext)

        assert result.true_positives == len(result.matched_relations)
        assert result.false_negatives == len(result.missed_relations)
        assert result.false_positives == len(result.spurious_relations)
