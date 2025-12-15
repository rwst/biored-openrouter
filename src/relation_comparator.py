"""Relation comparison and evaluation module."""

from dataclasses import dataclass
from typing import List, Set, Tuple
import re


@dataclass
class ComparisonResult:
    """Result of comparing extracted relations against ground truth."""
    doc_id: str
    model_name: str
    total_ground_truth: int
    total_extracted: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f_score: float
    matched_relations: List[str]      # List of matched relation IDs
    missed_relations: List[str]       # List of missed ground truth relation IDs
    spurious_relations: List[str]     # List of FP relation descriptions


class RelationComparator:
    """Compare extracted relations against ground truth."""

    def __init__(self):
        pass

    def compare(
        self,
        doc_id: str,
        model_name: str,
        ground_truth_relations: List,  # List[Relation] from data_loader
        extracted_relations: List      # List[ExtractedRelation] from openrouter_client
    ) -> ComparisonResult:
        """
        Compare extracted relations to ground truth.

        Uses text-based matching with case-insensitive, symmetric comparison.
        Computes precision, recall, and F-score per the BioRED evaluation methodology.

        Args:
            doc_id: Document identifier
            model_name: Model name used for extraction
            ground_truth_relations: Ground truth relations from BioRED
            extracted_relations: Relations extracted by the LLM

        Returns:
            ComparisonResult with detailed metrics and matched/missed relations
        """
        # Normalize ground truth to comparable tuples
        gt_set = self._normalize_relations(ground_truth_relations, is_ground_truth=True)
        ext_set = self._normalize_relations(extracted_relations, is_ground_truth=False)

        # Calculate matches
        tp = gt_set & ext_set
        fp = ext_set - gt_set
        fn = gt_set - ext_set

        # Calculate metrics
        precision = len(tp) / len(ext_set) if ext_set else 0.0
        recall = len(tp) / len(gt_set) if gt_set else 0.0
        f_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return ComparisonResult(
            doc_id=doc_id,
            model_name=model_name,
            total_ground_truth=len(gt_set),
            total_extracted=len(ext_set),
            true_positives=len(tp),
            false_positives=len(fp),
            false_negatives=len(fn),
            precision=precision,
            recall=recall,
            f_score=f_score,
            matched_relations=self._format_relations(tp),
            missed_relations=self._format_relations(fn),
            spurious_relations=self._format_relations(fp)
        )

    def _normalize_relations(
        self,
        relations: List,
        is_ground_truth: bool
    ) -> Set[Tuple[str, str, str]]:
        """
        Convert relations to normalized tuples for comparison.

        Tuple format: (entity1_text_normalized, entity2_text_normalized, relation_type)
        Entities are sorted alphabetically for symmetric matching.

        Args:
            relations: List of Relation or ExtractedRelation objects
            is_ground_truth: Whether these are ground truth relations

        Returns:
            Set of normalized relation tuples
        """
        normalized = set()
        for rel in relations:
            if is_ground_truth:
                e1 = self._normalize_text(rel.entity1_text)
                e2 = self._normalize_text(rel.entity2_text)
                rel_type = rel.relation_type
            else:
                e1 = self._normalize_text(rel.entity1_text)
                e2 = self._normalize_text(rel.entity2_text)
                rel_type = rel.relation_type

            # Sort entities for symmetric comparison
            entities = tuple(sorted([e1, e2]))
            normalized.add((entities[0], entities[1], rel_type))

        return normalized

    def _normalize_text(self, text: str) -> str:
        """
        Normalize entity text for comparison.

        Applies lowercase and whitespace normalization.

        Args:
            text: Entity text to normalize

        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _format_relations(self, relation_set: Set[Tuple]) -> List[str]:
        """
        Format relation tuples as readable strings.

        Args:
            relation_set: Set of relation tuples

        Returns:
            List of formatted relation strings
        """
        return [f"<{r[0]}, {r[1]}, {r[2]}>" for r in relation_set]
