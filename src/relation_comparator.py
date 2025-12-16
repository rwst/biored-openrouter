"""Relation comparison and evaluation module."""

from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Optional
import re


@dataclass
class MatchDetails:
    """Details about how a relation matched."""
    ground_truth_e1: str
    ground_truth_e2: str
    extracted_e1: str
    extracted_e2: str
    relation_type: str
    match_type: str  # "exact", "synonym", or "fuzzy"


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
    match_details: List[MatchDetails] = None  # Detailed match information

    def __post_init__(self):
        if self.match_details is None:
            self.match_details = []


class RelationComparator:
    """Compare extracted relations against ground truth."""

    def __init__(self, synonym_sets: Optional[Dict[str, 'SynonymSet']] = None):
        """
        Initialize relation comparator.

        Args:
            synonym_sets: Optional mapping of identifier -> SynonymSet for synonym matching
        """
        self.synonym_sets = synonym_sets or {}
        # Build reverse lookup: normalized_text -> set of identifiers
        self.text_to_identifiers = self._build_text_index()

    def _build_text_index(self) -> Dict[str, Set[str]]:
        """Build reverse index from normalized text to identifiers."""
        text_index = {}
        for identifier, syn_set in self.synonym_sets.items():
            for text in syn_set.texts:
                normalized = self._normalize_text(text)
                if normalized not in text_index:
                    text_index[normalized] = set()
                text_index[normalized].add(identifier)
        return text_index

    def _apply_fuzzy_normalization(self, text: str) -> str:
        """
        Apply aggressive fuzzy normalization for non-matching entities.

        Removes: parentheses (keeping content), hyphens, special characters.
        Example: "Na(v)1.5" -> "nav15", "IL-6" -> "il6"
        """
        # Remove parentheses but keep their content
        text = text.replace('(', '').replace(')', '')
        # Remove hyphens, dots, and special chars
        text = re.sub(r'[^a-z0-9\s]', '', text.lower())
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _get_match_type(self, text1: str, text2: str) -> str:
        """
        Determine match type between two entity texts.

        Returns: "exact", "synonym", "fuzzy", or "none"
        """
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)

        # Exact match
        if norm1 == norm2:
            return "exact"

        # Synonym match: check if they map to same identifier
        ids1 = self.text_to_identifiers.get(norm1, set())
        ids2 = self.text_to_identifiers.get(norm2, set())

        if ids1 and ids2 and ids1 & ids2:
            return "synonym"

        # Fuzzy match: apply aggressive normalization
        fuzzy1 = self._apply_fuzzy_normalization(text1)
        fuzzy2 = self._apply_fuzzy_normalization(text2)

        if fuzzy1 and fuzzy2 and fuzzy1 == fuzzy2:
            return "fuzzy"

        return "none"

    def compare(
        self,
        doc_id: str,
        model_name: str,
        ground_truth_relations: List,  # List[Relation] from data_loader
        extracted_relations: List      # List[ExtractedRelation] from openrouter_client
    ) -> ComparisonResult:
        """
        Compare extracted relations to ground truth with synonym matching.

        Uses text-based matching with synonym awareness and fuzzy normalization.

        Args:
            doc_id: Document identifier
            model_name: Model name used for extraction
            ground_truth_relations: Ground truth relations from BioRED
            extracted_relations: Relations extracted by the LLM

        Returns:
            ComparisonResult with detailed metrics and synonym match information
        """
        # Normalize ground truth and extracted relations for legacy output
        gt_set = self._normalize_relations(ground_truth_relations, is_ground_truth=True)
        ext_set = self._normalize_relations(extracted_relations, is_ground_truth=False)

        # Find matches with synonym awareness
        matched_gt_indices = set()
        matched_ext_indices = set()
        match_details = []

        for ext_idx, ext_rel in enumerate(extracted_relations):
            ext_tuple = self._relation_to_tuple(ext_rel)

            for gt_idx, gt_rel in enumerate(ground_truth_relations):
                if gt_idx in matched_gt_indices:
                    continue  # Already matched

                gt_tuple = self._relation_to_tuple(gt_rel)

                # Check if relations match
                if self._relations_match(ext_tuple, gt_tuple, ext_rel, gt_rel):
                    matched_gt_indices.add(gt_idx)
                    matched_ext_indices.add(ext_idx)

                    # Determine match types for entities
                    e1_match = self._get_match_type(ext_rel.entity1_text, gt_rel.entity1_text)
                    e2_match = self._get_match_type(ext_rel.entity2_text, gt_rel.entity2_text)

                    # Overall match type is the "weaker" of the two
                    match_type = max(e1_match, e2_match,
                                   key=lambda x: {"exact": 0, "synonym": 1, "fuzzy": 2}.get(x, 3))

                    match_details.append(MatchDetails(
                        ground_truth_e1=gt_rel.entity1_text,
                        ground_truth_e2=gt_rel.entity2_text,
                        extracted_e1=ext_rel.entity1_text,
                        extracted_e2=ext_rel.entity2_text,
                        relation_type=ext_rel.relation_type,
                        match_type=match_type
                    ))
                    break

        # Calculate metrics using deduplicated counts for consistency
        tp = len(matched_ext_indices)
        fp = len(ext_set) - tp  # Use deduplicated count
        fn = len(gt_set) - len(matched_gt_indices)  # Use deduplicated count

        precision = tp / len(ext_set) if ext_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        f_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return ComparisonResult(
            doc_id=doc_id,
            model_name=model_name,
            total_ground_truth=len(gt_set),  # Use deduplicated count
            total_extracted=len(ext_set),  # Use deduplicated count
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f_score=f_score,
            matched_relations=self._format_relations(gt_set & ext_set),
            missed_relations=self._format_relations(gt_set - ext_set),
            spurious_relations=self._format_relations(ext_set - gt_set),
            match_details=match_details
        )

    def _relation_to_tuple(self, rel) -> Tuple[str, str, str]:
        """Convert relation to tuple with sorted, normalized entity texts."""
        e1 = self._normalize_text(rel.entity1_text)
        e2 = self._normalize_text(rel.entity2_text)
        entities = tuple(sorted([e1, e2]))
        return (entities[0], entities[1], rel.relation_type)

    def _relations_match(self, ext_tuple, gt_tuple, ext_rel, gt_rel) -> bool:
        """Check if two relations match considering synonyms and fuzzy matching."""
        # Relation types must match exactly
        if ext_tuple[2] != gt_tuple[2]:
            return False

        # Check entity matches (accounting for sorted order)
        ext_e1, ext_e2 = ext_rel.entity1_text, ext_rel.entity2_text
        gt_e1, gt_e2 = gt_rel.entity1_text, gt_rel.entity2_text

        # Try both orderings (since tuples are sorted)
        match_direct = (
            self._get_match_type(ext_e1, gt_e1) != "none" and
            self._get_match_type(ext_e2, gt_e2) != "none"
        )

        match_swapped = (
            self._get_match_type(ext_e1, gt_e2) != "none" and
            self._get_match_type(ext_e2, gt_e1) != "none"
        )

        return match_direct or match_swapped

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
