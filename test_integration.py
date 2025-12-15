"""Integration test for the full BioRED evaluation pipeline."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Set up environment
os.environ['OPENROUTER_API_KEY'] = 'test-key-for-integration'

from src.data_loader import BioREDDataLoader
from src.openrouter_client import OpenRouterClient
from src.relation_comparator import RelationComparator
from src.csv_manager import CSVManager


def test_full_pipeline():
    """Test the complete pipeline from data loading to CSV output."""

    print("="*60)
    print("INTEGRATION TEST: Full Pipeline")
    print("="*60)

    # Create temporary CSV
    temp_csv = Path(tempfile.mktemp(suffix='.csv'))

    # Step 1: Load documents
    print("\n1. Loading BioRED data...")
    loader = BioREDDataLoader()
    documents = loader.load('sample.json')
    docs_with_relations = [d for d in documents if d.has_relations()]
    print(f"   Loaded {len(docs_with_relations)} documents with relations")

    # Step 2: Mock OpenRouter API call
    print("\n2. Simulating OpenRouter extraction...")
    client = OpenRouterClient("test-model")

    # Create mock response based on first ground truth relation
    doc = docs_with_relations[0]
    print(f"   Processing document: {doc.doc_id}")
    print(f"   Ground truth relations: {len(doc.relations)}")

    # Mock API response - simulate extracting some of the ground truth relations
    mock_extracted = []
    if len(doc.relations) >= 2:
        # Correctly extract first relation
        rel1 = doc.relations[0]
        mock_extracted.append({
            "entity1_text": rel1.entity1_text,
            "entity1_type": "Gene",
            "entity2_text": rel1.entity2_text,
            "entity2_type": "Disease",
            "relation_type": rel1.relation_type
        })

        # Correctly extract second relation
        rel2 = doc.relations[1]
        mock_extracted.append({
            "entity1_text": rel2.entity1_text,
            "entity1_type": "Gene",
            "entity2_text": rel2.entity2_text,
            "entity2_type": "Disease",
            "relation_type": rel2.relation_type
        })

        # Add a false positive
        mock_extracted.append({
            "entity1_text": "FakeGene",
            "entity1_type": "Gene",
            "entity2_text": "FakeDisease",
            "entity2_type": "Disease",
            "relation_type": "Bind"
        })

    mock_response_content = json.dumps({"relations": mock_extracted})

    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": mock_response_content}
            }]
        }
        mock_post.return_value = mock_response

        extraction_result = client.extract_relations(doc.full_text)

    print(f"   Extracted relations: {len(extraction_result.relations)}")
    print(f"   Extraction successful: {extraction_result.success}")

    # Step 3: Compare results
    print("\n3. Comparing against ground truth...")
    comparator = RelationComparator()
    comparison = comparator.compare(
        doc_id=doc.doc_id,
        model_name="test-model",
        ground_truth_relations=doc.relations,
        extracted_relations=extraction_result.relations
    )

    print(f"   True Positives: {comparison.true_positives}")
    print(f"   False Positives: {comparison.false_positives}")
    print(f"   False Negatives: {comparison.false_negatives}")
    print(f"   Precision: {comparison.precision:.2%}")
    print(f"   Recall: {comparison.recall:.2%}")
    print(f"   F-score: {comparison.f_score:.2%}")

    # Step 4: Save to CSV
    print("\n4. Saving to CSV...")
    csv_manager = CSVManager(temp_csv)
    csv_manager.save_result(comparison)
    print(f"   Saved to: {temp_csv}")

    # Verify CSV contents
    rows = csv_manager._read_all()
    print(f"   CSV rows: {len(rows)}")

    # Step 5: Get aggregate stats
    print("\n5. Computing aggregate statistics...")
    stats = csv_manager.get_aggregate_stats("test-model")
    print(f"   Documents: {stats['count']}")
    print(f"   Micro-Precision: {stats['micro_precision']:.2%}")
    print(f"   Micro-Recall: {stats['micro_recall']:.2%}")
    print(f"   Micro-F1: {stats['micro_f_score']:.2%}")

    # Cleanup
    temp_csv.unlink()

    print("\n" + "="*60)
    print("INTEGRATION TEST: PASSED ✓")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        test_full_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"\nINTEGRATION TEST: FAILED ✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
