"""CLI entry point for BioRED relation extraction evaluation."""

import argparse
import sys
from pathlib import Path

from src.data_loader import BioREDDataLoader
from src.openrouter_client import OpenRouterClient
from src.relation_comparator import RelationComparator
from src.csv_manager import CSVManager


def main():
    """Main entry point for the BioRED evaluation system."""
    parser = argparse.ArgumentParser(
        description="Evaluate biomedical relation extraction using BioRED dataset"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to BioRED-format JSON file"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="OpenRouter model name (e.g., 'anthropic/claude-3-sonnet')"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results.csv",
        help="Path to results CSV file (default: results.csv)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Initialize components
    try:
        loader = BioREDDataLoader()
        client = OpenRouterClient(args.model)
        csv_manager = CSVManager(Path(args.output))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load documents
    print(f"Loading documents from {args.input}...")
    try:
        documents = loader.load(args.input)
    except Exception as e:
        print(f"Error loading documents: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter to documents with relations
    docs_with_relations = [d for d in documents if d.has_relations()]
    print(f"Found {len(docs_with_relations)} documents with annotated relations")

    if not docs_with_relations:
        print("No documents with relations found. Exiting.")
        sys.exit(0)

    # Process each document
    for i, doc in enumerate(docs_with_relations, 1):
        print(f"\n[{i}/{len(docs_with_relations)}] Processing document {doc.doc_id}...")

        # Send to OpenRouter
        if args.verbose:
            print(f"  Sending to {args.model}...")
            print(f"  Document length: {len(doc.full_text)} characters")
            print(f"  Synonym sets: {len(doc.synonym_sets)}")

        extraction_result = client.extract_relations(doc.full_text)

        if not extraction_result.success:
            print(f"  ERROR: {extraction_result.error_message}")
            # Exit on 400 errors (invalid model) - no point continuing
            if "400 Bad Request" in extraction_result.error_message:
                sys.exit(1)
            continue

        if args.verbose:
            print(f"  Extracted {len(extraction_result.relations)} relations")

        # Compare results with synonym awareness
        comparator = RelationComparator(doc.synonym_sets)
        comparison = comparator.compare(
            doc_id=doc.doc_id,
            model_name=args.model,
            ground_truth_relations=doc.relations,
            extracted_relations=extraction_result.relations
        )

        # Save to CSV
        csv_manager.save_result(comparison)

        # Report
        print(f"  P={comparison.precision:.2%} R={comparison.recall:.2%} F1={comparison.f_score:.2%}")
        print(f"  TP={comparison.true_positives} FP={comparison.false_positives} FN={comparison.false_negatives}")

        # Show synonym match details
        if args.verbose and comparison.match_details:
            synonym_matches = [m for m in comparison.match_details if m.match_type == "synonym"]
            fuzzy_matches = [m for m in comparison.match_details if m.match_type == "fuzzy"]

            if synonym_matches:
                print(f"  Synonym matches: {len(synonym_matches)}")
                for match in synonym_matches[:3]:
                    print(f"    GT: <{match.ground_truth_e1}, {match.ground_truth_e2}>")
                    print(f"    EX: <{match.extracted_e1}, {match.extracted_e2}>")

            if fuzzy_matches:
                print(f"  Fuzzy matches: {len(fuzzy_matches)}")
                for match in fuzzy_matches[:3]:
                    print(f"    GT: <{match.ground_truth_e1}, {match.ground_truth_e2}>")
                    print(f"    EX: <{match.extracted_e1}, {match.extracted_e2}>")

        if args.verbose and comparison.true_positives > 0:
            print(f"  Matched relations: {len(comparison.matched_relations)}")
            for rel in comparison.matched_relations[:3]:  # Show first 3
                print(f"    {rel}")
            if len(comparison.matched_relations) > 3:
                print(f"    ... and {len(comparison.matched_relations) - 3} more")

    # Print aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)

    stats = csv_manager.get_aggregate_stats(args.model)

    if stats["count"] == 0:
        print("No results found.")
    else:
        print(f"Documents processed: {stats['count']}")
        print(f"Total True Positives: {stats['total_true_positives']}")
        print(f"Total False Positives: {stats['total_false_positives']}")
        print(f"Total False Negatives: {stats['total_false_negatives']}")
        print(f"\nMicro-Precision: {stats['micro_precision']:.2%}")
        print(f"Micro-Recall: {stats['micro_recall']:.2%}")
        print(f"Micro-F1: {stats['micro_f_score']:.2%}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
