# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a BioRED Relation Extraction Evaluation System - a Python tool that evaluates biomedical relation extraction performance using LLMs via the OpenRouter API. The system sends scientific document text to an LLM, extracts biomedical relations, compares results against ground truth annotations from the BioRED dataset, and maintains a persistent CSV database of evaluation results.

## Core Architecture

The system follows a linear pipeline with five main components:

1. **Data Loader** (`src/data_loader.py`): Parses BioRED-format JSON files
   - Extracts entities and relations from multi-passage documents
   - Key insight: Relations reference entities by `identifier` (database ID), not by annotation `id`
   - Normalizes entity types: `GeneOrGeneProduct` → `Gene`, `DiseaseOrPhenotypicFeature` → `Disease`, etc.
   - Concatenates all passage texts into `full_text` for LLM consumption

2. **OpenRouter Client** (`src/openrouter_client.py`): LLM API integration
   - Loads prompt template from `prompts/relation_extraction.txt`
   - Substitutes `{document_text}` placeholder in template
   - Uses `temperature=0.0` for deterministic outputs
   - Handles markdown-wrapped JSON responses (strips ` ```json ` blocks)
   - Requires `OPENROUTER_API_KEY` environment variable

3. **Relation Comparator** (`src/relation_comparator.py`): Evaluation logic
   - Normalizes entity text: lowercase + whitespace normalization
   - **Symmetric matching**: Entity order doesn't matter (A-B matches B-A)
   - Converts relations to tuples: `(sorted_entity1, sorted_entity2, relation_type)`
   - Computes precision, recall, F-score using set operations

4. **CSV Manager** (`src/csv_manager.py`): Results persistence
   - Primary key: `(model_name, doc_id)`
   - Upsert logic: Overwrites existing rows with same key
   - Stores relation lists as JSON-encoded strings
   - Computes micro-averaged aggregate statistics across documents
   - **Important**: Uses `datetime.now(timezone.utc)` (not deprecated `utcnow()`)

5. **Main CLI** (`src/main.py`): Entry point and orchestration
   - Coordinates all components in sequence
   - Filters to documents with `has_relations() == True`
   - Displays per-document and aggregate metrics

## Data Models

**Key dataclasses to understand:**

- `Entity`: Has `identifier` (database ID like "MESH:D001234") and `text` (mention text)
- `Relation`: References entities by `identifier`, stores resolved `entity1_text`/`entity2_text`
- `ExtractedRelation`: LLM output - only has text and types, no identifiers
- `ComparisonResult`: Complete evaluation metrics + matched/missed/spurious relation lists

**Critical distinction**: Ground truth relations use identifiers for linking, but evaluation compares only the text mentions (case-insensitive, symmetric).

## BioRED Schema

**Entity Types**: Gene, Disease, Chemical, Variant, Species, CellLine

**Relation Types**: Positive_Correlation, Negative_Correlation, Association, Bind, Drug_Interaction, Cotreatment, Comparison, Conversion

The prompt template in `prompts/relation_extraction.txt` documents all valid entity pair combinations.

## Development Commands

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific test module
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/test_openrouter_client.py -v
python -m pytest tests/test_relation_comparator.py -v
python -m pytest tests/test_csv_manager.py -v

# Integration test
python test_integration.py
```

### Running the System
```bash
# Set API key (required)
export OPENROUTER_API_KEY="your-api-key-here"

# Basic run
python -m src.main --input data/test.json --model "anthropic/claude-3-sonnet"

# With custom output and verbose mode
python -m src.main -i sample.json -m "anthropic/claude-3-sonnet" -o results.csv -v
```

## Testing Approach

- All components have isolated unit tests with mocked dependencies
- Tests use pytest fixtures extensively (`@pytest.fixture`)
- API calls are mocked with `unittest.mock.patch`
- Temporary files for CSV tests use `tempfile.NamedTemporaryFile`
- Integration test (`test_integration.py`) validates the full pipeline end-to-end

## Common Pitfalls

1. **Entity Resolution**: Relations link by `identifier` (e.g., "MESH:D001234"), not by `id` (e.g., "T1")
2. **Symmetric Matching**: Always sort entity texts before comparison to ensure A-B matches B-A
3. **Text Normalization**: Apply lowercase + whitespace normalization consistently
4. **API Key**: System fails with clear error if `OPENROUTER_API_KEY` not set
5. **JSON Parsing**: Handle markdown code blocks (` ```json ... ``` `) in LLM responses
6. **Datetime**: Use `datetime.now(timezone.utc)` to avoid deprecation warnings

## File Locations

- Source code: `src/*.py`
- Tests: `tests/test_*.py` (unit tests) + `test_integration.py` (integration)
- Prompt template: `prompts/relation_extraction.txt`
- Sample data: `sample.json` (BioRED format)
- Results: `results.csv` (default output location)
