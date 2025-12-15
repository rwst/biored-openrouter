# BioRED Relation Extraction Evaluation System

A Python-based evaluation system for biomedical relation extraction using the BioRED dataset schema. This system sends document passages to OpenRouter-hosted LLMs, extracts relations, compares results against ground truth annotations, and maintains a persistent CSV database of results.

## Features

- **Data Loading**: Parse BioRED-format JSON files with entities and relations
- **LLM Integration**: Extract relations using OpenRouter API (supports any model)
- **Evaluation**: Text-based matching with precision, recall, and F-score metrics
- **Persistence**: CSV database with automatic update/overwrite logic
- **Aggregate Statistics**: Micro-averaged metrics across multiple documents

## Installation

### Prerequisites

- Python 3.8+
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd biored-openrouter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```bash
python -m src.main --input data/test.json --model "anthropic/claude-3-sonnet"
```

### Command-Line Options

```
--input, -i    Path to BioRED-format JSON file (required)
--model, -m    OpenRouter model name (required)
--output, -o   Path to results CSV file (default: results.csv)
--verbose, -v  Enable verbose output
```

### Examples

**Evaluate with Claude 3 Sonnet:**
```bash
python -m src.main -i sample.json -m "anthropic/claude-3-sonnet"
```

**Evaluate with GPT-4 and custom output:**
```bash
python -m src.main -i data/biored_test.json -m "openai/gpt-4" -o my_results.csv
```

**Verbose mode:**
```bash
python -m src.main -i sample.json -m "meta-llama/llama-3-70b" -v
```

## Project Structure

```
biored-openrouter/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # BioRED JSON parser
│   ├── openrouter_client.py    # OpenRouter API client
│   ├── relation_comparator.py  # Evaluation logic
│   ├── csv_manager.py          # CSV database manager
│   └── main.py                 # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_openrouter_client.py
│   ├── test_relation_comparator.py
│   └── test_csv_manager.py
├── prompts/
│   └── relation_extraction.txt # LLM prompt template
├── requirements.txt
├── README.md
└── sample.json                 # Sample BioRED data
```

## BioRED Dataset Schema

### Entity Types

- **Gene**: Genes, proteins, mRNA, and other gene products
- **Disease**: Diseases, symptoms, syndromes, and disease-related phenotypes
- **Chemical**: Chemicals, drugs, and pharmacological substances
- **Variant**: Genomic/protein variants (substitutions, deletions, insertions)
- **Species**: Taxonomic species names (human, mouse, etc.)
- **CellLine**: Cell lines used in research

### Relation Types

1. **Positive_Correlation**: Entity1 positively affects/causes/increases Entity2
2. **Negative_Correlation**: Entity1 negatively affects/inhibits/decreases Entity2
3. **Association**: General association between entities
4. **Bind**: Physical binding interaction between entities
5. **Drug_Interaction**: Pharmacological interaction between drugs/chemicals
6. **Cotreatment**: Two entities used together in treatment
7. **Comparison**: Entities being compared in the study
8. **Conversion**: One entity converts/transforms into another

## Evaluation Methodology

The system follows the BioRED paper evaluation methodology:

- **Per-Relation Evaluation**: Each relation is evaluated independently
- **Text-Based Matching**: Entity pairs compared using normalized text (case-insensitive)
- **Symmetric Matching**: Entity order doesn't matter (A-B = B-A)
- **Metrics**:
  - Precision = TP / (TP + FP)
  - Recall = TP / (TP + FN)
  - F-score = 2 × (Precision × Recall) / (Precision + Recall)

## CSV Output Format

The system outputs results to a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| model_name | OpenRouter model identifier |
| doc_id | PubMed document ID |
| timestamp | ISO 8601 timestamp |
| total_ground_truth | Number of ground truth relations |
| total_extracted | Number of extracted relations |
| true_positives | Correctly extracted relations |
| false_positives | Incorrectly extracted relations |
| false_negatives | Missed relations |
| precision | Precision score (0-1) |
| recall | Recall score (0-1) |
| f_score | F-score (0-1) |
| matched_relations | JSON list of matched relations |
| missed_relations | JSON list of missed relations |
| spurious_relations | JSON list of false positive relations |

## Running Tests

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

Run specific test modules:

```bash
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/test_openrouter_client.py -v
python -m pytest tests/test_relation_comparator.py -v
python -m pytest tests/test_csv_manager.py -v
```

Run integration test:

```bash
python test_integration.py
```

## Example Output

```
Loading documents from sample.json...
Found 2 documents with annotated relations

[1/2] Processing document 15485686...
  P=66.67% R=11.11% F1=19.05%
  TP=2 FP=1 FN=16

[2/2] Processing document 30808312...
  P=75.00% R=20.00% F1=31.58%
  TP=3 FP=1 FN=12

============================================================
AGGREGATE RESULTS
============================================================
Documents processed: 2
Total True Positives: 5
Total False Positives: 2
Total False Negatives: 28

Micro-Precision: 71.43%
Micro-Recall: 15.15%
Micro-F1: 25.00%

Results saved to: results.csv
```

## Supported Models

This system works with any model available on OpenRouter. Popular choices include:

- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `anthropic/claude-3-haiku`
- `openai/gpt-4`
- `openai/gpt-3.5-turbo`
- `meta-llama/llama-3-70b`
- `google/gemini-pro`

See [OpenRouter models](https://openrouter.ai/models) for the complete list.

## Development

### Architecture

The system is organized into five main components:

1. **Data Loader** (`data_loader.py`): Parses BioRED JSON files and extracts entities/relations
2. **OpenRouter Client** (`openrouter_client.py`): Handles API communication with LLMs
3. **Relation Comparator** (`relation_comparator.py`): Computes evaluation metrics
4. **CSV Manager** (`csv_manager.py`): Maintains persistent results database
5. **Main CLI** (`main.py`): Orchestrates the entire pipeline

### Adding New Features

To extend the system:

- **New entity types**: Update `ENTITY_TYPE_MAP` in `data_loader.py`
- **New relation types**: Update prompt in `prompts/relation_extraction.txt`
- **Custom evaluation metrics**: Extend `RelationComparator` class
- **Alternative output formats**: Create new manager class similar to `CSVManager`

## Troubleshooting

### API Key Issues

```
Error: OPENROUTER_API_KEY environment variable not set
```

**Solution**: Set the environment variable:
```bash
export OPENROUTER_API_KEY="your-api-key"
```

### Rate Limiting

If you encounter rate limits, the system will report API errors. You can:
- Use a different model tier
- Add delays between requests (modify `main.py`)
- Process documents in smaller batches

### Invalid JSON Responses

The system handles markdown-wrapped JSON responses automatically. If you still encounter parsing errors:
- Check the raw response in verbose mode (`-v`)
- The error will be logged and the document skipped
- Consider adjusting the prompt temperature (currently 0.0)

## License

[Add your license here]

## Citation

If you use this system in your research, please cite the BioRED paper:

```
[Add BioRED citation]
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Contact

[Add contact information]
