# BioRED Relation Extraction Evaluation System
## Comprehensive Implementation Plan

---

## Executive Summary

This plan details the implementation of a Python-based evaluation system for biomedical relation extraction using the BioRED dataset schema. The system will send document passages to OpenRouter-hosted LLMs, extract relations, compare results against ground truth annotations, and maintain a persistent CSV database of results.

**Key Metrics (from BioRED paper):**
- Evaluation is performed **per-relation** (each relation is an individual prediction)
- Standard precision, recall, and F-score metrics
- Entity pairs compared using text-based matching (not database identifiers)

---

## Project Structure

```
biored_eval/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Phase 1: JSON parsing
│   ├── openrouter_client.py    # Phase 2: API integration
│   ├── relation_comparator.py  # Phase 3: Result comparison
│   ├── csv_manager.py          # Phase 4: CSV database
│   └── main.py                 # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_openrouter_client.py
│   ├── test_relation_comparator.py
│   └── test_csv_manager.py
├── prompts/
│   └── relation_extraction.txt
├── requirements.txt
└── README.md
```

---

## Phase 1: Data Loading

### 1.1 Objective
Read and parse BioRED-format JSON files, validating the schema and extracting document passages with their annotated entities and relations.

### 1.2 JSON Schema Understanding

Based on the sample.json analysis:

```python
# Document structure
{
    "source": str,
    "date": str,
    "key": str,
    "documents": [
        {
            "id": str,                    # PubMed ID (e.g., "15485686")
            "passages": [
                {
                    "offset": int,
                    "text": str,
                    "annotations": [
                        {
                            "id": str,
                            "infons": {
                                "identifier": str,  # Database ID
                                "type": str         # Entity type
                            },
                            "text": str,            # Entity text mention
                            "locations": [{"offset": int, "length": int}]
                        }
                    ]
                }
            ],
            "relations": [
                {
                    "id": str,              # e.g., "R0"
                    "infons": {
                        "entity1": str,     # References annotation identifier
                        "entity2": str,
                        "type": str,        # Relation type
                        "novel": str        # "Novel" or "No" (ignored per requirements)
                    }
                }
            ]
        }
    ]
}
```

### 1.3 Entity Types (from BioRED)
- `GeneOrGeneProduct` → Gene
- `DiseaseOrPhenotypicFeature` → Disease
- `ChemicalEntity` → Chemical
- `SequenceVariant` → Variant
- `OrganismTaxon` → Species
- `CellLine` → CellLine

### 1.4 Relation Types (from BioRED Figure 2B)
1. **Positive_Correlation** - Entities have a positive relationship
2. **Negative_Correlation** - Entities have an inhibiting/negative relationship
3. **Association** - General association between entities
4. **Bind** - Physical binding interaction
5. **Drug_Interaction** - Drug-drug interaction
6. **Cotreatment** - Entities used together in treatment
7. **Comparison** - Entities being compared
8. **Conversion** - One entity converts to another

### 1.5 Implementation Details

```python
# src/data_loader.py

from dataclasses import dataclass
from typing import List, Dict, Optional
import json

@dataclass
class Entity:
    id: str
    identifier: str          # Database ID
    entity_type: str         # Normalized type
    text: str                # Text mention
    offset: int
    length: int

@dataclass
class Relation:
    id: str
    entity1_id: str          # References Entity.identifier
    entity2_id: str
    entity1_text: str        # Resolved text mention
    entity2_text: str
    relation_type: str
    novel: str               # Kept for reference but ignored in evaluation

@dataclass
class Document:
    doc_id: str              # PubMed ID
    full_text: str           # Concatenated passage texts
    entities: List[Entity]
    relations: List[Relation]
    
    def has_relations(self) -> bool:
        return len(self.relations) > 0

class BioREDDataLoader:
    """Load and parse BioRED-format JSON files."""
    
    ENTITY_TYPE_MAP = {
        "GeneOrGeneProduct": "Gene",
        "DiseaseOrPhenotypicFeature": "Disease",
        "ChemicalEntity": "Chemical",
        "SequenceVariant": "Variant",
        "OrganismTaxon": "Species",
        "CellLine": "CellLine"
    }
    
    def load(self, json_path: str) -> List[Document]:
        """Load JSON file and return list of Document objects."""
        pass
    
    def _parse_document(self, doc_data: dict) -> Document:
        """Parse a single document from JSON."""
        pass
    
    def _resolve_entity_text(self, entity_id: str, entities: List[Entity]) -> str:
        """Find entity text by database identifier."""
        pass
    
    def _normalize_entity_type(self, raw_type: str) -> str:
        """Map raw entity types to normalized names."""
        pass
```

### 1.6 Test Specification: `test_data_loader.py`

```python
import pytest
import json
import tempfile
from src.data_loader import BioREDDataLoader, Document

class TestBioREDDataLoader:
    
    @pytest.fixture
    def sample_json_path(self):
        """Create a temporary JSON file with sample BioRED data."""
        sample_data = {
            "source": "PubTator",
            "date": "2021-11-30",
            "key": "BioC.key",
            "documents": [
                {
                    "id": "12345",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "Gene A is associated with Disease B.",
                            "annotations": [
                                {
                                    "id": "0",
                                    "infons": {"identifier": "GENE001", "type": "GeneOrGeneProduct"},
                                    "text": "Gene A",
                                    "locations": [{"offset": 0, "length": 6}]
                                },
                                {
                                    "id": "1",
                                    "infons": {"identifier": "DISEASE001", "type": "DiseaseOrPhenotypicFeature"},
                                    "text": "Disease B",
                                    "locations": [{"offset": 27, "length": 9}]
                                }
                            ]
                        }
                    ],
                    "relations": [
                        {
                            "id": "R0",
                            "infons": {
                                "entity1": "GENE001",
                                "entity2": "DISEASE001",
                                "type": "Association",
                                "novel": "Novel"
                            }
                        }
                    ]
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            return f.name
    
    def test_load_returns_documents(self, sample_json_path):
        """Verify loader returns list of Document objects."""
        loader = BioREDDataLoader()
        documents = loader.load(sample_json_path)
        assert isinstance(documents, list)
        assert len(documents) == 1
        assert isinstance(documents[0], Document)
    
    def test_document_has_correct_id(self, sample_json_path):
        """Verify document ID is correctly extracted."""
        loader = BioREDDataLoader()
        docs = loader.load(sample_json_path)
        assert docs[0].doc_id == "12345"
    
    def test_full_text_concatenation(self, sample_json_path):
        """Verify passages are concatenated into full_text."""
        loader = BioREDDataLoader()
        docs = loader.load(sample_json_path)
        assert "Gene A is associated with Disease B." in docs[0].full_text
    
    def test_entities_parsed_correctly(self, sample_json_path):
        """Verify entities are extracted with correct attributes."""
        loader = BioREDDataLoader()
        docs = loader.load(sample_json_path)
        entities = docs[0].entities
        assert len(entities) == 2
        assert entities[0].text == "Gene A"
        assert entities[0].entity_type == "Gene"  # Normalized
    
    def test_relations_with_resolved_text(self, sample_json_path):
        """Verify relations have entity text resolved."""
        loader = BioREDDataLoader()
        docs = loader.load(sample_json_path)
        relations = docs[0].relations
        assert len(relations) == 1
        assert relations[0].entity1_text == "Gene A"
        assert relations[0].entity2_text == "Disease B"
        assert relations[0].relation_type == "Association"
    
    def test_has_relations_filter(self, sample_json_path):
        """Verify has_relations() returns True for documents with relations."""
        loader = BioREDDataLoader()
        docs = loader.load(sample_json_path)
        assert docs[0].has_relations() == True
    
    def test_invalid_json_raises_error(self):
        """Verify appropriate error for invalid JSON."""
        loader = BioREDDataLoader()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json")
            f.flush()
            with pytest.raises(json.JSONDecodeError):
                loader.load(f.name)
    
    def test_entity_type_normalization(self):
        """Verify all entity types are correctly normalized."""
        loader = BioREDDataLoader()
        assert loader._normalize_entity_type("GeneOrGeneProduct") == "Gene"
        assert loader._normalize_entity_type("DiseaseOrPhenotypicFeature") == "Disease"
        assert loader._normalize_entity_type("ChemicalEntity") == "Chemical"
```

### 1.7 Acceptance Criteria
- [ ] Successfully loads valid BioRED JSON files
- [ ] Correctly parses all 6 entity types
- [ ] Concatenates multiple passages into coherent full_text
- [ ] Resolves entity identifiers to text mentions for relations
- [ ] Filters documents to only those with annotated relations
- [ ] Handles edge cases (empty relations, missing fields) gracefully
- [ ] All 8+ unit tests pass

---

## Phase 2: OpenRouter API Integration

### 2.1 Objective
Send document passages to OpenRouter-hosted LLMs with a carefully crafted prompt requesting structured JSON output of extracted relations.

### 2.2 Environment Configuration
```bash
# Required environment variable
export OPENROUTER_API_KEY="your-api-key-here"
```

### 2.3 Prompt Design

The prompt must clearly communicate:
1. The task definition
2. Entity types to identify
3. Relation types to extract
4. Output format requirements

**Prompt Template (`prompts/relation_extraction.txt`):**

```
You are a biomedical relation extraction system. Your task is to identify and extract relationships between biomedical entities from scientific text.

## Entity Types to Identify:
- Gene: genes, proteins, mRNA, and other gene products
- Disease: diseases, symptoms, syndromes, and disease-related phenotypes
- Chemical: chemicals, drugs, and pharmacological substances
- Variant: genomic/protein variants (substitutions, deletions, insertions)
- Species: taxonomic species names (human, mouse, etc.)
- CellLine: cell lines used in research

## Relation Types to Extract:
1. Positive_Correlation: Entity1 positively affects/causes/increases Entity2
2. Negative_Correlation: Entity1 negatively affects/inhibits/decreases Entity2 (includes treatment relationships)
3. Association: General association between entities without clear directionality
4. Bind: Physical binding interaction between entities
5. Drug_Interaction: Pharmacological interaction between drugs/chemicals
6. Cotreatment: Two entities used together in treatment
7. Comparison: Entities being compared in the study
8. Conversion: One entity converts/transforms into another

## Valid Entity Pair Combinations:
- Disease-Gene, Disease-Chemical, Disease-Variant
- Gene-Chemical, Gene-Gene
- Chemical-Variant, Chemical-Chemical
- Variant-Variant

## Rules:
1. Only extract EXPLICIT relations stated in the text
2. Do NOT infer implicit relationships
3. Each relation must have exactly two entities
4. Entity text should match EXACTLY as it appears in the document
5. Relation types must be one of the 8 types listed above
6. Do not include species as part of relation pairs
7. For relations involving variants, include the variant notation exactly as written

## Output Format:
Return ONLY a valid JSON object with this structure:
{
  "relations": [
    {
      "entity1_text": "exact text of first entity",
      "entity1_type": "Gene|Disease|Chemical|Variant|Species|CellLine",
      "entity2_text": "exact text of second entity",
      "entity2_type": "Gene|Disease|Chemical|Variant|Species|CellLine",
      "relation_type": "Positive_Correlation|Negative_Correlation|Association|Bind|Drug_Interaction|Cotreatment|Comparison|Conversion"
    }
  ]
}

If no relations are found, return: {"relations": []}

## Document Text:
{document_text}

## Extracted Relations (JSON only):
```

### 2.4 Implementation Details

```python
# src/openrouter_client.py

import os
import json
import requests
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class ExtractedRelation:
    entity1_text: str
    entity1_type: str
    entity2_text: str
    entity2_type: str
    relation_type: str

@dataclass
class ExtractionResult:
    success: bool
    relations: List[ExtractedRelation]
    raw_response: str
    error_message: Optional[str] = None

class OpenRouterClient:
    """Client for OpenRouter API to extract relations."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "relation_extraction.txt"
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from file."""
        with open(self.PROMPT_PATH, 'r') as f:
            return f.read()
    
    def extract_relations(self, document_text: str) -> ExtractionResult:
        """
        Send document to OpenRouter and extract relations.
        Synchronous call - waits for response.
        """
        prompt = self.prompt_template.replace("{document_text}", document_text)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/biored-eval",  # Assumption
            "X-Title": "BioRED Relation Extraction Evaluation"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,  # Deterministic output
            "max_tokens": 4096
        }
        
        try:
            response = requests.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_response(content)
            
        except requests.RequestException as e:
            return ExtractionResult(
                success=False,
                relations=[],
                raw_response="",
                error_message=str(e)
            )
    
    def _parse_response(self, content: str) -> ExtractionResult:
        """Parse JSON response from LLM."""
        try:
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            
            relations = [
                ExtractedRelation(
                    entity1_text=r["entity1_text"],
                    entity1_type=r["entity1_type"],
                    entity2_text=r["entity2_text"],
                    entity2_type=r["entity2_type"],
                    relation_type=r["relation_type"]
                )
                for r in data.get("relations", [])
            ]
            
            return ExtractionResult(
                success=True,
                relations=relations,
                raw_response=content
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            return ExtractionResult(
                success=False,
                relations=[],
                raw_response=content,
                error_message=f"Failed to parse response: {e}"
            )
```

### 2.5 Test Specification: `test_openrouter_client.py`

```python
import pytest
import os
import json
from unittest.mock import patch, MagicMock
from src.openrouter_client import OpenRouterClient, ExtractionResult, ExtractedRelation

class TestOpenRouterClient:
    
    @pytest.fixture
    def mock_env(self):
        """Set up mock environment variable."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            yield
    
    @pytest.fixture
    def client(self, mock_env):
        """Create client with mocked prompt template."""
        with patch.object(OpenRouterClient, '_load_prompt_template', 
                         return_value="Test prompt {document_text}"):
            return OpenRouterClient("test-model")
    
    def test_init_requires_api_key(self):
        """Verify client raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                OpenRouterClient("test-model")
    
    def test_init_stores_model_name(self, client):
        """Verify model name is stored correctly."""
        assert client.model_name == "test-model"
    
    @patch('requests.post')
    def test_extract_relations_success(self, mock_post, client):
        """Verify successful extraction returns correct structure."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "relations": [{
                            "entity1_text": "Gene A",
                            "entity1_type": "Gene",
                            "entity2_text": "Disease B",
                            "entity2_type": "Disease",
                            "relation_type": "Association"
                        }]
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = client.extract_relations("Sample text")
        
        assert result.success == True
        assert len(result.relations) == 1
        assert result.relations[0].entity1_text == "Gene A"
        assert result.relations[0].relation_type == "Association"
    
    @patch('requests.post')
    def test_extract_relations_handles_markdown_wrapper(self, mock_post, client):
        """Verify parser handles ```json``` wrapped responses."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '```json\n{"relations": []}\n```'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = client.extract_relations("Sample text")
        
        assert result.success == True
        assert result.relations == []
    
    @patch('requests.post')
    def test_extract_relations_network_error(self, mock_post, client):
        """Verify network errors are handled gracefully."""
        mock_post.side_effect = requests.RequestException("Network error")
        
        result = client.extract_relations("Sample text")
        
        assert result.success == False
        assert "Network error" in result.error_message
    
    @patch('requests.post')
    def test_extract_relations_invalid_json(self, mock_post, client):
        """Verify invalid JSON responses are handled."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "not valid json"}
            }]
        }
        mock_post.return_value = mock_response
        
        result = client.extract_relations("Sample text")
        
        assert result.success == False
        assert "Failed to parse" in result.error_message
    
    @patch('requests.post')
    def test_api_call_uses_correct_headers(self, mock_post, client):
        """Verify API call includes required headers."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"relations": []}'}}]
        }
        mock_post.return_value = mock_response
        
        client.extract_relations("Sample text")
        
        call_kwargs = mock_post.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
    
    @patch('requests.post')
    def test_api_call_uses_specified_model(self, mock_post, client):
        """Verify API call uses the specified model name."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"relations": []}'}}]
        }
        mock_post.return_value = mock_response
        
        client.extract_relations("Sample text")
        
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "test-model"
```

### 2.6 Acceptance Criteria
- [ ] Reads API key from environment variable
- [ ] Constructs valid API requests with model name from argparse
- [ ] Sends document text within prompt template
- [ ] Waits synchronously for response
- [ ] Parses JSON response correctly (including markdown-wrapped)
- [ ] Handles network errors gracefully
- [ ] Handles invalid JSON responses gracefully
- [ ] All 8+ unit tests pass

---

## Phase 3: Response Parsing & Comparison

### 3.1 Objective
Compare extracted relations against ground truth annotations using text-based matching, computing precision, recall, and F-score per the BioRED evaluation methodology.

### 3.2 Evaluation Methodology (from BioRED Paper)

**Per-Relation Evaluation:**
- Each relation is evaluated independently
- True Positive (TP): Extracted relation matches ground truth
- False Positive (FP): Extracted relation not in ground truth
- False Negative (FN): Ground truth relation not extracted

**Matching Criteria:**
- Entity text must match (case-insensitive, normalized whitespace)
- Relation type must match exactly
- Entity order is symmetric (entity1-entity2 = entity2-entity1)

**Metrics:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F-score = 2 * (Precision * Recall) / (Precision + Recall)
```

### 3.3 Implementation Details

```python
# src/relation_comparator.py

from dataclasses import dataclass
from typing import List, Set, Tuple
import re

@dataclass
class ComparisonResult:
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
        ground_truth_relations: List['Relation'],  # From data_loader
        extracted_relations: List['ExtractedRelation']  # From openrouter_client
    ) -> ComparisonResult:
        """
        Compare extracted relations to ground truth.
        Returns detailed comparison results.
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
        """Normalize entity text for comparison."""
        # Lowercase
        text = text.lower()
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _format_relations(self, relation_set: Set[Tuple]) -> List[str]:
        """Format relation tuples as readable strings."""
        return [f"<{r[0]}, {r[1]}, {r[2]}>" for r in relation_set]
```

### 3.4 Test Specification: `test_relation_comparator.py`

```python
import pytest
from src.relation_comparator import RelationComparator, ComparisonResult
from dataclasses import dataclass

# Mock data classes for testing
@dataclass
class MockGroundTruthRelation:
    id: str
    entity1_text: str
    entity2_text: str
    relation_type: str

@dataclass
class MockExtractedRelation:
    entity1_text: str
    entity1_type: str
    entity2_text: str
    entity2_type: str
    relation_type: str

class TestRelationComparator:
    
    @pytest.fixture
    def comparator(self):
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
    
    def test_symmetric_entity_matching(self, comparator):
        """Verify entity order doesn't matter (symmetric relations)."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = [MockExtractedRelation("Disease B", "Disease", "Gene A", "Gene", "Association")]
        
        result = comparator.compare("doc1", "model1", gt, ext)
        
        assert result.true_positives == 1
    
    def test_whitespace_normalization(self, comparator):
        """Verify whitespace is normalized in entity names."""
        gt = [MockGroundTruthRelation("R0", "Gene  A", "Disease   B", "Association")]
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")]
        
        result = comparator.compare("doc1", "model1", gt, ext)
        
        assert result.true_positives == 1
    
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
    
    def test_empty_ground_truth(self, comparator):
        """Verify handling of empty ground truth."""
        gt = []
        ext = [MockExtractedRelation("Gene A", "Gene", "Disease B", "Disease", "Association")]
        
        result = comparator.compare("doc1", "model1", gt, ext)
        
        assert result.recall == 0.0  # No ground truth to match
        assert result.false_positives == 1
    
    def test_empty_extraction(self, comparator):
        """Verify handling of empty extraction."""
        gt = [MockGroundTruthRelation("R0", "Gene A", "Disease B", "Association")]
        ext = []
        
        result = comparator.compare("doc1", "model1", gt, ext)
        
        assert result.precision == 0.0  # No extractions
        assert result.false_negatives == 1
    
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
```

### 3.5 Acceptance Criteria
- [ ] Correctly identifies true positives, false positives, false negatives
- [ ] Case-insensitive entity text matching
- [ ] Symmetric entity pair matching (order independent)
- [ ] Whitespace normalization
- [ ] Exact relation type matching
- [ ] Correct precision, recall, F-score calculation
- [ ] Handles edge cases (empty sets, duplicates)
- [ ] All 10+ unit tests pass

---

## Phase 4: CSV Database Management

### 4.1 Objective
Maintain a persistent CSV file (`results.csv`) that stores evaluation results, updating/overwriting rows based on model name and document ID combination.

### 4.2 CSV Schema

```csv
model_name,doc_id,timestamp,total_ground_truth,total_extracted,true_positives,false_positives,false_negatives,precision,recall,f_score,matched_relations,missed_relations,spurious_relations
```

**Column Descriptions:**
| Column | Type | Description |
|--------|------|-------------|
| model_name | str | OpenRouter model identifier |
| doc_id | str | PubMed document ID |
| timestamp | str | ISO format timestamp of evaluation |
| total_ground_truth | int | Number of ground truth relations |
| total_extracted | int | Number of extracted relations |
| true_positives | int | Correctly extracted relations |
| false_positives | int | Incorrectly extracted relations |
| false_negatives | int | Missed relations |
| precision | float | Precision score (0-1) |
| recall | float | Recall score (0-1) |
| f_score | float | F-score (0-1) |
| matched_relations | str | JSON-encoded list of matched relation descriptions |
| missed_relations | str | JSON-encoded list of missed relation descriptions |
| spurious_relations | str | JSON-encoded list of spurious relation descriptions |

### 4.3 Update Logic
- If row with same (model_name, doc_id) exists: **overwrite**
- If row doesn't exist: **append**
- File is created if it doesn't exist

### 4.4 Implementation Details

```python
# src/csv_manager.py

import csv
import json
import os
from datetime import datetime
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
        self.csv_path = csv_path or self.DEFAULT_PATH
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create CSV with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
    
    def save_result(self, result: ComparisonResult) -> None:
        """
        Save or update a comparison result.
        Overwrites existing row with same (model_name, doc_id).
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
        """Convert ComparisonResult to CSV row dict."""
        return {
            "model_name": result.model_name,
            "doc_id": result.doc_id,
            "timestamp": datetime.utcnow().isoformat(),
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
        """Read all rows from CSV."""
        rows = []
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows
    
    def _write_all(self, rows: list) -> None:
        """Write all rows to CSV."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    
    def get_aggregate_stats(self, model_name: Optional[str] = None) -> dict:
        """
        Calculate aggregate statistics across all documents.
        Optionally filter by model name.
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
```

### 4.5 Test Specification: `test_csv_manager.py`

```python
import pytest
import csv
import json
import tempfile
from pathlib import Path
from src.csv_manager import CSVManager
from src.relation_comparator import ComparisonResult

class TestCSVManager:
    
    @pytest.fixture
    def temp_csv_path(self):
        """Create a temporary CSV file path."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            return Path(f.name)
    
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
        
        with open(temp_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            assert "model_name" in reader.fieldnames
            assert "doc_id" in reader.fieldnames
            assert "precision" in reader.fieldnames
    
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
```

### 4.6 Acceptance Criteria
- [ ] Creates CSV file with headers if not exists
- [ ] Appends new rows for new (model, doc_id) combinations
- [ ] Overwrites existing rows for same (model, doc_id)
- [ ] Correct numeric precision in output
- [ ] JSON-encodes list fields
- [ ] Includes timestamp in ISO format
- [ ] Calculates aggregate statistics correctly
- [ ] All 9+ unit tests pass

---

## Phase 5: CLI Entry Point

### 5.1 Main Script Implementation

```python
# src/main.py

import argparse
import sys
from pathlib import Path

from src.data_loader import BioREDDataLoader
from src.openrouter_client import OpenRouterClient
from src.relation_comparator import RelationComparator
from src.csv_manager import CSVManager

def main():
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
    loader = BioREDDataLoader()
    client = OpenRouterClient(args.model)
    comparator = RelationComparator()
    csv_manager = CSVManager(Path(args.output))
    
    # Load documents
    print(f"Loading documents from {args.input}...")
    documents = loader.load(args.input)
    
    # Filter to documents with relations
    docs_with_relations = [d for d in documents if d.has_relations()]
    print(f"Found {len(docs_with_relations)} documents with annotated relations")
    
    # Process each document
    for i, doc in enumerate(docs_with_relations, 1):
        print(f"\n[{i}/{len(docs_with_relations)}] Processing document {doc.doc_id}...")
        
        # Send to OpenRouter
        if args.verbose:
            print(f"  Sending to {args.model}...")
        
        extraction_result = client.extract_relations(doc.full_text)
        
        if not extraction_result.success:
            print(f"  ERROR: {extraction_result.error_message}")
            continue
        
        if args.verbose:
            print(f"  Extracted {len(extraction_result.relations)} relations")
        
        # Compare results
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
    
    # Print aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    
    stats = csv_manager.get_aggregate_stats(args.model)
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
```

### 5.2 Usage Examples

```bash
# Basic usage
python -m src.main --input data/test.json --model "anthropic/claude-3-sonnet"

# With verbose output
python -m src.main -i data/biored_test.json -m "openai/gpt-4" -v

# Custom output file
python -m src.main -i data/sample.json -m "meta-llama/llama-3-70b" -o my_results.csv
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|--------------|
| Phase 1: Data Loading | 2-3 hours | None | `data_loader.py`, tests passing |
| Phase 2: OpenRouter Client | 2-3 hours | Phase 1 | `openrouter_client.py`, prompt file, tests passing |
| Phase 3: Comparator | 2-3 hours | Phases 1-2 | `relation_comparator.py`, tests passing |
| Phase 4: CSV Manager | 1-2 hours | Phase 3 | `csv_manager.py`, tests passing |
| Phase 5: CLI Integration | 1-2 hours | All phases | `main.py`, end-to-end testing |
| **Total** | **8-13 hours** | | |

---

## Requirements.txt

```
requests>=2.28.0
pytest>=7.0.0
python-dateutil>=2.8.0
```

---

## Areas Requiring Clarification

The following variables and preferences need to be defined to tailor this plan to your specific needs:

### 1. **Entity Text Matching Strategy**

**Current assumption:** Case-insensitive, whitespace-normalized exact matching.

**Why it matters:** The BioRED paper evaluates using database identifiers (concept IDs), but your requirement specifies text-based comparison. More sophisticated matching (fuzzy, stemming, synonym expansion) would significantly change Phase 3 implementation complexity and accuracy.

**Questions:**
- Should "IL-2" match "interleukin 2" or "IL2"?
- How to handle abbreviations vs. full names?
- What similarity threshold, if any, for fuzzy matching?

### 2. **OpenRouter API Specifics**

**Current assumption:** Standard OpenRouter chat completions endpoint with default rate limits.

**Why it matters:** Rate limits, retry logic, and timeout handling affect reliability.

**Questions:**
- What is the expected request volume? (affects batching strategy)
- Should there be exponential backoff retry logic?
- Any specific OpenRouter tier/plan constraints?

### 3. **Error Handling Philosophy**

**Current assumption:** Log errors and continue processing remaining documents.

**Why it matters:** Affects robustness and debugging.

**Questions:**
- Should the script fail fast on first error?
- Should failed documents be retried automatically?
- How to handle partial API responses?

### 4. **Prompt Tuning Approach**

**Current assumption:** Single static prompt template.

**Why it matters:** Different models may require different prompting strategies.

**Questions:**
- Should the prompt be configurable per model?
- Is there a prompt version tracking requirement?
- Should few-shot examples be included?

### 5. **CSV Location and Permissions**

**Current assumption:** CSV file in current working directory with read/write access.

**Why it matters:** Affects deployment and concurrent usage.

**Questions:**
- Is the CSV shared across multiple users/runs?
- Should file locking be implemented for concurrent access?
- Any specific file path requirements?

### 6. **Progress and Logging**

**Current assumption:** Basic stdout progress reporting.

**Why it matters:** Affects monitoring and debugging in production.

**Questions:**
- Should there be structured logging (JSON format)?
- Is there a specific log level requirement?
- Should progress be saved for resumable runs?

### 7. **Test Data Availability**

**Current assumption:** Tests use synthetic mock data.

**Why it matters:** Integration tests with real BioRED data would be more reliable.

**Questions:**
- Is there a test subset of BioRED data to use?
- Should integration tests call the real OpenRouter API?
- What is the acceptable test execution time?

### 8. **Relation Type Normalization**

**Current assumption:** Exact string matching for relation types (e.g., "Positive_Correlation").

**Why it matters:** LLMs may produce variations like "PositiveCorrelation" or "positive correlation".

**Questions:**
- Should relation types be normalized (lowercase, no underscores)?
- Is there a mapping table for common variations?

### 9. **Multi-Passage Documents**

**Current assumption:** All passages in a document are concatenated with a single space separator.

**Why it matters:** Sentence boundary detection and cross-sentence relations depend on this.

**Questions:**
- What separator between passages? (space, newline, paragraph break)
- Should passage offsets be preserved for debugging?

### 10. **Evaluation Scope**

**Current assumption:** Evaluate all 8 relation types equally.

**Why it matters:** Some relation types are much rarer (e.g., Variant-Variant at 0%).

**Questions:**
- Should there be per-relation-type breakdowns in output?
- Are any relation types to be excluded from evaluation?
- Should the CSV include per-relation-type columns?

---

## Appendix A: BioRED Relation Types Distribution (from paper)

| Relation Type | Percentage |
|---------------|------------|
| Association | 52% |
| Positive_Correlation | 27% |
| Negative_Correlation | 17% |
| Triple Relations (Cotreatment, etc.) | 2% |
| Bind | <1% |
| Drug_Interaction | rare |
| Comparison | rare |
| Conversion | rare |

This distribution should inform expectations for model performance across relation types.

---

## Appendix B: Expected CLI Output

```
$ python -m src.main -i sample.json -m "anthropic/claude-3-sonnet" -v

Loading documents from sample.json...
Found 2 documents with annotated relations

[1/2] Processing document 15485686...
  Sending to anthropic/claude-3-sonnet...
  Extracted 12 relations
  P=75.00% R=60.00% F1=66.67%
  TP=9 FP=3 FN=6

[2/2] Processing document 18497585...
  Sending to anthropic/claude-3-sonnet...
  Extracted 3 relations
  P=100.00% R=100.00% F1=100.00%
  TP=3 FP=0 FN=0

============================================================
AGGREGATE RESULTS
============================================================
Documents processed: 2
Total True Positives: 12
Total False Positives: 3
Total False Negatives: 6

Micro-Precision: 80.00%
Micro-Recall: 66.67%
Micro-F1: 72.73%

Results saved to: results.csv
```
