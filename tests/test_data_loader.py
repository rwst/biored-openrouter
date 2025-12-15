"""Tests for BioRED data loader."""

import pytest
import json
import tempfile
from pathlib import Path
from src.data_loader import BioREDDataLoader, Document, Entity, Relation


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

    @pytest.fixture
    def multi_passage_json_path(self):
        """Create a temporary JSON file with multiple passages."""
        sample_data = {
            "source": "PubTator",
            "date": "2021-11-30",
            "key": "BioC.key",
            "documents": [
                {
                    "id": "67890",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "First passage with Chemical X.",
                            "annotations": [
                                {
                                    "id": "0",
                                    "infons": {"identifier": "CHEM001", "type": "ChemicalEntity"},
                                    "text": "Chemical X",
                                    "locations": [{"offset": 19, "length": 10}]
                                }
                            ]
                        },
                        {
                            "offset": 32,
                            "text": "Second passage mentions Gene Y.",
                            "annotations": [
                                {
                                    "id": "1",
                                    "infons": {"identifier": "GENE002", "type": "GeneOrGeneProduct"},
                                    "text": "Gene Y",
                                    "locations": [{"offset": 56, "length": 6}]
                                }
                            ]
                        }
                    ],
                    "relations": [
                        {
                            "id": "R0",
                            "infons": {
                                "entity1": "CHEM001",
                                "entity2": "GENE002",
                                "type": "Bind",
                                "novel": "No"
                            }
                        }
                    ]
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            return f.name

    @pytest.fixture
    def no_relations_json_path(self):
        """Create a temporary JSON file with no relations."""
        sample_data = {
            "source": "PubTator",
            "date": "2021-11-30",
            "key": "BioC.key",
            "documents": [
                {
                    "id": "99999",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "Just some text.",
                            "annotations": []
                        }
                    ],
                    "relations": []
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

    def test_multi_passage_concatenation(self, multi_passage_json_path):
        """Verify multiple passages are concatenated with spaces."""
        loader = BioREDDataLoader()
        docs = loader.load(multi_passage_json_path)
        full_text = docs[0].full_text
        assert "First passage with Chemical X." in full_text
        assert "Second passage mentions Gene Y." in full_text
        # Check they are separated by space
        assert "Chemical X. Second passage" in full_text

    def test_entities_parsed_correctly(self, sample_json_path):
        """Verify entities are extracted with correct attributes."""
        loader = BioREDDataLoader()
        docs = loader.load(sample_json_path)
        entities = docs[0].entities
        assert len(entities) == 2
        assert entities[0].text == "Gene A"
        assert entities[0].entity_type == "Gene"  # Normalized
        assert entities[0].identifier == "GENE001"
        assert entities[1].text == "Disease B"
        assert entities[1].entity_type == "Disease"  # Normalized

    def test_relations_with_resolved_text(self, sample_json_path):
        """Verify relations have entity text resolved."""
        loader = BioREDDataLoader()
        docs = loader.load(sample_json_path)
        relations = docs[0].relations
        assert len(relations) == 1
        assert relations[0].entity1_text == "Gene A"
        assert relations[0].entity2_text == "Disease B"
        assert relations[0].relation_type == "Association"
        assert relations[0].entity1_id == "GENE001"
        assert relations[0].entity2_id == "DISEASE001"

    def test_has_relations_filter(self, sample_json_path):
        """Verify has_relations() returns True for documents with relations."""
        loader = BioREDDataLoader()
        docs = loader.load(sample_json_path)
        assert docs[0].has_relations() == True

    def test_has_relations_false_for_empty(self, no_relations_json_path):
        """Verify has_relations() returns False for documents without relations."""
        loader = BioREDDataLoader()
        docs = loader.load(no_relations_json_path)
        assert docs[0].has_relations() == False

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
        assert loader._normalize_entity_type("SequenceVariant") == "Variant"
        assert loader._normalize_entity_type("OrganismTaxon") == "Species"
        assert loader._normalize_entity_type("CellLine") == "CellLine"

    def test_unknown_entity_type_passthrough(self):
        """Verify unknown entity types are passed through unchanged."""
        loader = BioREDDataLoader()
        assert loader._normalize_entity_type("UnknownType") == "UnknownType"

    def test_entity_identifier_resolution(self):
        """Verify entity text is resolved by identifier, not annotation id."""
        loader = BioREDDataLoader()
        docs = loader.load(Path(__file__).parent.parent / "sample.json")
        # The sample.json should have relations that reference entity identifiers
        if docs and docs[0].has_relations():
            relation = docs[0].relations[0]
            # Verify the relation has resolved text
            assert relation.entity1_text != ""
            assert relation.entity2_text != ""

    def test_multiple_documents(self):
        """Verify loading multiple documents from single JSON file."""
        sample_data = {
            "source": "PubTator",
            "date": "2021-11-30",
            "key": "BioC.key",
            "documents": [
                {
                    "id": "111",
                    "passages": [{"offset": 0, "text": "Doc 1", "annotations": []}],
                    "relations": []
                },
                {
                    "id": "222",
                    "passages": [{"offset": 0, "text": "Doc 2", "annotations": []}],
                    "relations": []
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_path = f.name

        loader = BioREDDataLoader()
        docs = loader.load(temp_path)
        assert len(docs) == 2
        assert docs[0].doc_id == "111"
        assert docs[1].doc_id == "222"

    def test_relation_novel_field_preserved(self, sample_json_path):
        """Verify the 'novel' field is preserved in relations."""
        loader = BioREDDataLoader()
        docs = loader.load(sample_json_path)
        assert docs[0].relations[0].novel == "Novel"
