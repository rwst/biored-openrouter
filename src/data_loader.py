"""BioRED data loader for parsing JSON files and extracting entities and relations."""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json


@dataclass
class Entity:
    """Represents a biomedical entity annotation."""
    id: str
    identifier: str          # Database ID
    entity_type: str         # Normalized type
    text: str                # Text mention
    offset: int
    length: int


@dataclass
class Relation:
    """Represents a relation between two entities."""
    id: str
    entity1_id: str          # References Entity.identifier
    entity2_id: str
    entity1_text: str        # Resolved text mention
    entity2_text: str
    relation_type: str
    novel: str               # Kept for reference but ignored in evaluation


@dataclass
class SynonymSet:
    """Represents all text variants for an entity identifier."""
    identifier: str
    entity_type: str
    texts: List[str]  # All observed text variants


@dataclass
class Document:
    """Represents a BioRED document with entities and relations."""
    doc_id: str              # PubMed ID
    full_text: str           # Concatenated passage texts
    entities: List[Entity]
    relations: List[Relation]
    synonym_sets: Dict[str, SynonymSet]  # identifier -> SynonymSet mapping

    def has_relations(self) -> bool:
        """Check if document has annotated relations."""
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
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for doc_data in data.get('documents', []):
            doc = self._parse_document(doc_data)
            documents.append(doc)

        return documents

    def _parse_document(self, doc_data: dict) -> Document:
        """Parse a single document from JSON."""
        doc_id = doc_data['id']

        # Extract and concatenate passage texts
        passages = doc_data.get('passages', [])
        full_text_parts = []
        all_entities = []

        for passage in passages:
            passage_text = passage.get('text', '')
            full_text_parts.append(passage_text)

            # Extract entities from this passage
            for annotation in passage.get('annotations', []):
                entity = self._parse_entity(annotation)
                all_entities.append(entity)

        full_text = ' '.join(full_text_parts)

        # Build synonym sets - collect ALL text variants per identifier
        synonym_dict = {}
        for entity in all_entities:
            if entity.identifier not in synonym_dict:
                synonym_dict[entity.identifier] = SynonymSet(
                    identifier=entity.identifier,
                    entity_type=entity.entity_type,
                    texts=[entity.text]
                )
            else:
                # Add new text variant if not already present
                if entity.text not in synonym_dict[entity.identifier].texts:
                    synonym_dict[entity.identifier].texts.append(entity.text)

        # Keep entity_lookup for backwards compatibility, using FIRST occurrence
        entity_lookup = {}
        for entity in all_entities:
            if entity.identifier not in entity_lookup:
                entity_lookup[entity.identifier] = entity

        # Parse relations
        relations = []
        for relation_data in doc_data.get('relations', []):
            relation = self._parse_relation(relation_data, entity_lookup)
            relations.append(relation)

        return Document(
            doc_id=doc_id,
            full_text=full_text,
            entities=all_entities,
            relations=relations,
            synonym_sets=synonym_dict
        )

    def _parse_entity(self, annotation: dict) -> Entity:
        """Parse an entity annotation."""
        infons = annotation.get('infons', {})
        locations = annotation.get('locations', [{}])
        location = locations[0] if locations else {}

        raw_type = infons.get('type', '')
        normalized_type = self._normalize_entity_type(raw_type)

        return Entity(
            id=annotation.get('id', ''),
            identifier=infons.get('identifier', ''),
            entity_type=normalized_type,
            text=annotation.get('text', ''),
            offset=location.get('offset', 0),
            length=location.get('length', 0)
        )

    def _parse_relation(self, relation_data: dict, entity_lookup: Dict[str, Entity]) -> Relation:
        """Parse a relation and resolve entity texts."""
        relation_id = relation_data.get('id', '')
        infons = relation_data.get('infons', {})

        entity1_id = infons.get('entity1', '')
        entity2_id = infons.get('entity2', '')
        relation_type = infons.get('type', '')
        novel = infons.get('novel', '')

        # Resolve entity texts
        entity1_text = self._resolve_entity_text(entity1_id, entity_lookup)
        entity2_text = self._resolve_entity_text(entity2_id, entity_lookup)

        return Relation(
            id=relation_id,
            entity1_id=entity1_id,
            entity2_id=entity2_id,
            entity1_text=entity1_text,
            entity2_text=entity2_text,
            relation_type=relation_type,
            novel=novel
        )

    def _resolve_entity_text(self, entity_id: str, entity_lookup: Dict[str, Entity]) -> str:
        """Find entity text by database identifier."""
        entity = entity_lookup.get(entity_id)
        if entity:
            return entity.text
        return ""

    def _normalize_entity_type(self, raw_type: str) -> str:
        """Map raw entity types to normalized names."""
        return self.ENTITY_TYPE_MAP.get(raw_type, raw_type)
