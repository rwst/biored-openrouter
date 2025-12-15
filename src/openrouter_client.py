"""OpenRouter API client for biomedical relation extraction."""

import os
import json
import requests
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class ExtractedRelation:
    """Represents a relation extracted by the LLM."""
    entity1_text: str
    entity1_type: str
    entity2_text: str
    entity2_type: str
    relation_type: str


@dataclass
class ExtractionResult:
    """Result of a relation extraction API call."""
    success: bool
    relations: List[ExtractedRelation]
    raw_response: str
    error_message: Optional[str] = None


class OpenRouterClient:
    """Client for OpenRouter API to extract relations."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "relation_extraction.txt"

    def __init__(self, model_name: str):
        """
        Initialize the OpenRouter client.

        Args:
            model_name: The OpenRouter model identifier (e.g., 'anthropic/claude-3-sonnet')

        Raises:
            ValueError: If OPENROUTER_API_KEY environment variable is not set
        """
        self.model_name = model_name
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load prompt template from file."""
        with open(self.PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read()

    def extract_relations(self, document_text: str) -> ExtractionResult:
        """
        Send document to OpenRouter and extract relations.

        Synchronous call - waits for response.

        Args:
            document_text: The document text to analyze

        Returns:
            ExtractionResult containing extracted relations or error information
        """
        prompt = self.prompt_template.replace("{document_text}", document_text)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/biored-eval",
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
        """
        Parse JSON response from LLM.

        Handles markdown code blocks and extracts relation data.

        Args:
            content: Raw response content from the LLM

        Returns:
            ExtractionResult with parsed relations or error information
        """
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
