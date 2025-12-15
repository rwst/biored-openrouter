"""Tests for OpenRouter API client."""

import pytest
import os
import json
import requests
from pathlib import Path
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
    def test_extract_relations_handles_generic_markdown(self, mock_post, client):
        """Verify parser handles generic ``` wrapped responses."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '```\n{"relations": []}\n```'
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

    @patch('requests.post')
    def test_api_call_sets_temperature_zero(self, mock_post, client):
        """Verify API call uses temperature 0.0 for deterministic output."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"relations": []}'}}]
        }
        mock_post.return_value = mock_response

        client.extract_relations("Sample text")

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["temperature"] == 0.0

    @patch('requests.post')
    def test_empty_relations_list(self, mock_post, client):
        """Verify handling of empty relations list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"relations": []}'
                }
            }]
        }
        mock_post.return_value = mock_response

        result = client.extract_relations("Sample text")

        assert result.success == True
        assert len(result.relations) == 0

    @patch('requests.post')
    def test_multiple_relations_extracted(self, mock_post, client):
        """Verify multiple relations are correctly parsed."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "relations": [
                            {
                                "entity1_text": "Gene A",
                                "entity1_type": "Gene",
                                "entity2_text": "Disease B",
                                "entity2_type": "Disease",
                                "relation_type": "Association"
                            },
                            {
                                "entity1_text": "Chemical X",
                                "entity1_type": "Chemical",
                                "entity2_text": "Gene Y",
                                "entity2_type": "Gene",
                                "relation_type": "Bind"
                            }
                        ]
                    })
                }
            }]
        }
        mock_post.return_value = mock_response

        result = client.extract_relations("Sample text")

        assert result.success == True
        assert len(result.relations) == 2
        assert result.relations[0].relation_type == "Association"
        assert result.relations[1].relation_type == "Bind"

    @patch('requests.post')
    def test_missing_relations_key(self, mock_post, client):
        """Verify handling of JSON without 'relations' key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"data": []}'  # Wrong key
                }
            }]
        }
        mock_post.return_value = mock_response

        result = client.extract_relations("Sample text")

        # Should still succeed but with empty relations
        assert result.success == True
        assert len(result.relations) == 0

    @patch('requests.post')
    def test_http_error_handling(self, mock_post, client):
        """Verify HTTP errors are caught and handled."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response

        result = client.extract_relations("Sample text")

        assert result.success == False
        assert result.error_message is not None

    @patch('requests.post')
    def test_timeout_setting(self, mock_post, client):
        """Verify request has proper timeout set."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"relations": []}'}}]
        }
        mock_post.return_value = mock_response

        client.extract_relations("Sample text")

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["timeout"] == 120

    @patch('requests.post')
    def test_raw_response_stored(self, mock_post, client):
        """Verify raw response content is stored."""
        test_content = '{"relations": []}'
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": test_content}}]
        }
        mock_post.return_value = mock_response

        result = client.extract_relations("Sample text")

        assert test_content in result.raw_response

    def test_load_prompt_template_file_exists(self, mock_env):
        """Verify prompt template file can be loaded."""
        client = OpenRouterClient("test-model")
        # Should not raise an exception and template should contain placeholder
        assert "{document_text}" in client.prompt_template

    @patch('requests.post')
    def test_document_text_inserted_into_prompt(self, mock_post, client):
        """Verify document text is inserted into the prompt."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"relations": []}'}}]
        }
        mock_post.return_value = mock_response

        test_text = "This is my test document"
        client.extract_relations(test_text)

        call_kwargs = mock_post.call_args[1]
        prompt_sent = call_kwargs["json"]["messages"][0]["content"]
        assert test_text in prompt_sent

    @patch('requests.post')
    def test_error_response_saved_to_file(self, mock_post, client):
        """Verify failed JSON parsing writes response to error file."""
        invalid_json = '{"relations": [{"unterminated": "string'
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": invalid_json}
            }]
        }
        mock_post.return_value = mock_response

        result = client.extract_relations("Sample text")

        # Should fail
        assert result.success == False
        assert "Failed to parse response" in result.error_message

        # Error message should include filename
        assert "error_response_" in result.error_message
        assert ".txt" in result.error_message
        assert "Raw response saved to" in result.error_message

        # Extract filename from error message - handle both formats
        import re
        # Pattern matches both "error_response_20231215_123456.txt" and full paths
        match = re.search(r'(error_response_\d{8}_\d{6}\.txt)', result.error_message)
        assert match is not None
        filename = match.group(1)

        # Verify file was created
        error_file = Path(filename)
        assert error_file.exists()

        # Verify file contents
        assert error_file.read_text() == invalid_json

        # Cleanup
        error_file.unlink()
