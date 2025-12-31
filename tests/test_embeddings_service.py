# FILE: tests/test_embeddings_service.py
"""
Tests for app/embeddings/service.py
Vector embeddings - generates and manages embeddings.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock
import math


class TestEmbeddingsServiceImports:
    """Test embeddings service module structure."""

    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.embeddings import service
        assert service is not None

    def test_core_functions_exist(self):
        """Test core functions are defined."""
        from app.embeddings.service import (
            generate_embedding,
            chunk_text,
            cosine_similarity,
            search_embeddings,
            store_embedding,
        )
        assert callable(generate_embedding)
        assert callable(chunk_text)
        assert callable(cosine_similarity)
        assert callable(search_embeddings)
        assert callable(store_embedding)


class TestEmbeddingGeneration:
    """Test embedding generation."""

    def test_empty_text_returns_none(self):
        """Test empty text returns None."""
        from app.embeddings.service import generate_embedding

        assert generate_embedding("") is None
        assert generate_embedding("   ") is None

    def test_none_text_returns_none(self):
        """Test None text handled gracefully."""
        from app.embeddings.service import generate_embedding

        # Function should handle None without crashing
        result = generate_embedding(None)
        assert result is None

    def test_missing_api_key_returns_none(self):
        """Test missing API key returns None gracefully."""
        from app.embeddings.service import generate_embedding

        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            result = generate_embedding("test text")
            assert result is None


class TestChunkText:
    """Test text chunking functionality."""

    def test_empty_text_returns_empty(self):
        """Test empty text returns empty list."""
        from app.embeddings.service import chunk_text
        assert chunk_text("") == []

    def test_none_text_returns_empty(self):
        """Test None text returns empty list."""
        from app.embeddings.service import chunk_text
        assert chunk_text(None) == []

    def test_short_text_single_chunk(self):
        """Test short text returns single chunk."""
        from app.embeddings.service import chunk_text
        text = "This is a short text."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        """Test long text is split into multiple chunks."""
        from app.embeddings.service import chunk_text, CHUNK_SIZE

        # Create text longer than default chunk size
        # CHUNK_SIZE is in tokens (~4 chars each), so multiply
        words = ["word"] * (CHUNK_SIZE * 2)
        text = " ".join(words)

        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_custom_chunk_size(self):
        """Test custom chunk size parameter."""
        from app.embeddings.service import chunk_text

        # Create text that would be multiple chunks
        text = " ".join(["word"] * 500)

        # NOTE: chunk_size must be > overlap to avoid infinite loop bug
        # When chunk_size < overlap, the loop never terminates
        chunks_small = chunk_text(text, chunk_size=100, overlap=20)
        chunks_large = chunk_text(text, chunk_size=400, overlap=50)

        # Smaller chunk size should produce more chunks
        assert len(chunks_small) >= len(chunks_large)

    def test_chunks_preserve_content(self):
        """Test chunking preserves all words."""
        from app.embeddings.service import chunk_text

        words = ["word1", "word2", "word3", "word4", "word5"]
        text = " ".join(words)

        chunks = chunk_text(text)
        # All words should appear in at least one chunk
        all_chunk_text = " ".join(chunks)
        for word in words:
            assert word in all_chunk_text


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_identical_vectors_similarity_one(self):
        """Test identical vectors have similarity 1.0."""
        from app.embeddings.service import cosine_similarity

        vec = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001

    def test_orthogonal_vectors_similarity_zero(self):
        """Test orthogonal vectors have similarity 0.0."""
        from app.embeddings.service import cosine_similarity

        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        sim = cosine_similarity(vec_a, vec_b)
        assert abs(sim - 0.0) < 0.0001

    def test_opposite_vectors_similarity_negative(self):
        """Test opposite vectors have similarity -1.0."""
        from app.embeddings.service import cosine_similarity

        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        sim = cosine_similarity(vec_a, vec_b)
        assert abs(sim - (-1.0)) < 0.0001

    def test_different_length_vectors_return_zero(self):
        """Test mismatched vector lengths return 0."""
        from app.embeddings.service import cosine_similarity

        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.0, 2.0]
        sim = cosine_similarity(vec_a, vec_b)
        assert sim == 0.0

    def test_zero_vector_returns_zero(self):
        """Test zero vectors return 0 similarity."""
        from app.embeddings.service import cosine_similarity

        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec_a, vec_b)
        assert sim == 0.0

    def test_normalized_vectors(self):
        """Test similarity with normalized vectors."""
        from app.embeddings.service import cosine_similarity

        # 45-degree angle vectors
        vec_a = [1.0, 0.0]
        vec_b = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        sim = cosine_similarity(vec_a, vec_b)
        # cos(45) ~ 0.707
        assert abs(sim - 0.707) < 0.01

    def test_high_dimensional_vectors(self):
        """Test similarity with high-dimensional vectors."""
        from app.embeddings.service import cosine_similarity

        # Simulate embedding-like vectors
        vec_a = [0.1] * 1536
        vec_b = [0.1] * 1536
        sim = cosine_similarity(vec_a, vec_b)
        assert abs(sim - 1.0) < 0.0001


class TestEmbeddingModels:
    """Test embedding model configuration."""

    def test_openai_embeddings(self):
        """Test OpenAI embedding model is configured."""
        from app.embeddings.service import EMBEDDING_MODEL
        assert EMBEDDING_MODEL == "text-embedding-3-small"

    def test_local_embeddings(self):
        """Test embedding dimensions match model."""
        from app.embeddings.service import EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS == 1536


class TestEmbeddingCache:
    """Test embedding caching behavior."""

    def test_embedding_exists_callable(self):
        """Test embedding_exists function is callable."""
        from app.embeddings.service import embedding_exists
        assert callable(embedding_exists)

    def test_delete_embeddings_callable(self):
        """Test delete_embeddings_for_source is callable."""
        from app.embeddings.service import delete_embeddings_for_source
        assert callable(delete_embeddings_for_source)

    def test_index_functions_accept_force_param(self):
        """Test indexing functions accept force parameter."""
        from app.embeddings.service import index_note, index_message, index_document
        import inspect
        
        # Verify force parameter exists in signatures
        note_sig = inspect.signature(index_note)
        assert "force" in note_sig.parameters
        assert note_sig.parameters["force"].default == False
        
        msg_sig = inspect.signature(index_message)
        assert "force" in msg_sig.parameters
        assert msg_sig.parameters["force"].default == False
        
        doc_sig = inspect.signature(index_document)
        assert "force" in doc_sig.parameters
        assert doc_sig.parameters["force"].default == False


class TestEmbeddingConfiguration:
    """Test embedding configuration constants."""

    def test_embedding_model_defined(self):
        """Test embedding model is configured."""
        from app.embeddings.service import EMBEDDING_MODEL
        assert EMBEDDING_MODEL is not None
        assert len(EMBEDDING_MODEL) > 0

    def test_embedding_dimensions_defined(self):
        """Test embedding dimensions is configured."""
        from app.embeddings.service import EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS > 0
        assert EMBEDDING_DIMENSIONS == 1536

    def test_chunk_size_defined(self):
        """Test chunk size is configured."""
        from app.embeddings.service import CHUNK_SIZE
        assert CHUNK_SIZE > 0
        assert CHUNK_SIZE == 400

    def test_chunk_overlap_less_than_size(self):
        """Test chunk overlap is less than chunk size."""
        from app.embeddings.service import CHUNK_SIZE, CHUNK_OVERLAP
        assert CHUNK_OVERLAP < CHUNK_SIZE
        assert CHUNK_OVERLAP == 50


class TestSearchCompatibility:
    """Test search function alias."""

    def test_search_alias_exists(self):
        """Test search function alias exists."""
        from app.embeddings.service import search
        assert callable(search)

    def test_search_embeddings_exists(self):
        """Test search_embeddings function exists."""
        from app.embeddings.service import search_embeddings
        assert callable(search_embeddings)


class TestStorageFunctions:
    """Test embedding storage functions."""

    def test_store_embedding_exists(self):
        """Test store_embedding function exists."""
        from app.embeddings.service import store_embedding
        assert callable(store_embedding)

    def test_delete_embeddings_for_source_exists(self):
        """Test delete function exists."""
        from app.embeddings.service import delete_embeddings_for_source
        assert callable(delete_embeddings_for_source)

    def test_get_embeddings_for_project_exists(self):
        """Test get function exists."""
        from app.embeddings.service import get_embeddings_for_project
        assert callable(get_embeddings_for_project)

    def test_embedding_exists_function(self):
        """Test embedding_exists function exists."""
        from app.embeddings.service import embedding_exists
        assert callable(embedding_exists)


class TestIndexingFunctions:
    """Test indexing functions."""

    def test_index_note_exists(self):
        """Test index_note function exists."""
        from app.embeddings.service import index_note
        assert callable(index_note)

    def test_index_message_exists(self):
        """Test index_message function exists."""
        from app.embeddings.service import index_message
        assert callable(index_message)

    def test_index_document_exists(self):
        """Test index_document function exists."""
        from app.embeddings.service import index_document
        assert callable(index_document)

    def test_index_project_exists(self):
        """Test index_project function exists."""
        from app.embeddings.service import index_project
        assert callable(index_project)

    def test_reindex_project_exists(self):
        """Test reindex_project function exists."""
        from app.embeddings.service import reindex_project
        assert callable(reindex_project)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
