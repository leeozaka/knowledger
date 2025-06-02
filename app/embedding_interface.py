"""
Embedding model interface and implementations.

This module provides an abstract interface for embedding models and concrete
implementations for different embedding providers. It ensures a pluggable
architecture for switching between different embedding models.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract base class for embedding models to ensure pluggable interface."""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of embeddings produced by this model.
        
        Returns:
            Integer dimension of the embedding vectors
        """
        pass


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Sentence Transformers embedding model."""
    
    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1"):
        """Initialize with a Sentence Transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text.strip():
            return [0.0] * self.dimension
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        """Return the dimension of embeddings produced by this model.
        
        Returns:
            Integer dimension of the embedding vectors
        """
        return self.dimension 