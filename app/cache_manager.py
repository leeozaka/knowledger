"""
Semantic cache manager for RAG query optimization.

This module provides intelligent caching functionality that:
- Stores query-response pairs with semantic similarity matching
- Reduces response time for similar queries
- Manages cache invalidation based on document updates
- Provides cache performance statistics and monitoring
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import os
from dotenv import load_dotenv

from redis_client import RedisVectorStore
from embedding_interface import EmbeddingModel
from rag_core import RAGCore

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticCacheManager:
    """Manages semantic caching for RAG queries using similarity-based retrieval."""
    
    def __init__(self, 
                 redis_store: RedisVectorStore,
                 embedding_model: EmbeddingModel,
                 rag_core: RAGCore,
                 similarity_threshold: float = 0.7,
                 cache_ttl: int = 3600):
        """Initialize semantic cache manager.
        
        Args:
            redis_store: Redis vector store for cache storage
            embedding_model: Model for generating query embeddings
            rag_core: RAG core for generating new responses
            similarity_threshold: Minimum similarity for cache hits
            cache_ttl: Time-to-live for cache entries in seconds
        """
        self.redis_store = redis_store
        self.embedding_model = embedding_model
        self.rag_core = rag_core
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = cache_ttl
        
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
    
    def get_cached_response(self, user_query: str) -> Dict[str, Any]:
        """Get response for query, using cache when possible.
        
        Args:
            user_query: User's query string
            
        Returns:
            Dictionary containing response and cache metadata
        """
        self.total_queries += 1
        
        try:
            logger.info(f"Processing query with caching: {user_query[:100]}...")
            
            query_embedding = self.embedding_model.get_embedding(user_query)
            
            cache_result = self._search_cache(query_embedding, user_query)
            
            if cache_result:
                cached_response, similarity_score = cache_result
                self.cache_hits += 1
                
                logger.info(f"Cache hit with similarity: {similarity_score}")
                
                return {
                    "query": user_query,
                    "answer": cached_response,
                    "cache_hit": True,
                    "cache_similarity": similarity_score,
                    "processing_status": "cached_response",
                    "response_time": "fast",
                    "confidence_score": similarity_score,
                    "context_used": 0,
                    "total_chunks_found": 0,
                    "sources": [],
                    "source_doc_ids": []
                }
            
            self.cache_misses += 1
            logger.info("Cache miss, generating new response")
            
            rag_response = self.rag_core.get_rag_response(user_query)
            
            if rag_response.get("processing_status") == "success":
                self._store_in_cache(
                    query_text=user_query,
                    query_embedding=query_embedding,
                    response=rag_response["answer"],
                    source_document_ids=rag_response.get("source_doc_ids")
                )
            
            rag_response.update({
                "cache_hit": False,
                "cache_similarity": 0.0,
                "response_time": "normal"
            })
            
            return rag_response
            
        except Exception as e:
            logger.error(f"Error in cached response generation: {e}")
            return {
                "query": user_query,
                "answer": "I apologize, but I encountered an error while processing your query. Please try again.",
                "cache_hit": False,
                "cache_similarity": 0.0,
                "processing_status": "error",
                "error": str(e),
                "confidence_score": 0.0,
                "context_used": 0,
                "total_chunks_found": 0,
                "sources": [],
                "source_doc_ids": []
            }
    
    def _search_cache(self, 
                     query_embedding: List[float], 
                     query_text: str) -> Optional[Tuple[str, float]]:
        """Search cache for semantically similar queries.
        
        Args:
            query_embedding: Embedding vector for the query
            query_text: Original query text for logging
            
        Returns:
            Tuple of (cached_response, similarity_score) if found, None otherwise
        """
        try:
            result = self.redis_store.search_cache(
                query_embedding=query_embedding,
                similarity_threshold=self.similarity_threshold
            )
            
            if result:
                cached_response, similarity_score = result
                logger.info(f"Found cached response with similarity: {similarity_score}")
                return cached_response, similarity_score
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching cache: {e}")
            return None
    
    def _store_in_cache(self, 
                       query_text: str, 
                       query_embedding: List[float], 
                       response: str,
                       source_document_ids: Optional[List[str]] = None) -> bool:
        """Store a query-response pair in the cache.
        
        Args:
            query_text: Original query text
            query_embedding: Query embedding vector
            response: Generated response
            source_document_ids: Optional list of source document IDs used for the response
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.redis_store.store_cache_entry(
                query_text=query_text,
                query_embedding=query_embedding,
                response=response,
                source_document_ids=source_document_ids
            )
            
            if success:
                logger.info(f"Cached response for query: {query_text[:50]}...")
            else:
                logger.warning(f"Failed to cache response for query: {query_text[:50]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        hit_rate = (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0
        miss_rate = (self.cache_misses / self.total_queries * 100) if self.total_queries > 0 else 0
        
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percentage": round(hit_rate, 2),
            "miss_rate_percentage": round(miss_rate, 2),
            "similarity_threshold": self.similarity_threshold,
            "cache_ttl": self.cache_ttl
        }
    
    def clear_cache(self) -> bool:
        """Clear all cached entries.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.redis_store.clear_all_data()
            
            if success:
                logger.info("Cache cleared successfully")
                self.cache_hits = 0
                self.cache_misses = 0
                self.total_queries = 0
            
            return success
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def update_similarity_threshold(self, new_threshold: float) -> None:
        """Update the similarity threshold for cache hits.
        
        Args:
            new_threshold (float): New similarity threshold (0.0 to 1.0)
            
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
        """
        if 0.0 <= new_threshold <= 1.0:
            old_threshold = self.similarity_threshold
            self.similarity_threshold = new_threshold
            logger.info(f"Updated similarity threshold from {old_threshold} to {new_threshold}")
        else:
            logger.warning(f"Invalid similarity threshold: {new_threshold}. Must be between 0.0 and 1.0")
    
    def invalidate_cache_for_documents(self, doc_ids: List[str]) -> int:
        """Invalidate all cache entries that used specified documents as sources.

        Args:
            doc_ids (List[str]): List of document IDs whose related cache entries should be cleared.

        Returns:
            int: The number of cache entries invalidated.
        """
        if not doc_ids:
            logger.info("No document IDs provided for cache invalidation.")
            return 0
        
        try:
            logger.info(f"Attempting to invalidate cache entries for doc_ids: {doc_ids}")
            invalidated_count = self.redis_store.invalidate_cache_by_doc_ids(doc_ids)
            logger.info(f"Successfully invalidated {invalidated_count} cache entries for doc_ids: {doc_ids}")
            return invalidated_count
        except Exception as e:
            logger.error(f"Error during cache invalidation for documents {doc_ids}: {e}")
            return 0
    
    def preload_cache_from_queries(self, query_response_pairs: List[Tuple[str, str]]) -> int:
        """Preload cache with known query-response pairs.
        
        Args:
            query_response_pairs (List[Tuple[str, str]]): List of (query, response) tuples
            
        Returns:
            int: Number of successfully cached pairs
        """
        cached_count = 0
        
        for query, response in query_response_pairs:
            try:
                query_embedding = self.embedding_model.get_embedding(query)
                
                if self._store_in_cache(query, query_embedding, response):
                    cached_count += 1
                
            except Exception as e:
                logger.error(f"Error preloading cache entry for query '{query[:50]}...': {e}")
                continue
        
        logger.info(f"Preloaded {cached_count} out of {len(query_response_pairs)} cache entries")
        return cached_count 