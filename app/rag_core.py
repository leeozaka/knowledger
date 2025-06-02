"""
Core RAG (Retrieval Augmented Generation) engine implementation.

This module provides the main RAG functionality including:
- Document retrieval using vector similarity search
- Context building from retrieved chunks
- Response generation using Ollama LLM
- Confidence scoring and source tracking
- Health monitoring and debugging utilities
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import os
import requests
import json
from dotenv import load_dotenv

from redis_client import RedisVectorStore
from embedding_interface import EmbeddingModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGCore:
    """Core RAG (Retrieval Augmented Generation) engine using Ollama with DeepSeek."""
    
    def __init__(self, 
                 redis_store: RedisVectorStore,
                 embedding_model: EmbeddingModel,
                 similarity_threshold: float = 0.3,
                 max_context_chunks: int = 5,
                 ollama_host: str = None,
                 ollama_port: int = None,
                 ollama_model: str = None):
        """Initialize RAG core with required components.
        
        Args:
            redis_store: Redis vector store for document retrieval
            embedding_model: Model for generating embeddings
            similarity_threshold: Minimum similarity score for relevant chunks
            max_context_chunks: Maximum number of chunks to include in context
            ollama_host: Ollama server host (defaults to localhost)
            ollama_port: Ollama server port (defaults to 11434)
            ollama_model: Ollama model to use (defaults to gemma3:1b)
        """
        self.redis_store = redis_store
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_context_chunks = max_context_chunks
        
        self.ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = ollama_port or int(os.getenv('OLLAMA_PORT', 11434))
        self.ollama_model = ollama_model or os.getenv('OLLAMA_MODEL', 'gemma3:1b')
        self.ollama_url = f"http://{self.ollama_host}:{self.ollama_port}/api/generate"
        
        self._test_ollama_connection()
    
    def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama server.
        
        Returns:
            True if connection successful, raises exception otherwise
        """
        try:
            test_url = f"http://{self.ollama_host}:{self.ollama_port}/"
            response = requests.get(test_url, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama at {self.ollama_host}:{self.ollama_port}")
            return True
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to Ollama at {self.ollama_host}:{self.ollama_port}: {e}"
            logger.error(error_msg)
            raise ConnectionError(f"Ollama server is required but unavailable. {error_msg}")
    
    def get_rag_response(self, 
                        user_query: str, 
                        include_sources: bool = True) -> Dict[str, Any]:
        """Generate a response to user query using RAG pipeline.
        
        Args:
            user_query: User's question or query
            include_sources: Whether to include source information in response
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"Processing query: {user_query[:100]}...")
            
            query_embedding = self.embedding_model.get_embedding(user_query)
            
            retrieved_chunks = self.retrieve_relevant_chunks(
                query_embedding=query_embedding,
                k_neighbors=self.max_context_chunks
            )
            
            relevant_chunks = [
                chunk for chunk in retrieved_chunks 
                if (1 - chunk['score']) >= self.similarity_threshold
            ]
            
            logger.info(f"Retrieved {len(retrieved_chunks)} total chunks, {len(relevant_chunks)} passed similarity threshold of {self.similarity_threshold}")
            if retrieved_chunks and not relevant_chunks:
                best_score = max([1 - chunk['score'] for chunk in retrieved_chunks])
                logger.warning(f"No chunks passed threshold. Best similarity score was {best_score:.3f} (threshold: {self.similarity_threshold})")
            
            if not relevant_chunks:
                return self._generate_no_context_response(user_query)
            
            context = self._build_context(relevant_chunks)
            
            llm_response = self._ollama_llm_call(user_query, context)
            
            response = {
                "query": user_query,
                "answer": llm_response,
                "context_used": len(relevant_chunks),
                "total_chunks_found": len(retrieved_chunks),
                "confidence_score": self._calculate_confidence_score(relevant_chunks),
                "processing_status": "success"
            }
            
            source_details, source_doc_ids = self._extract_source_info(relevant_chunks)
            response["source_doc_ids"] = source_doc_ids

            if include_sources:
                response["sources"] = source_details
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
            return {
                "query": user_query,
                "answer": "I apologize, but I encountered an error while processing your query. Please try again.",
                "context_used": 0,
                "total_chunks_found": 0,
                "confidence_score": 0.0,
                "processing_status": "error",
                "error": str(e)
            }
    
    def retrieve_relevant_chunks(self, 
                               query_embedding: List[float], 
                               k_neighbors: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks using KNN search.
        
        Args:
            query_embedding: Embedding vector for the query
            k_neighbors: Number of nearest neighbors to retrieve
            
        Returns:
            List of relevant document chunks with scores
        """
        try:
            results = self.redis_store.knn_search(
                query_vector=query_embedding,
                k_neighbors=k_neighbors
            )
            
            logger.info(f"Retrieved {len(results)} chunks from vector store")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks.
        
        Args:
            chunks (List[Dict[str, Any]]): List of retrieved document chunks
            
        Returns:
            str: Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text_content', '')
            doc_id = chunk.get('doc_id', 'unknown')
            similarity = 1 - chunk.get('score', 1)
            
            context_part = f"[Source {i+1} - {doc_id} (similarity: {similarity:.2f})]:\n{text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _ollama_llm_call(self, query: str, context: str) -> str:
        """Call Ollama for generating responses.
        
        Args:
            query (str): User query
            context (str): Retrieved context from documents
            
        Returns:
            str: Generated response string
            
        Raises:
            TimeoutError: If Ollama request times out
            ConnectionError: If unable to connect to Ollama server
            RuntimeError: If Ollama API call fails
            ValueError: If Ollama returns empty response
        """

        logger.info(f"Generating response for query: {query[:50]}...")
        logger.info(f"Context: {context}")

        try:
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from documents. Please provide accurate, helpful answers based solely on the information given in the context.

IMPORTANT: You must carefully analyze the provided context to determine if it contains sufficient information to answer the user's question. If the context contains relevant information, use it to provide a comprehensive answer. If the context does not contain enough information or is not relevant to the question, clearly state that the context doesn't contain sufficient information.
IMPORTANT: Don't need to mention the context in your response. The context is for you to use to answer the question. You can use the context to answer the question, but you don't need to mention the context in your response.

Context from documents:
{context}

User Question: {query}

Instructions:
1. First, analyze whether the provided context contains information relevant to answering the question
2. If relevant information is found, provide a comprehensive answer based on that information
3. If the context doesn't contain sufficient relevant information, clearly state this fact
4. Always cite which sources you're referencing when possible
5. Be specific about what information is available and what is missing"""

            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 2048,
                }
            }
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=120,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            
            response_data = response.json()
            generated_text = response_data.get('response', '').strip()
            
            if not generated_text:
                raise ValueError("Ollama returned empty response")
            
            logger.info(f"Successfully generated Ollama response for query: {query[:50]}...")
            return generated_text
            
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out. The model may be overloaded.")
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to Ollama server. Please ensure Ollama is running.")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Ollama API: {e}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Ollama response: {e}")
            
        except Exception as e:
            raise RuntimeError(f"Unexpected error in Ollama call: {e}")
    
    def debug_query_processing(self, user_query: str) -> Dict[str, Any]:
        """Debug the query processing pipeline to identify issues.
        
        Args:
            user_query (str): User's question or query
            
        Returns:
            Dict[str, Any]: Debug information about the query processing including embedding, retrieval, filtering, context building, and Ollama connection status
        """
        debug_info = {
            "query": user_query,
            "steps": {}
        }
        
        try:
            query_embedding = self.embedding_model.get_embedding(user_query)
            debug_info["steps"]["embedding"] = {
                "status": "success",
                "dimension": len(query_embedding)
            }
            
            retrieved_chunks = self.retrieve_relevant_chunks(
                query_embedding=query_embedding,
                k_neighbors=self.max_context_chunks
            )
            debug_info["steps"]["retrieval"] = {
                "status": "success",
                "total_chunks": len(retrieved_chunks),
                "chunk_scores": [{"score": chunk.get('score', 'unknown'), "similarity": 1 - chunk.get('score', 1)} for chunk in retrieved_chunks]
            }
            
            relevant_chunks = [
                chunk for chunk in retrieved_chunks 
                if (1 - chunk['score']) >= self.similarity_threshold
            ]
            debug_info["steps"]["filtering"] = {
                "status": "success",
                "similarity_threshold": self.similarity_threshold,
                "chunks_before_filter": len(retrieved_chunks),
                "chunks_after_filter": len(relevant_chunks),
                "filtered_out": len(retrieved_chunks) - len(relevant_chunks)
            }
            
            context = self._build_context(relevant_chunks)
            debug_info["steps"]["context_building"] = {
                "status": "success",
                "context_length": len(context),
                "context_preview": context[:200] + "..." if len(context) > 200 else context
            }
            
            ollama_available = self._test_ollama_connection()
            debug_info["steps"]["ollama_connection"] = {
                "status": "success",
                "available": ollama_available,
                "host": self.ollama_host,
                "port": self.ollama_port,
                "model": self.ollama_model
            }
            
        except Exception as e:
            debug_info["steps"]["error"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return debug_info
    
    def _generate_no_context_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no relevant context is found.
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Response dictionary for queries with no relevant context
        """
        return {
            "query": query,
            "answer": "I apologize, but I couldn't find relevant information in the available documents to answer your question. Please try rephrasing your query or ensure that relevant documents have been uploaded to the system.",
            "context_used": 0,
            "total_chunks_found": 0,
            "confidence_score": 0.0,
            "processing_status": "no_relevant_context"
        }
    
    def _calculate_confidence_score(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieved chunks.
        
        Args:
            chunks (List[Dict[str, Any]]): List of retrieved chunks
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        similarities = [1 - chunk.get('score', 1) for chunk in chunks]
        avg_similarity = np.mean(similarities)
        
        source_boost = min(len(chunks) / self.max_context_chunks, 1.0) * 0.1
        
        confidence = min(avg_similarity + source_boost, 1.0)
        return round(confidence, 3)
    
    def _extract_source_info(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Extract source information and unique document IDs from retrieved chunks.
        
        Args:
            chunks (List[Dict[str, Any]]): List of retrieved chunks
            
        Returns:
            Tuple[List[Dict[str, Any]], List[str]]: A tuple containing: 
                - List of source information dictionaries for display
                - List of unique source document IDs
        """
        sources_display = []
        unique_doc_ids = set()
        for chunk in chunks:
            doc_id = chunk.get('doc_id', 'unknown')
            unique_doc_ids.add(doc_id)
            source_info = {
                "document_id": doc_id,
                "chunk_id": chunk.get('chunk_id', 'unknown'),
                "similarity_score": round(1 - chunk.get('score', 1), 3),
                "text_preview": chunk.get('text_content', '')[:150] + "..." 
                    if len(chunk.get('text_content', '')) > 150 
                    else chunk.get('text_content', '')
            }
            sources_display.append(source_info)
        
        return sources_display, list(unique_doc_ids)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on RAG system components.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            "status": "healthy",
            "components": {},
            "issues": []
        }
        
        try:
            health_status["components"]["redis"] = {
                "status": "healthy",
                "document_count": self.redis_store.get_document_count()
            }
        except Exception as e:
            health_status["components"]["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["issues"].append("Redis connection failed")
            health_status["status"] = "degraded"
        
        try:
            test_embedding = self.embedding_model.get_embedding("test")
            health_status["components"]["embedding_model"] = {
                "status": "healthy",
                "dimension": len(test_embedding)
            }
        except Exception as e:
            health_status["components"]["embedding_model"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["issues"].append("Embedding model failed")
            health_status["status"] = "degraded"
        
        try:
            self._test_ollama_connection()
            health_status["components"]["ollama"] = {
                "status": "healthy",
                "host": self.ollama_host,
                "port": self.ollama_port,
                "model": self.ollama_model
            }
        except Exception as e:
            health_status["components"]["ollama"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["issues"].append("Ollama connection failed")
            health_status["status"] = "degraded"
        
        if health_status["issues"]:
            health_status["status"] = "unhealthy" if len(health_status["issues"]) > 1 else "degraded"
        
        return health_status