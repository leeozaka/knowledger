"""
Redis vector store implementation for document storage and similarity search.

This module provides a Redis-based vector store that supports:
- Document chunk storage with embeddings
- KNN similarity search using RediSearch
- Semantic caching for query responses
- Index management for documents and cache
"""

import redis
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
from dotenv import load_dotenv
from redis.commands.search.query import Query

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisVectorStore:
    """Redis client for vector storage and similarity search operations."""
    
    def __init__(self, embedding_dimension: int = 768):
        """Initialize Redis connection and schema.
        
        Args:
            embedding_dimension: Dimension of embedding vectors
        """
        self.host = os.getenv('REDIS_HOST', 'localhost')
        self.port = int(os.getenv('REDIS_PORT', 6379))
        self.db = int(os.getenv('REDIS_DB', 0))
        self.embedding_dimension = embedding_dimension
        
        self.documents_index = "idx:documents"
        self.cache_index = "idx:cache"
        
        self.client = None
        self.connect()
        self.create_indices()
    
    def connect(self) -> None:
        """Establish connection to Redis server."""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            self.text_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def create_indices(self) -> None:
        """Create RediSearch indices for documents and cache if they don't exist."""
        try:
            try:
                self.text_client.ft(self.documents_index).info()
                logger.info(f"Documents index '{self.documents_index}' already exists")
            except:
                self._create_documents_index()
            
            try:
                self.text_client.ft(self.cache_index).info()
                logger.info(f"Cache index '{self.cache_index}' already exists")
            except:
                self._create_cache_index()
                
        except Exception as e:
            logger.error(f"Error creating indices: {e}")
            raise
    
    def _create_documents_index(self) -> None:
        """Create the documents index for storing PDF chunks."""
        from redis.commands.search.field import TextField, VectorField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        
        schema = [
            TextField("doc_id"),
            TextField("chunk_id"),
            TextField("text_content"),
            VectorField("vector_embedding", 
                       "FLAT", 
                       {
                           "TYPE": "FLOAT32",
                           "DIM": self.embedding_dimension,
                           "DISTANCE_METRIC": "COSINE"
                       })
        ]
        
        definition = IndexDefinition(
            prefix=["doc:"], 
            index_type=IndexType.HASH
        )
        
        self.text_client.ft(self.documents_index).create_index(
            schema, 
            definition=definition
        )
        logger.info(f"Created documents index '{self.documents_index}'")
    
    def _create_cache_index(self) -> None:
        """Create the cache index for storing query-response pairs."""
        from redis.commands.search.field import TextField, VectorField, TagField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        
        schema = [
            TextField("query_text"),
            TextField("response"),
            VectorField("query_embedding", 
                       "FLAT", 
                       {
                           "TYPE": "FLOAT32",
                           "DIM": self.embedding_dimension,
                           "DISTANCE_METRIC": "COSINE"
                       }),
            TagField("source_doc_ids", separator=',')
        ]
        
        definition = IndexDefinition(
            prefix=["cache:"], 
            index_type=IndexType.HASH
        )
        
        self.text_client.ft(self.cache_index).create_index(
            schema, 
            definition=definition
        )
        logger.info(f"Created cache index '{self.cache_index}'")
    
    def store_document_chunk(self, 
                           doc_id: str, 
                           chunk_id: str, 
                           text_content: str, 
                           vector_embedding: List[float]) -> bool:
        """Store a document chunk with its embedding in Redis.
        
        Args:
            doc_id: Identifier for the original document
            chunk_id: Identifier for the specific chunk
            text_content: The actual text of the chunk
            vector_embedding: The numerical embedding of the text chunk
            
        Returns:
            True if successful, False otherwise
        """
        try:
            embedding_bytes = np.array(vector_embedding, dtype=np.float32).tobytes()
            
            doc_key = f"doc:{doc_id}:{chunk_id}"
            
            mapping = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text_content": text_content,
                "vector_embedding": embedding_bytes
            }
            
            self.client.hset(doc_key, mapping=mapping)
            logger.info(f"Stored document chunk: {doc_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document chunk: {e}")
            return False
    
    def knn_search(self, 
                   query_vector: List[float], 
                   k_neighbors: int = 5, 
                   index_name: str = None) -> List[Dict[str, Any]]:
        """Perform KNN search to find similar document chunks.
        
        Args:
            query_vector: Query embedding vector
            k_neighbors: Number of neighbors to return
            index_name: Index to search (defaults to documents index)
            
        Returns:
            List of dictionaries containing search results with scores
        """
        if index_name is None:
            index_name = self.documents_index
        
        vector_field_name = "vector_embedding"
        if index_name == self.cache_index:
            vector_field_name = "query_embedding"
            
        try:
            query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
            
            if index_name == self.cache_index:
                return_fields = ["query_text", "response", "source_doc_ids", "vector_score"]
            else:
                return_fields = ["doc_id", "chunk_id", "text_content", "vector_score"]
            
            query = (
                Query(f"(*)=>[KNN {k_neighbors} @{vector_field_name} $query_vec AS vector_score]")
                .sort_by("vector_score")
                .return_fields(*return_fields)
                .paging(0, k_neighbors)
                .dialect(2)
            )
            
            results = self.text_client.ft(index_name).search(
                query,
                query_params={"query_vec": query_bytes}
            )
            
            parsed_results = []
            logger.info(f"KNN search found {len(results.docs)} documents")
            
            for doc_idx, doc in enumerate(results.docs):
                result = {"score": 0.0}
                
                doc_key = doc.id
                
                if hasattr(doc, 'vector_score'):
                    result["score"] = float(doc.vector_score)
                
                try:
                    if index_name == self.cache_index:
                        field_names = ["query_text", "response", "source_doc_ids"]
                    else:
                        field_names = ["doc_id", "chunk_id", "text_content"]
                    
                    field_values = self.client.hmget(doc_key, field_names)
                    
                    if any(field_values):
                        for field_name, field_value in zip(field_names, field_values):
                            if field_value is not None:
                                if isinstance(field_value, bytes):
                                    try:
                                        result[field_name] = field_value.decode('utf-8')
                                    except UnicodeDecodeError:
                                        logger.warning(f"Could not decode field {field_name} for {doc_key}")
                                        result[field_name] = ''
                                else:
                                    result[field_name] = str(field_value)
                            else:
                                result[field_name] = ''
                    else:
                        logger.warning(f"No hash data found for document key: {doc_key}")
                        
                except Exception as e:
                    logger.error(f"Error fetching hash data for {doc_key}: {e}")
                    continue
                
                if index_name == self.documents_index:
                    if 'doc_id' not in result:
                        key_parts = doc_key.split(':')
                        if len(key_parts) >= 3:
                            result['doc_id'] = key_parts[1]
                            result['chunk_id'] = key_parts[2]
                        else:
                            result['doc_id'] = 'unknown'
                            result['chunk_id'] = doc_key
                    
                    if 'text_content' not in result:
                        result['text_content'] = ''

                parsed_results.append(result)
                logger.info(f"Processed doc {doc_idx}: doc_id={result.get('doc_id', 'unknown')}, "
                          f"chunk_id={result.get('chunk_id', 'unknown')}, "
                          f"text_length={len(result.get('text_content', ''))}, "
                          f"score={result.get('score', 0.0)}")
            
            logger.info(f"KNN search returned {len(parsed_results)} results from index '{index_name}'")
            return parsed_results
            
        except Exception as e:
            logger.error(f"Error performing KNN search on index '{index_name}': {e}")
            return []
    
    def store_cache_entry(self, 
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
            embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            import hashlib
            query_hash = hashlib.md5(query_text.encode('utf-8')).hexdigest()
            cache_key = f"cache:{query_hash}"
            
            mapping = {
                "query_text": query_text,
                "response": response,
                "query_embedding": embedding_bytes,
                "source_doc_ids": ",".join(source_document_ids) if source_document_ids else ""
            }
            
            self.client.hset(cache_key, mapping=mapping)
            
            ttl = int(os.getenv('CACHE_TTL', 3600))
            self.client.expire(cache_key, ttl)
            
            logger.info(f"Stored cache entry: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing cache entry: {e}")
            return False
    
    def search_cache(self, 
                     query_embedding: List[float], 
                     similarity_threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """Search cache for similar queries.
        
        Args:
            query_embedding: Query embedding vector
            similarity_threshold: Minimum similarity threshold for cache hits
            
        Returns:
            Tuple of (cached_response, similarity_score) if found, None otherwise
        """
        try:
            results = self.knn_search(
                query_vector=query_embedding,
                k_neighbors=1,
                index_name=self.cache_index
            )
            
            if results and len(results) > 0:
                result = results[0]
                similarity_score = 1 - result.get('score', 2.0)
                
                if similarity_score >= similarity_threshold and 'response' in result:
                    cached_response = result['response']
                    logger.info(f"Cache hit with similarity: {similarity_score:.4f}")
                    return cached_response, similarity_score
            
            logger.info("No cache hit found or similarity below threshold")
            return None
            
        except Exception as e:
            logger.error(f"Error searching cache: {e}")
            return None
    
    def get_document_count(self) -> int:
        """Get the total number of stored document chunks.
        
        Returns:
            Number of document chunks in the index
        """
        try:
            info = self.text_client.ft(self.documents_index).info()
            return info.get('num_docs', 0)
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def clear_all_data(self) -> bool:
        """Clear all stored data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for key in self.client.scan_iter(match="doc:*"):
                self.client.delete(key)
            
            for key in self.client.scan_iter(match="cache:*"):
                self.client.delete(key)
            
            logger.info("Cleared all data from Redis")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            return False

    def invalidate_cache_by_doc_ids(self, doc_ids: List[str]) -> int:
        """Invalidate cache entries associated with specific document IDs.

        Args:
            doc_ids: List of document IDs to invalidate cache entries for.

        Returns:
            Number of cache entries deleted.
        """
        if not doc_ids:
            return 0

        deleted_count = 0
        escaped_doc_ids = [doc_id.replace('-', '\\-') for doc_id in doc_ids]
        query_tags = "|".join(escaped_doc_ids)
        
        cache_query = Query(f"@source_doc_ids:{{{query_tags}}}") \
            .return_fields("id")

        try:
            results = self.text_client.ft(self.cache_index).search(cache_query)
            keys_to_delete = [doc.id for doc in results.docs]
            
            if keys_to_delete:
                deleted_count = self.client.delete(*keys_to_delete)
                logger.info(f"Invalidated {deleted_count} cache entries associated with doc_ids: {doc_ids}")
            else:
                logger.info(f"No cache entries found for invalidation for doc_ids: {doc_ids}")
                
            return deleted_count if deleted_count is not None else 0

        except Exception as e:
            logger.error(f"Error invalidating cache entries by doc_ids: {e}")
            return 0 