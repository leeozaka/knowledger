"""
PDF document processing and text extraction utilities.

This module handles PDF document processing for the RAG system including:
- Text extraction from PDF files
- Text cleaning and normalization
- Document chunking with configurable overlap
- PDF validation and metadata extraction
"""

import PyPDF2
import os
import logging
from typing import List, Dict, Generator
from io import BytesIO
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and chunking for RAG pipeline."""
    
    def __init__(self, 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50):
        """Initialize PDF processor with chunking parameters.
        
        Args:
            chunk_size: Target size of text chunks in words
            chunk_overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is corrupt or unreadable
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            extracted_text = ""
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF {file_path} is encrypted, attempting to decrypt")
                    try:
                        pdf_reader.decrypt("")
                    except:
                        raise ValueError(f"Cannot decrypt PDF: {file_path}")
                
                num_pages = len(pdf_reader.pages)
                logger.info(f"Processing {num_pages} pages from {file_path}")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        logger.debug(f"Page {page_num + 1} raw text length: {len(page_text) if page_text else 0}")
                        if page_text and page_text.strip():
                            extracted_text += f"\n--- Page {page_num + 1} ---\n"
                            extracted_text += page_text
                        else:
                            logger.debug(f"Page {page_num + 1} has no text or only whitespace.")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
            
            if not extracted_text.strip():
                raise ValueError(f"No text could be extracted from PDF: {file_path}")
            
            logger.info(f"Successfully extracted {len(extracted_text)} characters from {file_path}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise ValueError(f"Failed to process PDF: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned and normalized text
        """
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'\n--- Page \d+ ---\n', ' ', text)
        
        text = ' '.join(text.split())
        
        return text.strip()
    
    def chunk_text(self, text: str) -> Generator[Dict[str, str], None, None]:
        """Split text into overlapping chunks for embedding.
        
        Args:
            text (str): Text to be chunked
            
        Yields:
            Dict[str, str]: Dictionary containing chunk information including chunk_id, text, word_count, start_word, and end_word
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return
        
        words = text.split()
        
        if len(words) <= self.chunk_size:
            yield {
                "chunk_id": "chunk_0",
                "text": text,
                "word_count": len(words),
                "start_word": 0,
                "end_word": len(words) - 1
            }
            return
        
        chunk_num = 0
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunk_info = {
                "chunk_id": f"chunk_{chunk_num}",
                "text": chunk_text,
                "word_count": len(chunk_words),
                "start_word": start_idx,
                "end_word": end_idx - 1
            }
            
            yield chunk_info
            
            if end_idx >= len(words):
                break
            
            start_idx = end_idx - self.chunk_overlap
            chunk_num += 1
        
        logger.info(f"Created {chunk_num + 1} chunks from text")
    
    def process_pdf(self, 
                   file_path: str, 
                   doc_id: str = None) -> Generator[Dict[str, any], None, None]:
        """Complete pipeline to process a PDF file into chunks.
        
        Args:
            file_path (str): Path to the PDF file
            doc_id (str, optional): Document identifier (defaults to filename)
            
        Yields:
            Dict[str, any]: Dictionary containing processed chunk information with doc_id, file_path, and processing_timestamp
            
        Raises:
            Exception: If PDF processing fails
        """
        if doc_id is None:
            doc_id = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            raw_text = self.extract_text_from_pdf(file_path)
            
            cleaned_text = self.clean_text(raw_text)
            logger.debug(f"Cleaned text for {doc_id} (first 200 chars): {cleaned_text[:200]}")
            
            if not cleaned_text:
                logger.error(f"No usable text extracted from {file_path}")
                return
            
            for chunk_info in self.chunk_text(cleaned_text):
                chunk_info.update({
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "processing_timestamp": self._get_timestamp()
                })
                
                yield chunk_info
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata.
        
        Returns:
            ISO format timestamp string
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_pdf_file(self, file_path: str) -> bool:
        """Validate that a file is a readable PDF.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is a valid PDF, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            if not file_path.lower().endswith('.pdf'):
                return False
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                num_pages = len(pdf_reader.pages)
                
                if num_pages == 0:
                    return False
                
                first_page = pdf_reader.pages[0]
                first_page.extract_text()
                
            return True
            
        except Exception as e:
            logger.warning(f"PDF validation failed for {file_path}: {e}")
            return False
    
    def get_pdf_metadata(self, file_path: str) -> Dict[str, any]:
        """Extract metadata from PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Dict[str, any]: Dictionary containing PDF metadata including num_pages, file_size, is_encrypted, and file_name
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                metadata = {
                    "num_pages": len(pdf_reader.pages),
                    "file_size": os.path.getsize(file_path),
                    "is_encrypted": pdf_reader.is_encrypted,
                    "file_name": os.path.basename(file_path)
                }
                
                if pdf_reader.metadata:
                    for key, value in pdf_reader.metadata.items():
                        if key.startswith('/'):
                            key = key[1:]
                        metadata[key.lower()] = str(value) if value else None
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error getting PDF metadata for {file_path}: {e}")
            return {
                "num_pages": 0,
                "file_size": 0,
                "is_encrypted": False,
                "file_name": os.path.basename(file_path) if os.path.exists(file_path) else "unknown",
                "error": str(e)
            } 