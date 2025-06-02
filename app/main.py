"""
Flask web application for RAG (Retrieval Augmented Generation) system.

This module provides a web interface and API endpoints for uploading PDF documents,
processing them into vector embeddings, and answering queries using semantic search
and language model generation.

Features:
- PDF upload and processing
- Vector-based document search
- Semantic caching for improved performance
- RESTful API endpoints
- Health monitoring and statistics
"""

import os
import logging
from flask import Flask, request, render_template, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from redis_client import RedisVectorStore
from embedding_interface import SentenceTransformerEmbeddingModel
from pdf_processor import PDFProcessor
from rag_core import RAGCore
from cache_manager import SemanticCacheManager

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

ALLOWED_EXTENSIONS = {'pdf'}

redis_store = None
embedding_model = None
pdf_processor = None
rag_core = None
cache_manager = None


def allowed_file(filename):
    """Check if uploaded file has allowed extension.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if file has allowed extension, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_embedding_model() -> SentenceTransformerEmbeddingModel:
    """Create and return the SentenceTransformer embedding model.
    
    Returns:
        SentenceTransformerEmbeddingModel: Configured SentenceTransformer embedding model instance
        
    Raises:
        ImportError: If sentence-transformers package is not installed
    """
    model_name = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'multi-qa-mpnet-base-dot-v1')
    logger.info(f"Initializing SentenceTransformer model: {model_name}")
    return SentenceTransformerEmbeddingModel(model_name=model_name)


def initialize_components():
    """Initialize all RAG system components.
    
    Raises:
        Exception: If any component fails to initialize
    """
    global redis_store, embedding_model, pdf_processor, rag_core, cache_manager
    
    try:
        logger.info("Initializing RAG system components...")

        embedding_model = create_embedding_model()
        embedding_dimension = embedding_model.get_dimension()
        logger.info(f"Embedding model initialized with dimension: {embedding_dimension}")
        
        redis_store = RedisVectorStore(embedding_dimension=embedding_dimension)
        logger.info("Redis vector store initialized")
        
        chunk_size = int(os.getenv('CHUNK_SIZE', 500))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info("PDF processor initialized")
        
        similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))
        knn_neighbors = int(os.getenv('KNN_NEIGHBORS', 5))
        
        ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        ollama_port = int(os.getenv('OLLAMA_PORT', 11434))
        ollama_model = os.getenv('OLLAMA_MODEL', 'gemma3:1b')
        
        rag_core = RAGCore(
            redis_store=redis_store,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            max_context_chunks=knn_neighbors,
            ollama_host=ollama_host,
            ollama_port=ollama_port,
            ollama_model=ollama_model
        )
        logger.info("RAG core initialized")
        
        cache_threshold = float(os.getenv('CACHE_SIMILARITY_THRESHOLD', 0.7))
        cache_ttl = int(os.getenv('CACHE_TTL', 3600))
        cache_manager = SemanticCacheManager(
            redis_store=redis_store,
            embedding_model=embedding_model,
            rag_core=rag_core,
            similarity_threshold=cache_threshold,
            cache_ttl=cache_ttl
        )
        logger.info("Cache manager initialized")
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page with upload and query forms.
    
    Returns:
        flask.Response: Rendered index template or redirected response for form submissions
    """
    if request.method == 'POST':
        if 'file' in request.files:
            return handle_file_upload()
        elif 'query' in request.form:
            return handle_query_submission()
    
    return render_template('index.html')


def handle_file_upload():
    """Handle PDF file upload and processing.
    
    Returns:
        flask.Response: Redirect response with flash messages indicating success or failure
    """
    try:
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if not (file and allowed_file(file.filename)):
            flash('Please upload a valid PDF file', 'error')
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        try:
            process_pdf_file(file_path, filename)
            flash(f'File {filename} uploaded and processed successfully!', 'success')
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            flash(f'Error processing file {filename}: {str(e)}', 'error')
        
        return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        flash(f'Upload error: {str(e)}', 'error')
        return redirect(url_for('index'))


def handle_query_submission():
    """Handle query submission and generate response.
    
    Returns:
        flask.Response: Rendered template with query results or redirect with error message
    """
    try:
        user_query = request.form.get('query', '').strip()
        
        if not user_query:
            flash('Please enter a query', 'error')
            return redirect(url_for('index'))
        
        response = cache_manager.get_cached_response(user_query)
        
        return render_template('index.html', 
                             query=user_query, 
                             response=response,
                             show_response=True)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        flash(f'Query processing error: {str(e)}', 'error')
        return redirect(url_for('index'))


def process_pdf_file(file_path, doc_id=None):
    """Process a PDF file and store chunks in Redis.
    
    Args:
        file_path (str): Path to the PDF file to process
        doc_id (str, optional): Document identifier. Defaults to filename without extension.
        
    Raises:
        ValueError: If PDF file is invalid or corrupted
        Exception: If processing fails for any reason
    """
    try:
        if doc_id is None:
            doc_id = os.path.splitext(os.path.basename(file_path))[0]
        
        logger.info(f"Processing PDF: {file_path}")
        
        if not pdf_processor.validate_pdf_file(file_path):
            raise ValueError(f"Invalid or corrupted PDF file: {file_path}")
        
        chunks_processed = 0
        
        for chunk_info in pdf_processor.process_pdf(file_path, doc_id):
            try:
                chunk_text = chunk_info['text']
                chunk_embedding = embedding_model.get_embedding(chunk_text)
                
                success = redis_store.store_document_chunk(
                    doc_id=chunk_info['doc_id'],
                    chunk_id=chunk_info['chunk_id'],
                    text_content=chunk_text,
                    vector_embedding=chunk_embedding
                )
                
                if success:
                    chunks_processed += 1
                else:
                    logger.warning(f"Failed to store chunk: {chunk_info['chunk_id']}")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_info.get('chunk_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully processed {chunks_processed} chunks from {file_path}")
        
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}")
        raise


@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for query processing.
    
    Returns:
        flask.Response: JSON response containing query results or error message
    """
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        response = cache_manager.get_cached_response(user_query)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in API query: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for file upload.
    
    Returns:
        flask.Response: JSON response indicating upload success or failure
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        
        process_pdf_file(file_path, filename)
        
        return jsonify({
            'message': f'File {filename} uploaded and processed successfully',
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Error in API upload: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint.
    
    Returns:
        flask.Response: JSON response containing system health status
    """
    try:
        health_status = rag_core.health_check()
        
        cache_stats = cache_manager.get_cache_statistics()
        health_status['cache_statistics'] = cache_stats
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/api/cache/clear', methods=['POST'])
def api_clear_cache():
    """Clear cache endpoint.
    
    Returns:
        flask.Response: JSON response indicating cache clear success or failure
    """
    try:
        success = cache_manager.clear_cache()
        
        if success:
            return jsonify({'message': 'Cache cleared successfully'})
        else:
            return jsonify({'error': 'Failed to clear cache'}), 500
            
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get system statistics.
    
    Returns:
        flask.Response: JSON response containing system statistics
    """
    try:
        stats = {
            'cache_statistics': cache_manager.get_cache_statistics(),
            'document_count': redis_store.get_document_count(),
            'system_health': rag_core.health_check()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error.
    
    Args:
        e: The error object
        
    Returns:
        flask.Response: Redirect to index with error message
    """
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors.
    
    Args:
        e: The error object
        
    Returns:
        flask.Response: Redirect to index with error message
    """
    logger.error(f"Internal server error: {e}")
    flash('Internal server error occurred', 'error')
    return redirect(url_for('index'))


def main():
    """Main application entry point."""
    try:
        initialize_components()
        
        host = os.getenv('FLASK_HOST', '0.0.0.0')
        port = int(os.getenv('FLASK_PORT', 5000))
        debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
        
        logger.info(f"Starting Flask app on {host}:{port}")
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


if __name__ == '__main__':
    main() 