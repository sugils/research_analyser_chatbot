# 📚 Research Chatbot API - Setup Guide

A Flask-based REST API that enables semantic search over PDF documents using vector embeddings and generates context-aware responses using Google's Gemini AI.

## 🎯 Overview

This application provides two main functionalities:

1. **PDF Upload & Processing**: Upload PDF files, extract text, chunk it, generate embeddings, and store them in PostgreSQL with pgvector
2. **Semantic Query**: Query the stored documents using natural language and receive AI-generated, context-aware responses

### Key Features

- 📄 PDF text extraction and processing
- 🧮 Semantic embeddings using `BAAI/bge-large-en` model (1024 dimensions)
- 🔍 Vector similarity search using PostgreSQL with pgvector extension
- 🤖 Context-aware responses powered by Google Gemini 2.5 Flash
- 💬 Session-based conversation memory
- 🛡️ Comprehensive error handling and logging

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Frontend  │────▶│  Flask API   │────▶│   PostgreSQL    │
│             │     │              │     │   (pgvector)    │
└─────────────┘     └──────────────┘     └─────────────────┘
                            │
                            │
                    ┌───────▼────────┐
                    │  Gemini AI API │
                    └────────────────┘
```

### Technology Stack

- **Backend**: Flask (Python)
- **Database**: PostgreSQL with pgvector extension
- **Embeddings**: SentenceTransformer (`BAAI/bge-large-en`)
- **AI Generation**: Google Gemini 2.5 Flash
- **PDF Processing**: PyPDF2
- **Vector Operations**: NumPy

---

## 📋 Prerequisites

Before setting up the application, ensure you have the following installed:

### Required Software

1. **Python 3.8+**
   ```bash
   python --version
   ```

2. **PostgreSQL 12+**
   ```bash
   psql --version
   ```

3. **pip** (Python package manager)
   ```bash
   pip --version
   ```

### Required Accounts

- **Google Cloud Account**: For Gemini AI API access
  - Get API key from: https://aistudio.google.com/app/apikey

---

## 🚀 Installation Steps

### Step 1: Clone or Download the Project

```bash
# Create project directory
mkdir research-chatbot
cd research-chatbot

# Save the app.py file in this directory
```

### Step 2: Set Up PostgreSQL Database

#### 2.1 Install PostgreSQL

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

**On macOS (using Homebrew):**
```bash
brew install postgresql@14
brew services start postgresql@14
```

**On Windows:**
- Download from: https://www.postgresql.org/download/windows/

#### 2.2 Install pgvector Extension

```bash
# Ubuntu/Debian
sudo apt install postgresql-14-pgvector

# macOS
brew install pgvector

# Or compile from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### 2.3 Create Database and Enable Extension

```bash
# Login to PostgreSQL
sudo -u postgres psql

# Inside PostgreSQL prompt:
CREATE DATABASE research_chatbot;
\c research_chatbot
CREATE EXTENSION vector;

# Create the required table
CREATE TABLE document_chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    paragraph TEXT NOT NULL,
    embedding_1024 vector(1024) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

# Create index for faster similarity search
CREATE INDEX ON document_chunks USING ivfflat (embedding_1024 vector_cosine_ops) WITH (lists = 100);

# Exit PostgreSQL
\q
```

### Step 3: Configure Database Connection

Update the database credentials in `app.py`:

```python
DB_CONFIG = {
    "dbname": "research_chatbot",
    "user": "postgres",           # Change to your PostgreSQL username
    "password": "YOUR_PASSWORD",  # Change to your PostgreSQL password
    "host": "localhost",
    "port": 5432
}
```

### Step 4: Install Python Dependencies

Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

Install required packages:

```bash
pip install flask flask-cors werkzeug psycopg2-binary sentence-transformers numpy PyPDF2 google-genai
```

**Package Details:**
- `flask`: Web framework
- `flask-cors`: Enable Cross-Origin Resource Sharing
- `werkzeug`: Utilities for secure filename handling
- `psycopg2-binary`: PostgreSQL adapter
- `sentence-transformers`: Generate text embeddings
- `numpy`: Numerical operations
- `PyPDF2`: PDF text extraction
- `google-genai`: Google Gemini AI SDK

### Step 5: Configure API Keys

Update the Google Gemini API key in `app.py`:

```python
API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY"

# Also update in the client initialization:
client = genai.Client(api_key='YOUR_GOOGLE_GEMINI_API_KEY')
```

**To get your API key:**
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Create a new API key
4. Copy and paste it in the code

### Step 6: Create Uploads Directory

```bash
mkdir uploads
```

---

## 🎮 Running the Application

### Start the Server

```bash
python app.py
```

You should see:

```
======================================================================
🚀 Starting Research Chatbot API Server
======================================================================
📁 Upload folder: ./uploads
🗄️  Database: research_chatbot on localhost:5432
======================================================================
✅ Sentence Transformer model loaded successfully
✅ Gemini AI client initialized successfully
 * Running on http://0.0.0.0:5001
```

The API is now accessible at: `http://localhost:5001`

---

## 📡 API Endpoints

### 1. Upload PDF

**Endpoint:** `POST /upload_pdf`

**Description:** Upload a PDF file, extract text, generate embeddings, and store in database.

**Request:**
```bash
curl -X POST http://localhost:5001/upload_pdf \
  -F "pdf=@/path/to/your/document.pdf"
```

**Response (Success):**
```json
{
  "message": "Stored 42 chunks."
}
```

**Response (Error):**
```json
{
  "error": "No PDF file provided"
}
```

**Error Codes:**
- `400`: Bad request (missing file, empty file, no text extracted)
- `500`: Server error (PDF reading error, database error, embedding error)

---

### 2. Query Documents

**Endpoint:** `POST /query_chunks`

**Description:** Query stored documents using natural language and receive AI-generated responses.

**Request:**
```bash
curl -X POST http://localhost:5001/query_chunks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings of the research?",
    "session_id": "user123"
  }'
```

**Request Body:**
```json
{
  "query": "Your question here",
  "session_id": "optional_session_id"
}
```

**Response (Success):**
```json
{
  "summary": "#### ✅ Response\n\nThe research presents three main findings...\n\n#### 💡 Notes\n\n..."
}
```

**Response (Error):**
```json
{
  "error": "Missing query"
}
```

**Error Codes:**
- `400`: Bad request (missing query, invalid JSON, empty query)
- `500`: Server error (embedding error, database error, AI generation error)

---

## 🔧 Configuration Options

### Text Chunking Parameters

In the `chunk_text()` function:

```python
def chunk_text(text, chunk_size=300, overlap=50):
    # chunk_size: Number of words per chunk (default: 300)
    # overlap: Overlapping words between chunks (default: 50)
```

**Adjustment Guide:**
- **Large documents**: Increase `chunk_size` to 500-700
- **Short documents**: Decrease `chunk_size` to 150-200
- **More context retention**: Increase `overlap` to 100-150

### Retrieval Limit

In the `/query_chunks` endpoint:

```python
LIMIT 5  # Number of relevant chunks to retrieve
```

**Adjustment Guide:**
- **More comprehensive answers**: Increase to 7-10
- **Faster responses**: Decrease to 3
- **Balance**: Keep at 5 (recommended)

### Conversation History

```python
history[-5:]  # Keep last 5 messages
```

**Adjustment Guide:**
- **Longer context**: Increase to 10
- **Less memory usage**: Decrease to 3

---

## 🧪 Testing the Application

### Test 1: Upload a Sample PDF

```bash
# Create a test PDF or use an existing one
curl -X POST http://localhost:5001/upload_pdf \
  -F "pdf=@sample_research_paper.pdf"
```

### Test 2: Query the Uploaded Document

```bash
curl -X POST http://localhost:5001/query_chunks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of this document?",
    "session_id": "test_session"
  }'
```

### Test 3: Follow-up Question (Testing Context Memory)

```bash
curl -X POST http://localhost:5001/query_chunks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can you elaborate on that?",
    "session_id": "test_session"
  }'
```

---

## 📊 Database Schema

### Table: `document_chunks`

| Column          | Type          | Description                           |
|-----------------|---------------|---------------------------------------|
| chunk_id        | VARCHAR(255)  | Unique identifier (UUID)              |
| paragraph       | TEXT          | Text content of the chunk             |
| embedding_1024  | vector(1024)  | 1024-dimensional embedding vector     |
| created_at      | TIMESTAMP     | Timestamp of insertion (auto)         |

**Indexes:**
- Primary key on `chunk_id`
- IVFFlat index on `embedding_1024` for fast similarity search

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Database Connection Error

**Error:**
```
❌ Database connection error: could not connect to server
```

**Solutions:**
- Check if PostgreSQL is running:
  ```bash
  sudo systemctl status postgresql
  # or
  brew services list
  ```
- Verify database credentials in `DB_CONFIG`
- Ensure database `research_chatbot` exists
- Check if port 5432 is accessible

#### 2. pgvector Extension Not Found

**Error:**
```
ERROR: extension "vector" does not exist
```

**Solutions:**
- Install pgvector extension (see Step 2.2)
- Verify installation:
  ```bash
  psql -d research_chatbot -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
  ```

#### 3. Model Download Issues

**Error:**
```
❌ Error loading Sentence Transformer model
```

**Solutions:**
- Ensure stable internet connection
- Manually download model:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("BAAI/bge-large-en")
  ```
- Check disk space (model is ~1.2GB)
- Try clearing cache: `rm -rf ~/.cache/torch/sentence_transformers/`

#### 4. PDF Text Extraction Fails

**Error:**
```
No text could be extracted from the PDF
```

**Solutions:**
- Ensure PDF is not image-based (scanned document)
- Try OCR tools for scanned PDFs (pytesseract, pdf2image)
- Verify PDF is not corrupted
- Check PDF permissions (not password-protected)

#### 5. Gemini API Error

**Error:**
```
Error generating summary: API key not valid
```

**Solutions:**
- Verify API key is correct
- Check API quota/limits at Google AI Studio
- Ensure billing is enabled (if required)
- Test API key with simple request

#### 6. Port Already in Use

**Error:**
```
Address already in use
```

**Solutions:**
- Change port in `app.run(port=5001)` to another port (e.g., 5002)
- Kill process using port 5001:
  ```bash
  # Linux/macOS
  lsof -ti:5001 | xargs kill -9
  # Windows
  netstat -ano | findstr :5001
  taskkill /PID <PID> /F
  ```

#### 7. CORS Issues

**Error:**
```
Access to fetch blocked by CORS policy
```

**Solutions:**
- Ensure `flask-cors` is installed
- Verify `CORS(app)` is called
- For specific origins:
  ```python
  CORS(app, origins=["http://localhost:3000", "http://example.com"])
  ```

---

## 🔒 Security Considerations

### Production Deployment Checklist

- [ ] **API Keys**: Use environment variables instead of hardcoding
  ```python
  import os
  API_KEY = os.getenv('GEMINI_API_KEY')
  ```

- [ ] **Database Credentials**: Store in environment variables or config file
  ```python
  DB_CONFIG = {
      "dbname": os.getenv('DB_NAME'),
      "user": os.getenv('DB_USER'),
      "password": os.getenv('DB_PASSWORD'),
      "host": os.getenv('DB_HOST'),
      "port": int(os.getenv('DB_PORT', 5432))
  }
  ```

- [ ] **File Upload Validation**: Add file size limits and type validation
  ```python
  ALLOWED_EXTENSIONS = {'pdf'}
  MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
  ```

- [ ] **Rate Limiting**: Implement rate limiting to prevent abuse
  ```python
  from flask_limiter import Limiter
  limiter = Limiter(app, key_func=lambda: request.remote_addr)
  
  @app.route("/query_chunks", methods=["POST"])
  @limiter.limit("10 per minute")
  def query_chunks():
      ...
  ```

- [ ] **HTTPS**: Use HTTPS in production
- [ ] **Input Sanitization**: Validate and sanitize all user inputs
- [ ] **SQL Injection Prevention**: Use parameterized queries (already implemented)
- [ ] **Session Management**: Implement proper session management with expiration
- [ ] **Logging**: Add comprehensive logging (without sensitive data)
- [ ] **Error Messages**: Don't expose internal details in production errors

---

## 📈 Performance Optimization

### Database Optimization

1. **Index Tuning:**
   ```sql
   -- Adjust lists parameter based on your data size
   -- For 1M+ vectors: lists = 1000
   -- For 100K vectors: lists = 100
   -- For 10K vectors: lists = 10
   CREATE INDEX ON document_chunks USING ivfflat (embedding_1024 vector_cosine_ops) WITH (lists = 100);
   ```

2. **Connection Pooling:**
   ```python
   from psycopg2 import pool
   
   connection_pool = pool.SimpleConnectionPool(
       minconn=1,
       maxconn=10,
       **DB_CONFIG
   )
   
   def get_conn():
       return connection_pool.getconn()
   ```

3. **Query Optimization:**
   ```sql
   -- Add WHERE clause if filtering by date
   SELECT chunk_id, paragraph
   FROM document_chunks
   WHERE created_at > NOW() - INTERVAL '30 days'
   ORDER BY embedding_1024 <=> %s::vector
   LIMIT 5;
   ```

### Application Optimization

1. **Caching Embeddings:**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_cached_embedding(text):
       return model.encode([text])[0]
   ```

2. **Async Processing:**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   executor = ThreadPoolExecutor(max_workers=4)
   
   def process_pdf_async(file_path):
       return executor.submit(process_pdf, file_path)
   ```

3. **Batch Processing:**
   ```python
   # Process multiple PDFs in batch
   embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
   ```

---

## 🧩 Code Structure Explained

### Main Components

#### 1. **Configuration & Initialization** (Lines 1-60)
- Flask app setup
- CORS configuration
- Directory creation
- Model loading
- Database configuration

#### 2. **Helper Functions** (Lines 61-120)
- `get_conn()`: Database connection management
- `chunk_text()`: Text preprocessing and chunking

#### 3. **AI Generation** (Lines 121-200)
- `generate_summary()`: Gemini AI response generation with detailed prompting

#### 4. **API Endpoints** (Lines 201-end)
- `/upload_pdf`: File upload and processing pipeline
- `/query_chunks`: Query processing and response generation

### Data Flow

#### Upload Flow:
```
PDF File → Extract Text → Chunk Text → Generate Embeddings → Store in DB
```

#### Query Flow:
```
User Query → Generate Embedding → Similarity Search → 
Retrieve Chunks → Generate AI Response → Update Session → Return Response
```

---

## 🔄 Session Management

The application maintains conversation context using in-memory session storage:

```python
session_contexts = {
    "session_id_1": [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is..."}
    ],
    "session_id_2": [...]
}
```

**Important Notes:**
- Sessions are stored in memory (lost on restart)
- For production, use Redis or database-backed sessions
- Each session keeps last 5 conversation turns
- Session IDs should be unique per user/conversation

---

## 📚 Additional Resources

### Documentation Links

- **Flask**: https://flask.palletsprojects.com/
- **pgvector**: https://github.com/pgvector/pgvector
- **SentenceTransformers**: https://www.sbert.net/
- **Google Gemini AI**: https://ai.google.dev/
- **PyPDF2**: https://pypdf2.readthedocs.io/

### Learning Resources

- **Vector Databases**: https://www.pinecone.io/learn/vector-database/
- **Semantic Search**: https://www.sbert.net/examples/applications/semantic-search/README.html
- **RAG (Retrieval Augmented Generation)**: https://www.promptingguide.ai/techniques/rag

---

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with proper error handling
4. Add comments explaining complex logic
5. Test thoroughly
6. Submit pull request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include error handling with try-except blocks
- Log important events and errors
- Write descriptive commit messages

---

## 📝 License

This project is provided as-is for educational purposes.

---

## 💡 Future Enhancements

### Planned Features

- [ ] User authentication and authorization
- [ ] Multiple file format support (DOCX, TXT, HTML)
- [ ] Batch PDF upload
- [ ] Document management (list, delete documents)
- [ ] Advanced filtering (date range, document type)
- [ ] Export conversation history
- [ ] Real-time streaming responses
- [ ] Multi-language support
- [ ] Custom embedding models
- [ ] Analytics dashboard

### Scalability Improvements

- [ ] Move to production WSGI server (Gunicorn/uWSGI)
- [ ] Implement Redis for session management
- [ ] Add message queue (Celery) for async processing
- [ ] Deploy on cloud (AWS, GCP, Azure)
- [ ] Implement horizontal scaling
- [ ] Add monitoring (Prometheus, Grafana)

---

## ❓ FAQ

### Q1: How much data can I store?

**A:** PostgreSQL with pgvector can handle millions of vectors. However, consider:
- Disk space for embeddings
- Query performance degrades with scale
- Consider sharding for 10M+ documents

### Q2: Can I use different embedding models?

**A:** Yes! Change the model in initialization:
```python
model = SentenceTransformer("your-model-name")
```
Ensure vector dimension matches database schema.

### Q3: How accurate is the semantic search?

**A:** Accuracy depends on:
- Quality of embeddings model
- Text chunking strategy
- Number of retrieved chunks
- Query formulation

### Q4: Can I deploy this in production?

**A:** Yes, but consider:
- Use production WSGI server
- Implement proper security
- Add monitoring and logging
- Use managed database
- Implement proper error handling
- Add rate limiting

### Q5: What happens if the API key expires?

**A:** The application will fail to generate responses. Implement:
- API key rotation mechanism
- Fallback response templates
- Proper error handling and user notification

---

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review error logs: `tail -f app.log`
3. Verify all prerequisites are installed
4. Check database connectivity
5. Validate API keys

---

## 🎉 Acknowledgments

This application uses the following open-source technologies:
- Flask web framework
- PostgreSQL database
- pgvector extension
- SentenceTransformers library
- Google Gemini AI

---



**Last Updated**: October 2025  
**Version**: 1.0.0