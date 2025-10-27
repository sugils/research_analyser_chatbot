# ============================================================================
# FLASK APPLICATION: Research Chatbot with PDF Upload & Semantic Search
# ============================================================================
# This application provides a REST API for:
# 1. Uploading PDF documents and storing their embeddings in PostgreSQL
# 2. Querying the stored documents using semantic search with context-aware responses
# ============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
from google import genai

load_dotenv()

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

# Step 1: Initialize Flask application
app = Flask(__name__)

# Step 2: Enable CORS (Cross-Origin Resource Sharing) for all routes
# This allows frontend applications from different domains to access this API
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Step 3: Configure upload folder for storing PDF files
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Step 4: Configure API keys and session management
API_KEY = os.getenv("API_KEY")
# Dictionary to store conversation history for each session
# Structure: {session_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
session_contexts = {}

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

# Step 5: Initialize the sentence transformer model for generating embeddings
# This model converts text into 1024-dimensional vectors for semantic search
try:
    model = SentenceTransformer("BAAI/bge-large-en")
    print("✅ Sentence Transformer model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Sentence Transformer model: {str(e)}")
    raise

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Step 6: Configure PostgreSQL database connection parameters
DB_CONFIG = {
    "dbname": os.getenv("dbname"),
    "user": os.getenv("user"),
    "password":os.getenv("password"),
    "host": os.getenv("host"),
    "port": 5432
}

def get_conn():
    """
    Establish and return a database connection with RealDictCursor.
    
    Returns:
        psycopg2.connection: Database connection object
        
    Raises:
        psycopg2.Error: If connection fails
    """
    try:
        # Step 1: Attempt to connect to PostgreSQL database
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except psycopg2.Error as e:
        # Step 2: Log and re-raise database connection errors
        print(f"❌ Database connection error: {str(e)}")
        raise

# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Split text into overlapping chunks for better context retention.
    
    Args:
        text (str): Input text to be chunked
        chunk_size (int): Number of words per chunk (default: 300)
        overlap (int): Number of overlapping words between chunks (default: 50)
        
    Returns:
        list: List of text chunks
    """
    try:
        # Step 1: Split text into individual words
        words = text.split()
        
        # Step 2: Create empty list to store chunks
        chunks = []
        
        # Step 3: Iterate through words with sliding window approach
        for i in range(0, len(words), chunk_size - overlap):
            # Step 3a: Extract chunk of specified size
            chunk = " ".join(words[i:i + chunk_size])
            
            # Step 3b: Add non-empty chunks to list
            if chunk:
                chunks.append(chunk)
        
        # Step 4: Return list of chunks
        return chunks
    except Exception as e:
        print(f"❌ Error chunking text: {str(e)}")
        return []

# ============================================================================
# GEMINI AI CLIENT INITIALIZATION
# ============================================================================

# Step 7: Initialize Gemini AI client for generating responses
try:
    client = genai.Client(API_KEY)
    print("✅ Gemini AI client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Gemini AI client: {str(e)}")
    raise

# ============================================================================
# RESPONSE GENERATION FUNCTION
# ============================================================================

def generate_summary(user_query: str, retrieved_chunks: list, conversation_context: str = ""):
    """
    Generate a contextually grounded, non-hallucinated summary using Gemini AI.
    
    Args:
        user_query (str): The user's question or query
        retrieved_chunks (list): List of dictionaries with 'chunk_id' and 'paragraph'
        conversation_context (str): Previous conversation history for context
        
    Returns:
        str: AI-generated response based on retrieved context
        
    Raises:
        Exception: If AI generation fails
    """
    try:
        # Step 1: Combine all retrieved chunks into a single context string
        context_text = "\n".join([f"Chunk {c['chunk_id']}: {c['paragraph']}" for c in retrieved_chunks])

        # Step 2: Prepare conversation context section (or placeholder if empty)
        conversation_section = conversation_context if conversation_context else "(no prior conversation)"

        # Step 3: Construct the detailed prompt with instructions for AI
        base_prompt = (
        "You are a **Friendly, Context-Aware Academic Assistant**.\n\n"
        "### How to Respond\n"
        "You will respond **factually** based strictly on the provided context (KB), but in a **human-like, conversational, and friendly tone**. "
        "Think of yourself as explaining to a smart colleague: precise on facts, approachable in tone, giving gentle guidance if something is off.\n\n"

        "### Speaking Style\n"
        "- If the user is correct about something, start with: '**Absolutely right!** …'\n"
        "- If the user is partially correct or slightly off, start with: '**Almost there, but…**' or '**Not quite, here's the missing piece…**'\n"
        "- Always **acknowledge correct parts** before explaining mistakes.\n"
        "- Provide reasoning and step-by-step explanation in a friendly manner.\n"
        "- Keep sentences concise and clear.\n\n"

        "### Core Rules\n"
        "1. **Use ONLY the provided context** for factual answers.\n"
        "2. Preserve **exact phrasing** or technical terms from the context for facts.\n"
        "3. If the context does not contain the answer, explicitly say: 'The provided context does not contain information to answer this query.'\n"
        "4. Humanize explanations for understanding and guidance, but **do not fabricate facts**.\n\n"

        "### Formatting Instructions\n"
        "1. Use **Markdown formatting**.\n"
        "2. Use **headings** (###, ####) for sections.\n"
        "3. Use **bold** for key terms, important points, or confirmations.\n"
        "4. Use **bullet points** or numbered lists for step-by-step instructions or multiple items.\n"
        "5. Italics for clarifications or subtle notes.\n"
        "6. Keep output structured for readability:\n\n"
        "#### \n"
        "(Friendly, human-like explanation with exact facts.)\n\n"
        "#### 💡 Notes\n"
        "(If something is unclear, partially correct, or missing, explain in a humanized, approachable way.)\n\n"

        "### Input\n"
        f"**User Query:** {user_query}\n\n"
        f"**Conversation Context:**\n{conversation_section}\n\n"
        f"**Retrieved Context (Chunks):**\n{context_text}\n\n"

        "### Expected Output\n"
        "- Provide a **Markdown response** that is: \n"
        "  - **Factually accurate** (use context only)\n"
        "  - **Friendly and humanized**\n"
        "  - **Readable and structured** (headings, bullets, bold for emphasis)\n"
        "  - Correctly acknowledges right statements and gently explains mistakes or missing points\n"
        "  - Step-by-step explanation where helpful\n"
    )

        # Step 4: Call Gemini 2.5 Flash model to generate response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=base_prompt
        )

        # Step 5: Return the generated text response
        return response.text
        
    except Exception as e:
        # Step 6: Handle any errors during response generation
        print(f"❌ Error generating summary: {str(e)}")
        return f"Error generating response: {str(e)}"

# ============================================================================
# API ENDPOINTS
# ============================================================================

# ----------------------------------------------------------------------------
# API 1: Upload PDF & Store Embeddings
# ----------------------------------------------------------------------------

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """
    API endpoint to upload PDF files and store their embeddings in the database.
    
    Expected Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: PDF file with key 'pdf'
        
    Returns:
        JSON response with success message or error
    """
    try:
        # Step 1: Validate that a PDF file is included in the request
        if 'pdf' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400

        # Step 2: Retrieve the uploaded file from request
        file = request.files['pdf']
        
        # Step 3: Validate that a file was actually selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Step 4: Secure the filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        
        # Step 5: Construct full path for saving the file
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Step 6: Save the uploaded file to disk
        file.save(path)
        print(f"✅ PDF saved to: {path}")

        # Step 7: Extract text from PDF file
        try:
            pdf_reader = PyPDF2.PdfReader(path)
            full_text = ""
            
            # Step 7a: Iterate through all pages and extract text
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    full_text += page.extract_text() + " "
                except Exception as page_error:
                    print(f"⚠️ Warning: Could not extract text from page {page_num}: {str(page_error)}")
                    continue
            
            # Step 7b: Validate that text was extracted
            if not full_text.strip():
                return jsonify({"error": "No text could be extracted from the PDF"}), 400
                
        except Exception as pdf_error:
            return jsonify({"error": f"Error reading PDF: {str(pdf_error)}"}), 500

        # Step 8: Split extracted text into manageable chunks
        chunks = chunk_text(full_text)
        
        # Step 8a: Validate that chunks were created
        if not chunks:
            return jsonify({"error": "No text chunks could be created from the PDF"}), 400
        
        print(f"✅ Created {len(chunks)} text chunks")

        # Step 9: Generate embeddings for all chunks
        try:
            embeddings = model.encode(chunks)
            print(f"✅ Generated {len(embeddings)} embeddings")
        except Exception as embed_error:
            return jsonify({"error": f"Error generating embeddings: {str(embed_error)}"}), 500

        # Step 10: Store chunks and embeddings in database
        try:
            # Step 10a: Establish database connection
            conn = get_conn()
            cur = conn.cursor()
            
            # Step 10b: Insert each chunk with its embedding
            inserted_count = 0
            for i, chunk in enumerate(chunks):
                try:
                    cur.execute("""
                        INSERT INTO document_chunks (chunk_id, paragraph, embedding_1024)
                        VALUES (%s, %s, %s)
                    """, (str(uuid.uuid4()), chunk, embeddings[i].tolist()))
                    inserted_count += 1
                except Exception as insert_error:
                    print(f"⚠️ Warning: Could not insert chunk {i}: {str(insert_error)}")
                    continue
            
            # Step 10c: Commit all insertions to database
            conn.commit()
            print(f"✅ Inserted {inserted_count} chunks into database")
            
            # Step 10d: Close database resources
            cur.close()
            conn.close()
            
            # Step 10e: Return success response
            return jsonify({"message": f"Stored {inserted_count} chunks."})
            
        except psycopg2.Error as db_error:
            return jsonify({"error": f"Database error: {str(db_error)}"}), 500
        except Exception as e:
            return jsonify({"error": f"Error storing embeddings: {str(e)}"}), 500
            
    except Exception as e:
        # Step 11: Catch any unexpected errors
        print(f"❌ Unexpected error in upload_pdf: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ----------------------------------------------------------------------------
# API 2: Query Relevant Chunks with Context-Aware Response
# ----------------------------------------------------------------------------

@app.route("/query_chunks", methods=["POST"])
def query_chunks():
    """
    API endpoint to query stored document chunks using semantic search
    and generate context-aware responses.
    
    Expected Request:
        - Method: POST
        - Content-Type: application/json
        - Body: {"query": "user question", "session_id": "optional_session_id"}
        
    Returns:
        JSON response with AI-generated summary
    """
    try:
        # Step 1: Parse JSON data from request
        data = request.json
        
        # Step 1a: Validate that request body is valid JSON
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400
        
        # Step 1b: Validate that query parameter is present
        if "query" not in data:
            return jsonify({"error": "Missing query"}), 400

        # Step 2: Extract query text and session ID from request
        query_text = data["query"]
        session_id = data.get("session_id", "default_session")  # simple session tracking
        
        # Step 2a: Validate that query is not empty
        if not query_text or not query_text.strip():
            return jsonify({"error": "Query cannot be empty"}), 400
        
        print(f"📝 Received query: {query_text} (Session: {session_id})")

        # Step 3: Retrieve prior conversation context from session storage
        history = session_contexts.get(session_id, [])
        
        # Step 3a: Build conversation context string from last 5 messages
        conversation_context = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in history[-5:]]  # keep last 5 messages
        )

        # Step 4: Build contextual query for better retrieval
        # Combines conversation history with current query for context-aware search
        contextual_query = f"{conversation_context}\nUser: {query_text}" if conversation_context else query_text

        # Step 5: Generate embedding for the contextual query
        try:
            query_embedding = model.encode([contextual_query])[0].tolist()
            print(f"✅ Generated query embedding")
        except Exception as embed_error:
            return jsonify({"error": f"Error generating query embedding: {str(embed_error)}"}), 500

        # Step 6: Run pgvector similarity search in database
        try:
            # Step 6a: Establish database connection
            conn = get_conn()
            cur = conn.cursor()
            
            # Step 6b: Execute similarity search query using vector distance
            # Orders results by cosine similarity (<=> operator) and returns top 5
            cur.execute("""
                SELECT chunk_id, paragraph
                FROM document_chunks
                ORDER BY embedding_1024 <=> %s::vector
                LIMIT 5
            """, (query_embedding,))
            
            # Step 6c: Fetch all matching results
            results = cur.fetchall()
            print(f"✅ Retrieved {len(results)} relevant chunks")
            
            # Step 6d: Close database resources
            cur.close()
            conn.close()
            
        except psycopg2.Error as db_error:
            return jsonify({"error": f"Database query error: {str(db_error)}"}), 500
        except Exception as e:
            return jsonify({"error": f"Error retrieving chunks: {str(e)}"}), 500

        # Step 7: Handle different cursor return formats (dict or tuple)
        chunks = []
        try:
            for r in results:
                if isinstance(r, dict):
                    # RealDictCursor returns dictionaries
                    chunks.append({"chunk_id": r["chunk_id"], "paragraph": r["paragraph"]})
                else:
                    # Regular cursor returns tuples
                    chunks.append({"chunk_id": r[0], "paragraph": r[1]})
        except Exception as parse_error:
            return jsonify({"error": f"Error parsing database results: {str(parse_error)}"}), 500

        # Step 8: Generate the context-aware response using Gemini AI
        try:
            summary = generate_summary(
                user_query=query_text,
                retrieved_chunks=chunks,
                conversation_context=conversation_context
            )
            print(f"✅ Generated AI response")
        except Exception as gen_error:
            return jsonify({"error": f"Error generating response: {str(gen_error)}"}), 500

        # Step 9: Update conversation memory for this session
        try:
            # Step 9a: Append user message to history
            history.append({"role": "user", "content": query_text})
            
            # Step 9b: Append assistant response to history
            history.append({"role": "assistant", "content": summary})
            
            # Step 9c: Save updated history back to session storage
            session_contexts[session_id] = history
            print(f"✅ Updated conversation history (Session: {session_id})")
        except Exception as mem_error:
            print(f"⚠️ Warning: Could not update conversation memory: {str(mem_error)}")
            # Don't fail the request if memory update fails

        # Step 10: Return successful response with summary
        return jsonify({"summary": summary})
        
    except Exception as e:
        # Step 11: Catch any unexpected errors
        print(f"❌ Unexpected error in query_chunks: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Step 1: Print startup information
    print("=" * 70)
    print("🚀 Starting Research Chatbot API Server")
    print("=" * 70)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🗄️  Database: {DB_CONFIG['dbname']} on {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print("=" * 70)
    
    # Step 2: Start Flask development server
    # host="0.0.0.0" makes server accessible from other machines
    # port=5001 specifies the port number
    app.run(host="0.0.0.0", port=5001)
