import os
import json
import logging
from typing import Any, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment configuration (.env files)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))
load_dotenv()

logger = logging.getLogger("offline_study_ai.cors")

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
    from backend.chunking import (
        chunk_text,
        chunk_pdf_pages,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
    )
    from backend.pdf_parser import extract_and_chunk_pdf
    from backend.vector_store import (
        index_document,
        search_documents,
        get_chroma_path,
        get_indexed_documents,
        delete_document,
    )
    from backend.qa_service import answer_study_question
    from backend.summarizer import summarize_document
    from backend.question_generator import (
        generate_important_questions,
        generate_question_answer,
    )
    from backend.ollama_client import (
        OllamaConnectionError,
        OllamaModelNotFoundError,
        OllamaError,
    )
    from backend.auth_service import (
        get_current_user,
        register_user,
        authenticate_user,
        create_access_token,
        authenticate_google_user,
        request_password_reset,
        reset_password,
    )
except ImportError:
    from chunking import (
        chunk_text,
        chunk_pdf_pages,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
    )
    from pdf_parser import extract_and_chunk_pdf
    from vector_store import (
        index_document,
        search_documents,
        get_chroma_path,
        get_indexed_documents,
        delete_document,
    )
    from qa_service import answer_study_question
    from summarizer import summarize_document
    from question_generator import (
        generate_important_questions,
        generate_question_answer,
    )
    from ollama_client import (
        OllamaConnectionError,
        OllamaModelNotFoundError,
        OllamaError,
    )
    from auth_service import (
        get_current_user,
        register_user,
        authenticate_user,
        create_access_token,
        authenticate_google_user,
        request_password_reset,
        reset_password,
    )


# ---------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30, description="Unique username (3-30 chars, alphanumeric/underscore/hyphen)")
    email: str = Field(..., description="Unique user email address")
    password: str = Field(..., min_length=6, description="User password (at least 6 characters)")


class LoginRequest(BaseModel):
    username_or_email: str = Field(..., min_length=1, description="Username or email address")
    password: str = Field(..., min_length=1, description="Password")


class AuthUser(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="User's unique username")
    email: str = Field(..., description="User's unique email address")


class AuthResponse(BaseModel):
    access_token: str = Field(..., description="Cryptographically signed JWT Bearer token")
    token_type: str = Field(default="bearer", description="Token type")
    user: AuthUser = Field(..., description="Authenticated user profile")


class UserProfileResponse(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    created_at: float = Field(..., description="Account creation timestamp")


class GoogleLoginRequest(BaseModel):
    credential: Optional[str] = Field(default=None, description="Google ID Token JWT")
    email: Optional[str] = Field(default=None, description="Google user email")
    name: Optional[str] = Field(default=None, description="Google user display name")
    google_id: Optional[str] = Field(default=None, description="Google sub identifier")


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., description="Registered account email address")


class ForgotPasswordResponse(BaseModel):
    message: str
    expires_in_minutes: int = 15


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1, description="Reset token or 6-digit verification code")
    new_password: str = Field(..., min_length=6, description="New user password (at least 6 characters)")
    email: Optional[str] = Field(default=None, description="Optional email address to verify against")


class ResetPasswordResponse(BaseModel):
    message: str
    success: bool = True


class ChunkResponse(BaseModel):
    chunk_number: int = Field(..., description="1-based sequential chunk number")
    page_number: int = Field(..., description="1-based PDF page number")
    text: str = Field(..., description="Extracted text chunk content")


class PDFUploadResponse(BaseModel):
    filename: str = Field(..., description="Filename of the uploaded PDF")
    total_pages: int = Field(..., description="Total number of pages in the PDF")
    total_chunks: int = Field(..., description="Total number of extracted chunks")
    chunks: list[ChunkResponse] = Field(..., description="List of extracted text chunks")


class IndexPDFResponse(BaseModel):
    message: str = Field(..., description="Status message")
    document_id: str = Field(..., description="Generated unique document identifier")
    filename: str = Field(..., description="Original filename of the PDF")
    total_pages: int = Field(..., description="Total pages in the PDF")
    total_chunks: int = Field(..., description="Total chunks divided")
    indexed_chunks: int = Field(..., description="Number of chunks successfully indexed in ChromaDB")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of relevant chunks to return")
    document_id: Optional[str] = Field(default=None, description="Optional document ID to restrict search results to a specific document")


class SearchResultItem(BaseModel):
    chunk_id: str = Field(..., description="ChromaDB vector chunk ID")
    document_id: str = Field(..., description="Document ID this chunk belongs to")
    filename: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="Page number where the chunk appears")
    chunk_number: int = Field(..., description="Chunk sequence number within document")
    text: str = Field(..., description="Content of the text chunk")
    distance: float = Field(..., description="Vector cosine distance (lower = more similar)")
    similarity_score: float = Field(..., description="Cosine similarity score between 0 and 1 (higher = more similar)")


class SearchResponse(BaseModel):
    query: str = Field(..., description="Executed search query")
    total_results: int = Field(..., description="Number of results found")
    results: list[SearchResultItem] = Field(..., description="List of matching chunks sorted by relevance")


class SourceReference(BaseModel):
    filename: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="Page number where the source information is located")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer using indexed study documents")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of relevant document chunks to retrieve for context")
    document_id: Optional[str] = Field(default=None, description="Optional document ID to scope question answering to a single document")


class AskResponse(BaseModel):
    question: str = Field(..., description="The user's question")
    answer: str = Field(..., description="Generated answer grounded strictly in uploaded documents")
    sources: list[SourceReference] = Field(..., description="List of unique source filenames and page numbers")
    context_used: list[SearchResultItem] = Field(default=[], description="Retrieved context chunks evaluated by the LLM")
    document_id: Optional[str] = Field(default=None, description="Document ID the answer was scoped to, if specified")


class ChapterSummaryItem(BaseModel):
    chapter_title: str = Field(..., description="Title of the detected chapter or section")
    page_range: str = Field(..., description="Page range of the chapter e.g. Pages 1-3")
    summary: str = Field(..., description="Concise chapter summary")


class SummarizeRequest(BaseModel):
    document_id: str = Field(..., min_length=1, description="Document ID to summarize")
    include_chapters: bool = Field(default=True, description="Whether to detect and summarize chapters if identifiable")


class SummarizeResponse(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Filename of the summarized PDF")
    total_pages: int = Field(..., description="Total pages in the PDF")
    total_chunks: int = Field(..., description="Total chunks processed")
    overview: str = Field(..., description="Document overview")
    main_topics: list[str] = Field(..., description="List of main topics")
    important_points: list[str] = Field(..., description="List of important points")
    key_takeaways: list[str] = Field(..., description="List of key takeaways")
    chapters: list[ChapterSummaryItem] = Field(default=[], description="Chapter-wise summaries if chapters were identified")
    full_summary: str = Field(..., description="Full formatted markdown summary")
    source_pages: list[int] = Field(..., description="List of all source pages summarized")


class DocumentSummary(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="PDF filename")
    total_pages: int = Field(..., description="Total pages")
    total_chunks: int = Field(..., description="Total chunks indexed in ChromaDB")


class DocumentsListResponse(BaseModel):
    total_documents: int
    documents: list[DocumentSummary]


class ImportantQuestionItem(BaseModel):
    id: int = Field(..., description="1-based question number")
    question: str = Field(..., description="Exam question text")
    question_type: str = Field(..., description="Question category: 'short', 'long', or 'conceptual'")
    page_reference: str = Field(..., description="Supporting PDF page reference e.g. Page 1 or Pages 1-2")
    page_numbers: list[int] = Field(default=[], description="List of 1-based page numbers supporting the question")
    answer: Optional[str] = Field(default=None, description="Generated answer if available")


class ImportantQuestionsGroup(BaseModel):
    category: str = Field(..., description="Category display title")
    category_id: str = Field(..., description="Category id: 'short', 'long', or 'conceptual'")
    description: str = Field(..., description="Category description")
    questions: list[ImportantQuestionItem] = Field(default=[], description="List of questions in this category")


class ImportantQuestionsRequest(BaseModel):
    document_id: str = Field(..., min_length=1, description="Document ID to generate questions for")
    count: int = Field(default=12, ge=5, le=30, description="Approximate number of questions to generate (default: 12)")


class ImportantQuestionsResponse(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Source PDF filename")
    total_pages: int = Field(..., description="Total pages in the PDF")
    total_questions: int = Field(..., description="Total questions generated")
    categories: list[ImportantQuestionsGroup] = Field(..., description="Questions grouped by category")
    questions: list[ImportantQuestionItem] = Field(..., description="Flat list of all questions with category and page references")


class QuestionAnswerRequest(BaseModel):
    document_id: str = Field(..., min_length=1, description="Document ID")
    question: str = Field(..., min_length=1, description="Question text to answer")
    question_type: Optional[str] = Field(default="short", description="Question category: 'short', 'long', or 'conceptual'")
    page_numbers: Optional[list[int]] = Field(default=[], description="Optional page numbers supporting the question")


class QuestionAnswerResponse(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    question: str = Field(..., description="The answered question")
    question_type: str = Field(..., description="Question category")
    answer: str = Field(..., description="Exam-friendly answer grounded strictly in the PDF")
    page_reference: str = Field(..., description="PDF page reference")
    sources: list[SourceReference] = Field(..., description="List of source filenames and page numbers")


# ---------------------------------------------------------
# CORS Configuration & Allowed Origins Setup
# ---------------------------------------------------------

DEFAULT_ALLOWED_ORIGINS: list[str] = [
    "https://offline-study-ai.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def parse_allowed_origins(origins_val: Optional[str] = None) -> list[str]:
    """Parse, sanitize, and validate allowed CORS origins from an environment variable or raw string.

    Rules:
    - Never permits wildcard '*' when credentials are enabled (Requirement 4).
    - Strips whitespace and trailing slashes for exact origin matching.
    - Supports comma-separated strings or JSON string arrays.
    - Falls back to safe default origins (production Vercel and local Vite dev) if unset or empty (Requirements 2, 3).
    """
    if origins_val is None:
        origins_val = os.getenv("ALLOWED_ORIGINS")

    if origins_val is None:
        return list(DEFAULT_ALLOWED_ORIGINS)

    trimmed = origins_val.strip()
    if not trimmed:
        return list(DEFAULT_ALLOWED_ORIGINS)

    # Support JSON array representation e.g. ["http://localhost:5173", "https://offline-study-ai.vercel.app"]
    origins_to_process: list[str] = []
    if trimmed.startswith("[") and trimmed.endswith("]"):
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, list):
                origins_to_process = [str(item) for item in parsed]
            else:
                origins_to_process = [trimmed]
        except Exception:
            origins_to_process = trimmed.strip("[]").split(",")
    else:
        origins_to_process = trimmed.split(",")

    cleaned: list[str] = []
    for item in origins_to_process:
        origin = item.strip().rstrip("/")
        # Requirement 4: Never use wildcard origins when credentials are enabled
        if not origin or origin == "*":
            continue
        if origin not in cleaned:
            cleaned.append(origin)

    if not cleaned:
        logger.warning(
            "No valid origins found in ALLOWED_ORIGINS (wildcard '*' is disallowed when allow_credentials=True). "
            "Falling back to default allowed origins."
        )
        return list(DEFAULT_ALLOWED_ORIGINS)

    return cleaned


def get_allowed_origins() -> list[str]:
    """Retrieves the active allowed origins list from environment or defaults."""
    return parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))


# ---------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------

app = FastAPI(
    title="Offline Study AI",
    description="Your Personal AI Learning Assistant - Offline RAG with Hugging Face, ChromaDB & Ollama",
    version="1.2.0"
)

# Resolve allowed origins from ALLOWED_ORIGINS env var or secure defaults
ALLOWED_ORIGINS: list[str] = get_allowed_origins()
app.state.allowed_origins = ALLOWED_ORIGINS

# Secure CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Health and Info Endpoints
# ---------------------------------------------------------

@app.get("/")
def home():
    return {"message": "Welcome to Offline Study AI!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ---------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------

@app.post(
    "/auth/register",
    response_model=AuthResponse,
    status_code=201,
    summary="Register a new user account",
    description="Registers a new user with username, email, and password, returning an authenticated JWT session."
)
async def register_endpoint(request: RegisterRequest):
    try:
        return register_user(
            username=request.username,
            email=request.email,
            password=request.password,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


@app.post(
    "/auth/login",
    response_model=AuthResponse,
    summary="Log into user account",
    description="Authenticates user by username or email and password, returning an authenticated JWT session."
)
async def login_endpoint(request: LoginRequest):
    return authenticate_user(
        username_or_email=request.username_or_email,
        password=request.password,
    )


@app.get(
    "/auth/me",
    response_model=UserProfileResponse,
    summary="Get current user profile",
    description="Returns the profile information of the currently authenticated user."
)
async def get_current_user_profile(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    return {
        "user_id": current_user["user_id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"],
    }


@app.post(
    "/auth/google",
    response_model=AuthResponse,
    summary="Sign in or register with Google",
    description="Authenticates or auto-registers a student via Google OAuth ID token or profile details, returning an authenticated JWT session."
)
async def google_auth_endpoint(request: GoogleLoginRequest):
    return authenticate_google_user(
        credential=request.credential,
        email=request.email,
        name=request.name,
        google_id=request.google_id,
    )


@app.post(
    "/auth/forgot-password",
    response_model=ForgotPasswordResponse,
    summary="Request password reset",
    description="Generates a password reset token and 6-digit verification code for a registered user."
)
async def forgot_password_endpoint(request: ForgotPasswordRequest):
    return request_password_reset(email=request.email)


@app.post(
    "/auth/reset-password",
    response_model=ResetPasswordResponse,
    summary="Reset user password",
    description="Resets user password using a valid reset token or 6-digit verification code."
)
async def reset_password_endpoint(request: ResetPasswordRequest):
    return reset_password(
        token_or_code=request.token,
        new_password=request.new_password,
        email=request.email,
    )


@app.get(
    "/documents",
    response_model=DocumentsListResponse,
    summary="List all indexed documents for authenticated user",
    description="Retrieves a list of all documents belonging to the authenticated user currently indexed in ChromaDB."
)
def list_documents(current_user: dict[str, Any] = Depends(get_current_user)):
    docs = get_indexed_documents(user_id=current_user["user_id"])
    return {
        "total_documents": len(docs),
        "documents": docs
    }


@app.delete(
    "/documents/{filename}",
    summary="Delete an indexed document for authenticated user",
    description="Removes all chunks of the specified document belonging to the authenticated user from ChromaDB."
)
def remove_document(
    filename: str,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    deleted_count = delete_document(filename, user_id=current_user["user_id"])
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in your library."
        )
    return {
        "message": f"Successfully deleted '{filename}' from vector store.",
        "deleted_chunks": deleted_count
    }


# ---------------------------------------------------------
# PDF Upload & Chunking Endpoint (Existing Feature)
# ---------------------------------------------------------

@app.post(
    "/upload-pdf",
    response_model=PDFUploadResponse,
    summary="Upload PDF and extract text chunks",
    description="Uploads a PDF file, extracts text page-by-page, and divides it into chunks with page numbers and chunk numbers preserved."
)
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    content = await file.read()
    total_pages, chunks = extract_and_chunk_pdf(
        content=content,
        filename=file.filename or "",
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    return {
        "filename": file.filename,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "chunks": chunks
    }


# ---------------------------------------------------------
# PDF Indexing Endpoint (Hugging Face Embeddings + ChromaDB)
# ---------------------------------------------------------

@app.post(
    "/index-pdf",
    response_model=IndexPDFResponse,
    summary="Index PDF into ChromaDB for authenticated user",
    description="Extracts PDF text, chunks it, generates Hugging Face embeddings (all-MiniLM-L6-v2), and persists vectors into ChromaDB scoped to the authenticated user."
)
async def index_pdf(
    file: UploadFile = File(...),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    content = await file.read()
    total_pages, chunks = extract_and_chunk_pdf(
        content=content,
        filename=file.filename or "",
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    try:
        index_result = index_document(
            filename=file.filename or "unknown.pdf",
            total_pages=total_pages,
            chunks=chunks,
            user_id=current_user["user_id"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index document into ChromaDB: {str(e)}"
        )

    return {
        "message": f"Successfully indexed '{file.filename}' into vector store.",
        "document_id": index_result["document_id"],
        "filename": index_result["filename"],
        "total_pages": index_result["total_pages"],
        "total_chunks": index_result["total_chunks"],
        "indexed_chunks": index_result["indexed_chunks"],
    }


# Alias /index for convenience
@app.post(
    "/index",
    response_model=IndexPDFResponse,
    include_in_schema=False
)
async def index_alias(
    file: UploadFile = File(...),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    return await index_pdf(file, current_user=current_user)


# ---------------------------------------------------------
# Vector Search Endpoints
# ---------------------------------------------------------

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Search indexed documents",
    description="Searches indexed PDF chunks belonging to the authenticated user using a semantic text query with Hugging Face embeddings."
)
async def search_endpoint(
    request: SearchRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    try:
        results = search_documents(
            query=request.query.strip(),
            top_k=request.top_k,
            document_id=request.document_id,
            user_id=current_user["user_id"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing search query: {str(e)}"
        )

    return {
        "query": request.query.strip(),
        "total_results": len(results),
        "results": results
    }


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Search indexed documents (GET query)",
    description="Convenience GET endpoint to search indexed chunks via query parameters."
)
async def search_get_endpoint(
    query: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(default=5, ge=1, le=50, description="Maximum number of relevant chunks to return"),
    document_id: Optional[str] = Query(default=None, description="Optional document ID to restrict search results"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    actual_query = query if not hasattr(query, "default") else ""
    actual_top_k = top_k if not hasattr(top_k, "default") else 5
    actual_doc_id = document_id if not hasattr(document_id, "default") else None
    return await search_endpoint(
        SearchRequest(query=actual_query, top_k=actual_top_k, document_id=actual_doc_id),
        current_user=current_user,
    )


# ---------------------------------------------------------
# Question-Answering Endpoint (RAG + Ollama qwen2.5:1.5b)
# ---------------------------------------------------------

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about indexed documents",
    description="Retrieves the most relevant chunks from ChromaDB and generates an answer strictly grounded in the documents using local Ollama (qwen2.5:1.5b). Supports document_id to focus exclusively on a single PDF."
)
async def ask_endpoint(
    request: AskRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = answer_study_question(
            question=request.question.strip(),
            top_k=request.top_k,
            document_id=request.document_id,
            user_id=current_user["user_id"],
        )
        return result
    except OllamaConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama connection error: {str(e)}"
        )
    except OllamaModelNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama model error: {str(e)}"
        )
    except OllamaError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama generation error: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


# ---------------------------------------------------------
# Document Summarization Endpoint (Map-Reduce with Ollama)
# ---------------------------------------------------------

@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Generate a complete document summary",
    description="Generates a structured, student-friendly document summary with Overview, Main Topics, Important Points, Key Takeaways, and optional Chapter breakdowns using local Ollama (qwen2.5:1.5b)."
)
async def summarize_endpoint(
    request: SummarizeRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    if not request.document_id or not request.document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Document ID cannot be empty."
        )

    try:
        return summarize_document(
            document_id=request.document_id.strip(),
            include_chapters=request.include_chapters,
            user_id=current_user["user_id"],
        )
    except OllamaConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama connection error: {str(e)}"
        )
    except OllamaModelNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama model error: {str(e)}"
        )
    except OllamaError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama generation error: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating document summary: {str(e)}"
        )


@app.get(
    "/documents/{document_id}/summary",
    response_model=SummarizeResponse,
    summary="Get or generate document summary (GET)",
    description="Convenience GET endpoint to summarize a document by its document_id."
)
async def get_document_summary_endpoint(
    document_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    return await summarize_endpoint(
        SummarizeRequest(document_id=document_id),
        current_user=current_user,
    )


# ---------------------------------------------------------
# Important Questions Generator Endpoints (Ollama qwen2.5:1.5b)
# ---------------------------------------------------------

@app.post(
    "/important-questions",
    response_model=ImportantQuestionsResponse,
    summary="Generate important exam study questions",
    description="Generates 10-15 important study questions categorized into Short-Answer, Long-Answer, and Conceptual questions covering the complete selected PDF using local Ollama (qwen2.5:1.5b)."
)
async def generate_important_questions_endpoint(
    request: ImportantQuestionsRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    if not request.document_id or not request.document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Document ID cannot be empty."
        )

    try:
        return generate_important_questions(
            document_id=request.document_id.strip(),
            count=request.count,
            user_id=current_user["user_id"],
        )
    except OllamaConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama connection error: {str(e)}"
        )
    except OllamaModelNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama model error: {str(e)}"
        )
    except OllamaError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama generation error: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating important questions: {str(e)}"
        )


@app.get(
    "/documents/{document_id}/important-questions",
    response_model=ImportantQuestionsResponse,
    summary="Get or generate important questions (GET)",
    description="Convenience GET endpoint to generate important questions by document_id."
)
async def get_document_important_questions_endpoint(
    document_id: str,
    count: int = Query(default=12, ge=5, le=30, description="Approximate number of questions to generate"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    return await generate_important_questions_endpoint(
        ImportantQuestionsRequest(document_id=document_id, count=count),
        current_user=current_user,
    )


@app.post(
    "/important-questions/answer",
    response_model=QuestionAnswerResponse,
    summary="Generate exam-friendly answer for an individual question",
    description="Generates a simple, exam-friendly answer for an individual study question grounded strictly in the selected PDF."
)
async def answer_individual_question_endpoint(
    request: QuestionAnswerRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    if not request.document_id or not request.document_id.strip():
        raise HTTPException(status_code=400, detail="Document ID cannot be empty.")
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return generate_question_answer(
            document_id=request.document_id.strip(),
            question=request.question.strip(),
            question_type=request.question_type or "short",
            page_numbers=request.page_numbers or [],
            user_id=current_user["user_id"],
        )
    except OllamaConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Ollama connection error: {str(e)}")
    except OllamaModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Ollama model error: {str(e)}")
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=f"Ollama generation error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating question answer: {str(e)}")
