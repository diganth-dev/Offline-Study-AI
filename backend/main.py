from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Optional

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


# ---------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------

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
# FastAPI App Initialization
# ---------------------------------------------------------

app = FastAPI(
    title="Offline Study AI",
    description="Your Personal AI Learning Assistant - Offline RAG with Hugging Face, ChromaDB & Ollama",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get(
    "/documents",
    response_model=DocumentsListResponse,
    summary="List all indexed documents",
    description="Retrieves a list of all documents currently indexed in ChromaDB."
)
def list_documents():
    docs = get_indexed_documents()
    return {
        "total_documents": len(docs),
        "documents": docs
    }


@app.delete(
    "/documents/{filename}",
    summary="Delete an indexed document",
    description="Removes all chunks of the specified document from ChromaDB."
)
def remove_document(filename: str):
    deleted_count = delete_document(filename)
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in the vector store."
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
async def upload_pdf(file: UploadFile = File(...)):
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
    summary="Index PDF into ChromaDB",
    description="Extracts PDF text, chunks it, generates Hugging Face embeddings (all-MiniLM-L6-v2), and persists vectors into ChromaDB."
)
async def index_pdf(file: UploadFile = File(...)):
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
async def index_alias(file: UploadFile = File(...)):
    return await index_pdf(file)


# ---------------------------------------------------------
# Vector Search Endpoints
# ---------------------------------------------------------

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Search indexed documents",
    description="Searches indexed PDF chunks using a semantic text query with Hugging Face embeddings."
)
async def search_endpoint(request: SearchRequest):
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
    document_id: Optional[str] = Query(default=None, description="Optional document ID to restrict search results")
):
    actual_query = query if not hasattr(query, "default") else ""
    actual_top_k = top_k if not hasattr(top_k, "default") else 5
    actual_doc_id = document_id if not hasattr(document_id, "default") else None
    return await search_endpoint(SearchRequest(query=actual_query, top_k=actual_top_k, document_id=actual_doc_id))


# ---------------------------------------------------------
# Question-Answering Endpoint (RAG + Ollama qwen2.5:1.5b)
# ---------------------------------------------------------

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about indexed documents",
    description="Retrieves the most relevant chunks from ChromaDB and generates an answer strictly grounded in the documents using local Ollama (qwen2.5:1.5b). Supports document_id to focus exclusively on a single PDF."
)
async def ask_endpoint(request: AskRequest):
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
async def summarize_endpoint(request: SummarizeRequest):
    if not request.document_id or not request.document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Document ID cannot be empty."
        )

    try:
        return summarize_document(
            document_id=request.document_id.strip(),
            include_chapters=request.include_chapters,
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
async def get_document_summary_endpoint(document_id: str):
    return await summarize_endpoint(SummarizeRequest(document_id=document_id))


# ---------------------------------------------------------
# Important Questions Generator Endpoints (Ollama qwen2.5:1.5b)
# ---------------------------------------------------------

@app.post(
    "/important-questions",
    response_model=ImportantQuestionsResponse,
    summary="Generate important exam study questions",
    description="Generates 10-15 important study questions categorized into Short-Answer, Long-Answer, and Conceptual questions covering the complete selected PDF using local Ollama (qwen2.5:1.5b)."
)
async def generate_important_questions_endpoint(request: ImportantQuestionsRequest):
    if not request.document_id or not request.document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Document ID cannot be empty."
        )

    try:
        return generate_important_questions(
            document_id=request.document_id.strip(),
            count=request.count,
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
    count: int = Query(default=12, ge=5, le=30, description="Approximate number of questions to generate")
):
    return await generate_important_questions_endpoint(
        ImportantQuestionsRequest(document_id=document_id, count=count)
    )


@app.post(
    "/important-questions/answer",
    response_model=QuestionAnswerResponse,
    summary="Generate exam-friendly answer for an individual question",
    description="Generates a simple, exam-friendly answer for an individual study question grounded strictly in the selected PDF."
)
async def answer_individual_question_endpoint(request: QuestionAnswerRequest):
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
