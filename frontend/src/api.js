/**
 * API Service for Offline Study AI.
 * Communicates with the local FastAPI backend on http://127.0.0.1:8000.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

/**
 * Helper to handle fetch responses and extract user-friendly error messages.
 */
async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const data = await response.json();
      if (data && data.detail) {
        errorDetail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      }
    } catch {
      // Fallback to status text
      errorDetail = response.statusText || errorDetail;
    }
    const error = new Error(errorDetail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

/**
 * Checks backend health and availability.
 */
export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    return await handleResponse(res);
  } catch (err) {
    throw new Error('Cannot connect to the Offline Study AI backend. Make sure the server is running at http://127.0.0.1:8000.');
  }
}

/**
 * Lists all documents currently indexed in ChromaDB.
 */
export async function fetchIndexedDocuments() {
  const res = await fetch(`${API_BASE_URL}/documents`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  return await handleResponse(res);
}

/**
 * Uploads and indexes a PDF document into ChromaDB.
 * @param {File} file - PDF file to upload.
 */
export async function uploadAndIndexPdf(file) {
  if (!file) {
    throw new Error('Please select a PDF file to upload.');
  }
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    throw new Error('Only .pdf files are supported.');
  }

  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE_URL}/index-pdf`, {
    method: 'POST',
    body: formData,
  });
  return await handleResponse(res);
}

/**
 * Asks the AI a question using RAG and local Ollama (qwen2.5:1.5b).
 * @param {string} question - The student's question.
 * @param {number} topK - Number of chunks to retrieve (default: 4).
 * @param {string|null} documentId - Optional document ID to scope chat to a specific PDF.
 */
export async function askAiQuestion(question, topK = 4, documentId = null) {
  if (!question || !question.trim()) {
    throw new Error('Please enter a question.');
  }

  const payload = {
    question: question.trim(),
    top_k: topK,
  };
  if (documentId) {
    payload.document_id = documentId;
  }

  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  return await handleResponse(res);
}

/**
 * Searches indexed document chunks using semantic similarity.
 * @param {string} query - Search query text.
 * @param {number} topK - Maximum results (default: 5).
 */
export async function searchDocumentChunks(query, topK = 5) {
  const res = await fetch(`${API_BASE_URL}/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      query: query.trim(),
      top_k: topK,
    }),
  });
  return await handleResponse(res);
}

/**
 * Deletes an indexed document from ChromaDB.
 * @param {string} filename - Name of the file to remove.
 */
export async function deleteIndexedDocument(filename) {
  const res = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    headers: { 'Accept': 'application/json' },
  });
  return await handleResponse(res);
}

/**
 * Generates a complete document summary with Overview, Main Topics, Important Points,
 * Key Takeaways, and Chapter breakdowns using local Ollama.
 * @param {string} documentId - The document ID to summarize.
 * @param {boolean} includeChapters - Whether to detect and summarize chapters.
 */
export async function summarizeDocument(documentId, includeChapters = true) {
  if (!documentId) {
    throw new Error('Please specify a document to summarize.');
  }

  const res = await fetch(`${API_BASE_URL}/summarize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      document_id: documentId,
      include_chapters: includeChapters,
    }),
  });
  return await handleResponse(res);
}

/**
 * Generates 10-15 important study questions organized into Short-Answer,
 * Long-Answer, and Conceptual categories based on the complete PDF.
 * @param {string} documentId - The document ID to generate questions for.
 * @param {number} count - Desired number of questions (default: 12).
 */
export async function generateImportantQuestions(documentId, count = 12) {
  if (!documentId) {
    throw new Error('Please select a document to generate questions.');
  }

  const res = await fetch(`${API_BASE_URL}/important-questions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      document_id: documentId,
      count: count,
    }),
  });
  return await handleResponse(res);
}

/**
 * Generates a simple, exam-friendly answer for an individual question.
 * @param {string} documentId - The document ID.
 * @param {string} question - Question text.
 * @param {string} questionType - Question category ('short', 'long', 'conceptual').
 * @param {Array<number>} pageNumbers - Optional list of supporting page numbers.
 */
export async function generateQuestionAnswer(documentId, question, questionType = 'short', pageNumbers = []) {
  if (!documentId) {
    throw new Error('Document ID is required.');
  }
  if (!question || !question.trim()) {
    throw new Error('Question text is required.');
  }

  const res = await fetch(`${API_BASE_URL}/important-questions/answer`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      document_id: documentId,
      question: question.trim(),
      question_type: questionType,
      page_numbers: pageNumbers || [],
    }),
  });
  return await handleResponse(res);
}
