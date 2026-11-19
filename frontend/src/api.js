/**
 * API Service for Offline Study AI.
 * Communicates with the local FastAPI backend on http://127.0.0.1:8000.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// Storage keys for authentication tokens and user state
const TOKEN_STORAGE_KEY = 'study_ai_token';
const USER_STORAGE_KEY = 'study_ai_user';

/**
 * Retrieves the stored JWT authentication token from localStorage.
 */
export function getStoredToken() {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

/**
 * Retrieves the stored user profile object from localStorage.
 */
export function getStoredUser() {
  try {
    const raw = localStorage.getItem(USER_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/**
 * Persists authenticated JWT token and user profile into localStorage.
 */
export function saveAuthSession(token, user) {
  try {
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    }
    if (user) {
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
    }
  } catch (err) {
    console.warn('Could not save auth session to localStorage:', err);
  }
}

/**
 * Clears authenticated token and user profile from localStorage.
 */
export function clearAuthSession() {
  try {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    localStorage.removeItem(USER_STORAGE_KEY);
  } catch (err) {
    console.warn('Could not clear auth session from localStorage:', err);
  }
}

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
 * Executes a network fetch with friendly diagnostics for offline and backend connection states.
 * Automatically injects the stored JWT Bearer token into headers if available.
 */
async function safeFetch(url, options = {}) {
  const token = getStoredToken();
  const headers = new Headers(options.headers || {});

  // Automatically attach Authorization Bearer header if token exists and not manually provided
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const mergedOptions = {
    ...options,
    headers,
  };

  try {
    const response = await fetch(url, mergedOptions);
    return await handleResponse(response);
  } catch (err) {
    // If it's an HTTP error handled by handleResponse, handle 401 and propagate
    if (err.status) {
      if (err.status === 401 && !url.includes('/auth/login') && !url.includes('/auth/register')) {
        // Expired or invalid token encountered on protected endpoint
        clearAuthSession();
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('study-ai:unauthorized', { detail: { error: err } }));
        }
      }
      throw err;
    }

    const isBrowserOffline = typeof navigator !== 'undefined' && !navigator.onLine;

    if (isBrowserOffline) {
      throw new Error(
        'You are currently offline. The application shell is loaded from offline cache, but local AI processing requires the local Python FastAPI backend server to be reachable on http://127.0.0.1:8000.'
      );
    }

    throw new Error(
      `Cannot connect to Offline Study AI backend at ${API_BASE_URL}. Please ensure your local FastAPI backend server is running with: python -m uvicorn backend.main:app --reload`
    );
  }
}

/**
 * Registers a new user account with username, email, and password.
 */
export async function registerUser(username, email, password) {
  const data = await safeFetch(`${API_BASE_URL}/auth/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({ username, email, password }),
  });
  if (data?.access_token && data?.user) {
    saveAuthSession(data.access_token, data.user);
  }
  return data;
}

/**
 * Authenticates an existing user by username or email and password.
 */
export async function loginUser(usernameOrEmail, password) {
  const data = await safeFetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      username_or_email: usernameOrEmail,
      password,
    }),
  });
  if (data?.access_token && data?.user) {
    saveAuthSession(data.access_token, data.user);
  }
  return data;
}

/**
 * Authenticates or registers a student via Google Sign-In.
 * @param {Object} payload - { credential } or { email, name, google_id }
 */
export async function loginWithGoogle(payload) {
  const data = await safeFetch(`${API_BASE_URL}/auth/google`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  if (data?.access_token && data?.user) {
    saveAuthSession(data.access_token, data.user);
  }
  return data;
}

/**
 * Requests a password reset token and 6-digit code for the specified email.
 * @param {string} email
 */
export async function requestForgotPassword(email) {
  return await safeFetch(`${API_BASE_URL}/auth/forgot-password`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({ email: (email || '').trim() }),
  });
}

/**
 * Resets user password using the verification token / code.
 * @param {string} token - Reset token or 6-digit code
 * @param {string} newPassword - New password
 * @param {string} email - Optional email for confirmation
 */
export async function resetUserPassword(token, newPassword, email = null) {
  const payload = {
    token: (token || '').trim(),
    new_password: newPassword,
  };
  if (email && email.trim()) {
    payload.email = email.trim();
  }

  return await safeFetch(`${API_BASE_URL}/auth/reset-password`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });
}

/**
 * Retrieves the current authenticated user's profile from /auth/me.
 */
export async function fetchCurrentUser() {
  const data = await safeFetch(`${API_BASE_URL}/auth/me`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (data) {
    const existing = getStoredUser() || {};
    saveAuthSession(getStoredToken(), {
      ...existing,
      user_id: data.user_id,
      username: data.username,
      email: data.email,
    });
  }
  return data;
}

/**
 * Clears session and logs out the current user.
 */
export function logoutUser() {
  clearAuthSession();
}

/**
 * Checks backend health and availability.
 */
export async function checkBackendHealth() {
  return await safeFetch(`${API_BASE_URL}/health`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
}

/**
 * Lists all documents currently indexed in ChromaDB.
 */
export async function fetchIndexedDocuments() {
  return await safeFetch(`${API_BASE_URL}/documents`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
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

  return await safeFetch(`${API_BASE_URL}/index-pdf`, {
    method: 'POST',
    body: formData,
  });
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

  return await safeFetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });
}

/**
 * Searches indexed document chunks using semantic similarity.
 * @param {string} query - Search query text.
 * @param {number} topK - Maximum results (default: 5).
 */
export async function searchDocumentChunks(query, topK = 5) {
  return await safeFetch(`${API_BASE_URL}/search`, {
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
}

/**
 * Deletes an indexed document from ChromaDB.
 * @param {string} filename - Name of the file to remove.
 */
export async function deleteIndexedDocument(filename) {
  return await safeFetch(`${API_BASE_URL}/documents/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    headers: { 'Accept': 'application/json' },
  });
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

  return await safeFetch(`${API_BASE_URL}/summarize`, {
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

  return await safeFetch(`${API_BASE_URL}/important-questions`, {
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

  return await safeFetch(`${API_BASE_URL}/important-questions/answer`, {
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
}
