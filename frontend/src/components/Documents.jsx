import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, Trash2, MessageSquare, Loader2, Sparkles, FileQuestion, ServerOff } from 'lucide-react';
import { uploadAndIndexPdf, deleteIndexedDocument } from '../api';

export default function Documents({
  documents = [],
  onRefreshDocuments,
  setActiveTab,
  onSelectDocumentForChat,
  onSelectDocumentForSummarize,
  onSelectDocumentForQuestions,
  isBackendHealthy = true,
  currentUser = null,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState(null); // { type: 'success'|'error', text: '', details: null }
  const [deletingFilename, setDeletingFilename] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file) return;

    if (!currentUser) {
      setStatusMessage({
        type: 'error',
        text: 'Authentication required. Please sign in or create an account to index PDFs into your private library.',
      });
      if (onOpenAuth) onOpenAuth();
      return;
    }

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setStatusMessage({
        type: 'error',
        text: 'Only PDF documents (.pdf) are allowed. Please select a valid PDF file.',
      });
      return;
    }

    setIsUploading(true);
    setUploadProgress(25);
    setStatusMessage(null);

    try {
      setUploadProgress(50);
      const res = await uploadAndIndexPdf(file);
      setUploadProgress(100);

      setStatusMessage({
        type: 'success',
        text: `Successfully uploaded and indexed "${file.name}"!`,
        details: `${res.total_pages} pages processed into ${res.indexed_chunks} study chunks in ChromaDB.`,
      });

      // Refresh document list
      if (onRefreshDocuments) {
        await onRefreshDocuments();
      }
    } catch (err) {
      setStatusMessage({
        type: 'error',
        text: err.message || 'Failed to upload and index document.',
      });
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDelete = async (filename) => {
    if (!window.confirm(`Are you sure you want to remove "${filename}" from your study library?`)) {
      return;
    }

    setDeletingFilename(filename);
    try {
      await deleteIndexedDocument(filename);
      setStatusMessage({
        type: 'success',
        text: `Document "${filename}" was removed from ChromaDB.`,
      });
      if (onRefreshDocuments) {
        await onRefreshDocuments();
      }
    } catch (err) {
      setStatusMessage({
        type: 'error',
        text: `Could not delete "${filename}": ${err.message}`,
      });
    } finally {
      setDeletingFilename(null);
    }
  };

  return (
    <div className="page-container">
      {/* Backend Offline Warning Banner */}
      {!isBackendHealthy && (
        <div className="card backend-warning-card" style={{ marginBottom: '1.5rem', borderColor: '#fecaca', backgroundColor: '#fff5f5' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.85rem' }}>
            <ServerOff size={24} color="#dc2626" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <h4 style={{ color: '#991b1b', fontWeight: 700, marginBottom: '0.25rem' }}>
                PDF Upload & Indexing Unavailable (Backend Offline)
              </h4>
              <p style={{ fontSize: '0.875rem', color: '#7f1d1d', marginBottom: '0.5rem' }}>
                The frontend application shell is cached and running offline. Document text extraction, embedding generation, and ChromaDB storage require the local FastAPI backend server:
              </p>
              <code style={{ display: 'inline-block', backgroundColor: '#fee2e2', color: '#991b1b', padding: '0.35rem 0.65rem', borderRadius: '6px', fontSize: '0.85rem', fontFamily: 'monospace' }}>
                python -m uvicorn backend.main:app --reload
              </code>
            </div>
          </div>
        </div>
      )}

      {/* Upload Dropzone */}
      <div
        className={`upload-dropzone ${isDragging ? 'dragging' : ''} ${!isBackendHealthy ? 'disabled' : ''}`}
        onDragOver={(e) => isBackendHealthy && handleDragOver(e)}
        onDragLeave={(e) => isBackendHealthy && handleDragLeave(e)}
        onDrop={(e) => isBackendHealthy && handleDrop(e)}
        onClick={() => isBackendHealthy && !isUploading && fileInputRef.current?.click()}
        style={!isBackendHealthy ? { opacity: 0.65, cursor: 'not-allowed', backgroundColor: '#f1f5f9' } : {}}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
          disabled={isUploading || !isBackendHealthy}
        />

        <div className="upload-icon-circle">
          {isUploading ? (
            <Loader2 size={32} className="spinner" />
          ) : (
            <UploadCloud size={32} />
          )}
        </div>

        <h3 className="upload-title">
          {isUploading ? 'Indexing Your PDF Notes...' : 'Upload Study Documents (PDF)'}
        </h3>
        
        <p className="upload-subtitle">
          {isUploading
            ? 'Extracting text page-by-page, generating all-MiniLM-L6-v2 embeddings, and storing in ChromaDB...'
            : 'Drag and drop your PDF notes here, or click to browse files from your computer'}
        </p>

        {isUploading ? (
          <div style={{ maxWidth: '400px', margin: '0 auto' }}>
            <div className="progress-container">
              <div
                className="progress-bar-fill"
                style={{ width: `${uploadProgress || 65}%` }}
              ></div>
            </div>
            <p style={{ fontSize: '0.8rem', color: '#64748b' }}>Processing offline in background...</p>
          </div>
        ) : (
          <button
            type="button"
            className="btn-primary"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
          >
            <UploadCloud size={18} /> Select PDF File
          </button>
        )}
      </div>

      {/* Status Messages */}
      {statusMessage && (
        <div className={`alert alert-${statusMessage.type === 'success' ? 'success' : 'danger'}`}>
          {statusMessage.type === 'success' ? (
            <CheckCircle2 size={22} style={{ flexShrink: 0 }} />
          ) : (
            <AlertCircle size={22} style={{ flexShrink: 0 }} />
          )}
          <div style={{ flex: 1 }}>
            <p style={{ fontWeight: 600 }}>{statusMessage.text}</p>
            {statusMessage.details && (
              <p style={{ fontSize: '0.85rem', opacity: 0.9 }}>{statusMessage.details}</p>
            )}
          </div>
          <button
            onClick={() => setStatusMessage(null)}
            style={{ background: 'transparent', color: 'inherit', fontWeight: 'bold', fontSize: '1.1rem' }}
          >
            ×
          </button>
        </div>
      )}

      {/* Document Library Table */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>
              My Study Library ({documents.length})
            </h3>
            <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
              All indexed notes stored locally in ChromaDB and ready for AI questioning.
            </p>
          </div>
        </div>

        {documents.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '3rem 1rem', color: '#64748b' }}>
            <FileText size={48} style={{ color: '#cbd5e1', margin: '0 auto 1rem auto' }} />
            <h4 style={{ color: '#334155', marginBottom: '0.25rem' }}>No documents uploaded yet</h4>
            <p style={{ fontSize: '0.875rem' }}>
              Use the upload area above to add lecture slides, textbooks, or notes.
            </p>
          </div>
        ) : (
          <div className="docs-table-wrapper">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Document Name</th>
                  <th>Pages</th>
                  <th>Indexed Chunks</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.filename}>
                    <td>
                      <div className="file-name-cell">
                        <div style={{ color: '#2563eb' }}>
                          <FileText size={20} />
                        </div>
                        <div>
                          <span>{doc.filename}</span>
                          <span style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', fontWeight: 400 }}>
                            {doc.document_id}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className="badge badge-blue">
                        {doc.total_pages} {doc.total_pages === 1 ? 'Page' : 'Pages'}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontWeight: 600, color: '#334155' }}>
                        {doc.total_chunks} chunks
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', flexWrap: 'wrap' }}>
                        <button
                          className="btn-summarize"
                          onClick={() => {
                            if (onSelectDocumentForSummarize) {
                              onSelectDocumentForSummarize(doc);
                            } else {
                              setActiveTab('summarizer');
                            }
                          }}
                          title={`Generate complete AI study summary for ${doc.filename}`}
                        >
                          <Sparkles size={15} /> Summarize
                        </button>
                        <button
                          className="btn-questions"
                          onClick={() => {
                            if (onSelectDocumentForQuestions) {
                              onSelectDocumentForQuestions(doc);
                            } else {
                              setActiveTab('questions');
                            }
                          }}
                          title={`Generate Important Exam Questions for ${doc.filename}`}
                        >
                          <FileQuestion size={15} /> Questions
                        </button>
                        <button
                          className="btn-secondary"
                          style={{ padding: '0.45rem 0.85rem', fontSize: '0.825rem' }}
                          onClick={() => {
                            if (onSelectDocumentForChat) {
                              onSelectDocumentForChat(doc);
                            } else {
                              setActiveTab('chat');
                            }
                          }}
                          title={`Ask AI questions specifically about ${doc.filename}`}
                        >
                          <MessageSquare size={15} /> Ask AI
                        </button>
                        <button
                          className="btn-danger"
                          onClick={() => handleDelete(doc.filename)}
                          disabled={deletingFilename === doc.filename}
                          title="Delete from vector store"
                        >
                          {deletingFilename === doc.filename ? (
                            <Loader2 size={16} className="spinner" />
                          ) : (
                            <Trash2 size={16} />
                          )}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
