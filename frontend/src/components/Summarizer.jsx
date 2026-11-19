import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  BookOpen,
  FileText,
  Copy,
  Check,
  RotateCcw,
  MessageSquare,
  AlertCircle,
  Loader2,
  ChevronRight,
  Layers,
  ListOrdered,
  Lightbulb,
  Bookmark,
  Share2,
  ServerOff,
} from 'lucide-react';
import { summarizeDocument } from '../api';
import MarkdownRenderer from './MarkdownRenderer';

export default function Summarizer({
  documents = [],
  selectedDocument = null,
  onSelectDocument,
  setActiveTab,
  onSelectDocumentForChat,
  isBackendHealthy = true,
  currentUser = null,
  onOpenAuth,
}) {
  const [currentDocId, setCurrentDocId] = useState(
    selectedDocument ? selectedDocument.document_id : (documents[0]?.document_id || '')
  );
  const [summaryData, setSummaryData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progressStage, setProgressStage] = useState('');
  const [errorMessage, setErrorMessage] = useState(null);
  const [includeChapters, setIncludeChapters] = useState(true);
  const [copied, setCopied] = useState(false);

  // Sync with selectedDocument prop if it changes
  useEffect(() => {
    if (selectedDocument && selectedDocument.document_id !== currentDocId) {
      setCurrentDocId(selectedDocument.document_id);
      setSummaryData(null);
      setErrorMessage(null);
    }
  }, [selectedDocument]);

  const activeDoc = documents.find((d) => d.document_id === currentDocId) || selectedDocument || documents[0];

  const handleGenerateSummary = async (docId = currentDocId) => {
    if (!docId) {
      setErrorMessage('Please select a document from the list to summarize.');
      return;
    }

    if (!currentUser) {
      setErrorMessage('Authentication required. Please sign in or create an account to summarize your documents.');
      if (onOpenAuth) onOpenAuth();
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setProgressStage('Step 1 of 3: Loading document chunks from ChromaDB...');

    try {
      setTimeout(() => {
        setProgressStage('Step 2 of 3: Analyzing document with local Ollama (qwen2.5:1.5b)...');
      }, 1500);

      setTimeout(() => {
        setProgressStage('Step 3 of 3: Synthesizing Overview, Topics, Points & Key Takeaways...');
      }, 4000);

      const result = await summarizeDocument(docId, includeChapters);
      setSummaryData(result);
    } catch (err) {
      setErrorMessage(err.message || 'Failed to generate summary.');
    } finally {
      setIsLoading(false);
      setProgressStage('');
    }
  };

  const handleCopySummary = () => {
    if (!summaryData) return;
    const textToCopy = summaryData.full_summary || `${summaryData.overview}\n\nImportant Points:\n${summaryData.important_points.join('\n')}`;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handleAskAboutSummary = () => {
    if (!activeDoc) return;
    if (onSelectDocumentForChat) {
      onSelectDocumentForChat(activeDoc);
    } else {
      setActiveTab('chat');
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
                AI Summarizer Unavailable (Backend Offline)
              </h4>
              <p style={{ fontSize: '0.875rem', color: '#7f1d1d', marginBottom: '0.5rem' }}>
                The frontend shell is loaded from offline cache. Generating complete summaries with local Ollama requires the Python FastAPI backend server to be running:
              </p>
              <code style={{ display: 'inline-block', backgroundColor: '#fee2e2', color: '#991b1b', padding: '0.35rem 0.65rem', borderRadius: '6px', fontSize: '0.85rem', fontFamily: 'monospace' }}>
                python -m uvicorn backend.main:app --reload
              </code>
            </div>
          </div>
        </div>
      )}

      {/* Header Bar */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div style={{ width: '42px', height: '42px', borderRadius: '10px', backgroundColor: '#eff6ff', color: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Sparkles size={24} />
            </div>
            <div>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>
                AI Study Summarizer
              </h3>
              <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
                Generates a structured, student-friendly study guide with Overview, Topics, Key Points, and Chapter summaries.
              </p>
            </div>
          </div>

          {/* Document Picker Dropdown */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            {documents.length > 0 && (
              <select
                className="doc-select-dropdown"
                style={{ padding: '0.5rem 0.85rem', fontSize: '0.85rem', maxWidth: '300px' }}
                value={currentDocId}
                onChange={(e) => {
                  const newId = e.target.value;
                  setCurrentDocId(newId);
                  setSummaryData(null);
                  setErrorMessage(null);
                  const newDoc = documents.find((d) => d.document_id === newId);
                  if (newDoc && onSelectDocument) {
                    onSelectDocument(newDoc);
                  }
                }}
              >
                {documents.map((doc) => (
                  <option key={doc.document_id} value={doc.document_id}>
                    📄 {doc.filename} ({doc.total_pages} {doc.total_pages === 1 ? 'page' : 'pages'})
                  </option>
                ))}
              </select>
            )}

            <button
              className="btn-primary"
              onClick={() => handleGenerateSummary(currentDocId)}
              disabled={isLoading || !currentDocId || !isBackendHealthy}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', padding: '0.55rem 1.15rem' }}
              title={!isBackendHealthy ? 'Backend offline' : 'Generate Summary'}
            >
              {isLoading ? (
                <>
                  <Loader2 size={16} className="spinner" /> Generating...
                </>
              ) : (
                <>
                  <Sparkles size={16} /> Generate Summary
                </>
              )}
            </button>
          </div>
        </div>

        {/* Selected Document Info Banner */}
        {activeDoc && (
          <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: '#334155' }}>
              <FileText size={16} color="#2563eb" />
              <span>Active PDF: <strong>{activeDoc.filename}</strong></span>
              <span className="badge badge-blue">{activeDoc.total_pages} {activeDoc.total_pages === 1 ? 'Page' : 'Pages'}</span>
              <span style={{ color: '#64748b' }}>({activeDoc.total_chunks} chunks in ChromaDB)</span>
            </div>

            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', color: '#475569', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={includeChapters}
                onChange={(e) => setIncludeChapters(e.target.checked)}
                disabled={isLoading}
              />
              Detect and summarize chapters
            </label>
          </div>
        )}
      </div>

      {/* Error Banner */}
      {errorMessage && (
        <div className="alert alert-danger" style={{ marginBottom: '1.5rem' }}>
          <AlertCircle size={22} style={{ flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <p style={{ fontWeight: 600 }}>Summarization Error</p>
            <p style={{ fontSize: '0.85rem' }}>{errorMessage}</p>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            style={{ background: 'transparent', color: 'inherit', fontWeight: 'bold' }}
          >
            ×
          </button>
        </div>
      )}

      {/* Loading State with Progress Steps */}
      {isLoading && (
        <div className="card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem', marginBottom: '1.5rem' }}>
          <div style={{ width: '56px', height: '56px', borderRadius: '50%', backgroundColor: '#eff6ff', color: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.25rem auto' }}>
            <Loader2 size={32} className="spinner" />
          </div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.5rem' }}>
            Synthesizing Document Summary...
          </h3>
          <p style={{ fontSize: '0.9rem', color: '#64748b', maxWidth: '480px', margin: '0 auto 1.5rem auto' }}>
            Processing full document chunks with local Ollama (<code style={{ color: '#2563eb' }}>qwen2.5:1.5b</code>). This ensures complete coverage across all pages.
          </p>

          <div style={{ maxWidth: '420px', margin: '0 auto' }}>
            <div className="progress-container">
              <div className="progress-bar-fill" style={{ width: '85%' }}></div>
            </div>
            <p style={{ fontSize: '0.825rem', fontWeight: 600, color: '#1d4ed8', marginTop: '0.75rem' }}>
              {progressStage}
            </p>
          </div>
        </div>
      )}

      {/* Empty State before first generation */}
      {!isLoading && !summaryData && !errorMessage && (
        <div className="card" style={{ textAlign: 'center', padding: '4rem 1.5rem' }}>
          <div style={{ width: '60px', height: '60px', borderRadius: '50%', backgroundColor: '#f1f5f9', color: '#64748b', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.25rem auto' }}>
            <BookOpen size={30} />
          </div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.5rem' }}>
            {activeDoc ? `Ready to summarize "${activeDoc.filename}"` : 'No document selected'}
          </h3>
          <p style={{ fontSize: '0.9rem', color: '#64748b', maxWidth: '520px', margin: '0 auto 1.5rem auto' }}>
            Click the "Generate Summary" button to analyze all chunks and extract the Document Overview, Main Topics, Important Points, and Key Takeaways.
          </p>
          {activeDoc && (
            <button
              className="btn-primary"
              onClick={() => handleGenerateSummary(activeDoc.document_id)}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 1.5rem' }}
            >
              <Sparkles size={18} /> Generate Complete Summary
            </button>
          )}
        </div>
      )}

      {/* Generated Summary View */}
      {!isLoading && summaryData && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Action Toolbar */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="badge badge-green" style={{ fontSize: '0.8rem', padding: '0.35rem 0.65rem' }}>
                Summary Ready
              </span>
              <span style={{ fontSize: '0.85rem', color: '#64748b' }}>
                {summaryData.total_pages} pages processed • {summaryData.total_chunks} chunks combined
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <button
                className="btn-secondary"
                onClick={handleCopySummary}
                style={{ padding: '0.45rem 0.85rem', fontSize: '0.825rem', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
              >
                {copied ? <Check size={15} color="#16a34a" /> : <Copy size={15} />}
                {copied ? 'Copied to Clipboard!' : 'Copy Summary'}
              </button>

              <button
                className="btn-secondary"
                onClick={handleAskAboutSummary}
                style={{ padding: '0.45rem 0.85rem', fontSize: '0.825rem', display: 'inline-flex', alignItems: 'center', gap: '0.4rem', color: '#2563eb', borderColor: '#bfdbfe' }}
                title="Jump to chat with this document selected"
              >
                <MessageSquare size={15} /> Ask AI in Chat
              </button>

              <button
                className="btn-secondary"
                onClick={() => handleGenerateSummary(currentDocId)}
                style={{ padding: '0.45rem 0.85rem', fontSize: '0.825rem', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
                title="Re-run summarization"
              >
                <RotateCcw size={15} /> Regenerate
              </button>
            </div>
          </div>

          {/* 1. Document Overview Section */}
          <div className="card summary-section-card">
            <div className="summary-section-header">
              <div className="summary-section-icon blue">
                <BookOpen size={20} />
              </div>
              <h4 className="summary-section-title">Document Overview</h4>
            </div>
            <div className="summary-overview-body">
              <MarkdownRenderer content={summaryData.overview} />
            </div>
          </div>

          {/* 2. Main Topics Section */}
          {summaryData.main_topics && summaryData.main_topics.length > 0 && (
            <div className="card summary-section-card">
              <div className="summary-section-header">
                <div className="summary-section-icon purple">
                  <Layers size={20} />
                </div>
                <h4 className="summary-section-title">Main Topics</h4>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                {summaryData.main_topics.map((topic, idx) => (
                  <div key={idx} className="topic-item-card">
                    <span className="topic-number-circle">{idx + 1}</span>
                    <span style={{ fontSize: '0.9rem', color: '#334155', fontWeight: 500 }}>{topic}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 3. Important Points Section */}
          {summaryData.important_points && summaryData.important_points.length > 0 && (
            <div className="card summary-section-card">
              <div className="summary-section-header">
                <div className="summary-section-icon green">
                  <ListOrdered size={20} />
                </div>
                <h4 className="summary-section-title">Important Points</h4>
              </div>
              <ul className="points-bullet-list">
                {summaryData.important_points.map((pt, idx) => (
                  <li key={idx} className="points-bullet-item">
                    <span className="bullet-dot"></span>
                    <span>{pt}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* 4. Key Takeaways Section */}
          {summaryData.key_takeaways && summaryData.key_takeaways.length > 0 && (
            <div className="card summary-section-card takeaways-highlight-card">
              <div className="summary-section-header">
                <div className="summary-section-icon orange">
                  <Lightbulb size={20} />
                </div>
                <h4 className="summary-section-title">Key Takeaways for Students</h4>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.85rem' }}>
                {summaryData.key_takeaways.map((takeaway, idx) => (
                  <div key={idx} className="takeaway-box">
                    <Bookmark size={16} color="#d97706" style={{ flexShrink: 0, marginTop: '0.2rem' }} />
                    <span style={{ fontSize: '0.875rem', color: '#451a03', fontWeight: 500 }}>
                      {takeaway}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 5. Chapter-wise Summaries (when identifiable) */}
          {summaryData.chapters && summaryData.chapters.length > 0 && (
            <div className="card summary-section-card">
              <div className="summary-section-header">
                <div className="summary-section-icon teal">
                  <FileText size={20} />
                </div>
                <div>
                  <h4 className="summary-section-title">Chapter-wise Summaries</h4>
                  <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
                    {summaryData.chapters.length} chapters identified
                  </span>
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {summaryData.chapters.map((chap, cIdx) => (
                  <div key={cIdx} className="chapter-item-box">
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                      <strong style={{ fontSize: '0.9rem', color: '#0f172a' }}>{chap.chapter_title}</strong>
                      <span className="badge badge-blue" style={{ fontSize: '0.75rem' }}>{chap.page_range}</span>
                    </div>
                    <p style={{ fontSize: '0.85rem', color: '#475569', margin: 0, lineHeight: 1.5 }}>
                      {chap.summary}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 6. Source Page References */}
          {summaryData.source_pages && summaryData.source_pages.length > 0 && (
            <div className="card" style={{ padding: '1rem 1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <BookOpen size={16} color="#2563eb" />
                <span style={{ fontWeight: 600, fontSize: '0.85rem', color: '#334155' }}>
                  Source Page Citations:
                </span>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {summaryData.source_pages.map((pg) => (
                  <span key={pg} className="source-tag-pill" style={{ margin: 0 }}>
                    Page {pg}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
