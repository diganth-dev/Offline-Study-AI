import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  RefreshCw,
  BookOpen,
  Bot,
  User,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  FileText,
  Loader2,
  Sparkles,
  Globe,
} from 'lucide-react';
import { askAiQuestion } from '../api';
import MarkdownRenderer from './MarkdownRenderer';

export default function Chat({
  documents = [],
  selectedDocument = null,
  onSelectDocument,
  onClearSelectedDocument,
  onQuestionAsked,
  setActiveTab,
}) {
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('study_ai_chat_history');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [inputQuestion, setInputQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [expandedContextIndex, setExpandedContextIndex] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Persist chat history to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('study_ai_chat_history', JSON.stringify(messages));
    } catch {
      // Ignore storage errors
    }
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleClearChat = () => {
    if (messages.length === 0) return;
    if (window.confirm('Start a new study conversation? This will clear the current chat history.')) {
      setMessages([]);
      setErrorMessage(null);
      localStorage.removeItem('study_ai_chat_history');
    }
  };

  const handleSend = async (questionText = inputQuestion) => {
    const q = (questionText || '').trim();
    if (!q || isLoading) return;

    if (documents.length === 0) {
      setErrorMessage('You have not uploaded any study documents yet. Please upload a PDF in "My Documents" first so the AI can answer your questions.');
      return;
    }

    setErrorMessage(null);
    setInputQuestion('');

    const docId = selectedDocument ? selectedDocument.document_id : null;
    const docName = selectedDocument ? selectedDocument.filename : null;

    // Add user message with document reference if scoped
    const userMsg = {
      id: Date.now().toString(),
      sender: 'user',
      text: q,
      documentName: docName,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await askAiQuestion(q, 4, docId);

      const aiMsg = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: response.answer,
        sources: response.sources || [],
        contextUsed: response.context_used || [],
        documentId: response.document_id || docId,
        documentName: docName,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, aiMsg]);
      if (onQuestionAsked) {
        onQuestionAsked();
      }
    } catch (err) {
      setErrorMessage(err.message || 'Failed to get an answer from AI.');
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 100);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const starterQuestions = selectedDocument
    ? [
        `What are the main concepts in ${selectedDocument.filename}?`,
        'Summarize this document in simple terms.',
        'How many pages are there?',
        'List the primary definitions and takeaways.',
      ]
    : [
        'What are the key concepts explained in my uploaded notes?',
        'Summarize the main takeaways from the document.',
        'How many total pages and documents do I have?',
        'Explain the most important ideas in simple terms.',
      ];

  return (
    <div className="chat-page-container">
      {/* Chat Header Card */}
      <div className="chat-header-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ width: '38px', height: '38px', borderRadius: '10px', backgroundColor: '#eff6ff', color: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Bot size={22} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '1rem', color: '#0f172a' }}>AI Study Assistant</div>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
              {selectedDocument ? (
                <span>Focused Study Mode • 1 Document Selected</span>
              ) : (
                <span>General Study Mode • Searching across {documents.length} document{documents.length === 1 ? '' : 's'}</span>
              )}
            </div>
          </div>
        </div>

        <button
          className="btn-secondary"
          onClick={handleClearChat}
          disabled={messages.length === 0 || isLoading}
          title="Start fresh conversation"
          style={{ padding: '0.45rem 0.85rem', fontSize: '0.825rem' }}
        >
          <RefreshCw size={15} /> New Chat
        </button>
      </div>

      {/* Scope Bar: Shows selected document or general chat with switch option */}
      <div className={`doc-scope-bar ${selectedDocument ? 'focused' : 'general'}`}>
        <div className="doc-scope-left">
          {selectedDocument ? (
            <>
              <span style={{ fontWeight: 600, color: '#1e3a8a', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                <FileText size={16} /> Selected PDF:
              </span>
              <span className="doc-scope-badge" title={selectedDocument.filename}>
                {selectedDocument.filename} ({selectedDocument.total_pages} {selectedDocument.total_pages === 1 ? 'pg' : 'pgs'})
              </span>
            </>
          ) : (
            <>
              <span style={{ fontWeight: 600, color: '#475569', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                <Globe size={16} /> Mode:
              </span>
              <span>General Chat (Searches across all {documents.length} uploaded PDFs)</span>
            </>
          )}
        </div>

        <div className="doc-scope-actions">
          {/* Document Switcher Dropdown */}
          {documents.length > 0 && (
            <select
              className="doc-select-dropdown"
              value={selectedDocument ? selectedDocument.document_id : ''}
              onChange={(e) => {
                const targetId = e.target.value;
                if (!targetId) {
                  if (onClearSelectedDocument) onClearSelectedDocument();
                } else {
                  const targetDoc = documents.find((d) => d.document_id === targetId);
                  if (targetDoc && onSelectDocument) {
                    onSelectDocument(targetDoc);
                  }
                }
              }}
              title="Switch document or select general chat"
            >
              <option value="">🌐 All Documents (General Chat)</option>
              {documents.map((d) => (
                <option key={d.document_id} value={d.document_id}>
                  📄 {d.filename} ({d.total_pages} {d.total_pages === 1 ? 'page' : 'pages'})
                </option>
              ))}
            </select>
          )}

          {selectedDocument && onClearSelectedDocument && (
            <button
              type="button"
              className="btn-switch-general"
              onClick={onClearSelectedDocument}
              title="Return to general chat across all documents"
            >
              <Globe size={13} /> Return to General Chat
            </button>
          )}
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="chat-messages-scroll">
        {documents.length === 0 && (
          <div className="alert alert-info" style={{ margin: '1rem 0' }}>
            <AlertCircle size={20} style={{ flexShrink: 0 }} />
            <div style={{ flex: 1 }}>
              <p style={{ fontWeight: 600 }}>No study documents loaded yet</p>
              <p style={{ fontSize: '0.85rem' }}>
                To ask questions, upload your lecture notes or PDF textbook first in <strong>My Documents</strong>.
              </p>
            </div>
            <button
              className="btn-primary"
              style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem' }}
              onClick={() => setActiveTab('documents')}
            >
              Upload PDF
            </button>
          </div>
        )}

        {messages.length === 0 ? (
          <div className="chat-welcome-state">
            <div className="chat-welcome-icon">
              {selectedDocument ? <FileText size={28} /> : <Sparkles size={28} />}
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.5rem' }}>
              {selectedDocument
                ? `Studying "${selectedDocument.filename}"`
                : 'How can I help you study today?'}
            </h3>
            <p style={{ fontSize: '0.9rem', maxWidth: '480px', margin: '0 auto' }}>
              {selectedDocument
                ? `Ask questions specifically about this document (${selectedDocument.total_pages} pages, ${selectedDocument.total_chunks} chunks). Answers are grounded only in this file.`
                : 'Ask questions about your uploaded course materials. Answers are synthesized strictly from your notes with exact page references.'}
            </p>

            <div className="starter-chips-grid">
              {starterQuestions.map((q, idx) => (
                <button
                  key={idx}
                  className="starter-chip"
                  onClick={() => handleSend(q)}
                  disabled={isLoading}
                >
                  "{q}"
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div key={msg.id} className={`chat-row ${msg.sender}`}>
              {msg.sender === 'ai' && (
                <div className="chat-avatar ai">
                  <Bot size={18} />
                </div>
              )}

              <div style={{ maxWidth: '82%', display: 'flex', flexDirection: 'column', alignItems: msg.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                {msg.documentName && (
                  <span style={{ fontSize: '0.72rem', color: '#64748b', marginBottom: '0.2rem', padding: '0 0.4rem' }}>
                    📄 {msg.documentName}
                  </span>
                )}

                <div className={`chat-bubble ${msg.sender}`}>
                  {msg.sender === 'ai' ? (
                    <MarkdownRenderer content={msg.text} />
                  ) : (
                    msg.text
                  )}

                  {/* Sources pill area for AI answers */}
                  {msg.sender === 'ai' && msg.sources && msg.sources.length > 0 && (
                    <div className="source-references-card">
                      <div style={{ fontWeight: 700, color: '#475569', fontSize: '0.775rem', marginBottom: '0.35rem' }}>
                        Source Citations:
                      </div>
                      <div>
                        {msg.sources.map((src, sIdx) => (
                          <span key={sIdx} className="source-tag-pill">
                            <BookOpen size={12} />
                            {src.filename} (Page {src.page_number})
                          </span>
                        ))}
                      </div>

                      {/* Expandable context snippets for curious students */}
                      {msg.contextUsed && msg.contextUsed.length > 0 && (
                        <div style={{ marginTop: '0.65rem' }}>
                          <button
                            type="button"
                            onClick={() =>
                              setExpandedContextIndex(expandedContextIndex === index ? null : index)
                            }
                            style={{
                              background: 'transparent',
                              color: '#64748b',
                              fontSize: '0.75rem',
                              fontWeight: 600,
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.25rem',
                              padding: '0.2rem 0',
                            }}
                          >
                            {expandedContextIndex === index ? (
                              <>Hide retrieved document excerpts <ChevronUp size={14} /></>
                            ) : (
                              <>View retrieved document excerpts ({msg.contextUsed.length}) <ChevronDown size={14} /></>
                            )}
                          </button>

                          {expandedContextIndex === index && (
                            <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                              {msg.contextUsed.map((chunk, cIdx) => (
                                <div
                                  key={cIdx}
                                  style={{
                                    backgroundColor: '#ffffff',
                                    border: '1px solid #e2e8f0',
                                    borderRadius: '6px',
                                    padding: '0.6rem 0.75rem',
                                    fontSize: '0.775rem',
                                    color: '#334155',
                                  }}
                                >
                                  <div style={{ fontWeight: 700, color: '#1d4ed8', marginBottom: '0.25rem' }}>
                                    Excerpt #{cIdx + 1} — {chunk.filename} (Page {chunk.page_number})
                                  </div>
                                  <div style={{ fontStyle: 'italic', whiteSpace: 'pre-wrap', maxHeight: '120px', overflowY: 'auto' }}>
                                    "{chunk.text}"
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <span style={{ fontSize: '0.725rem', color: '#94a3b8', marginTop: '0.3rem', padding: '0 0.4rem' }}>
                  {msg.timestamp}
                </span>
              </div>

              {msg.sender === 'user' && (
                <div className="chat-avatar user">
                  <User size={18} />
                </div>
              )}
            </div>
          ))
        )}

        {/* Loading Indicator */}
        {isLoading && (
          <div className="chat-row ai">
            <div className="chat-avatar ai">
              <Bot size={18} />
            </div>
            <div className="chat-bubble ai" style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', color: '#475569' }}>
              <Loader2 size={18} className="spinner" style={{ color: '#2563eb' }} />
              <span style={{ fontSize: '0.875rem' }}>
                {selectedDocument
                  ? `AI is studying "${selectedDocument.filename}" and synthesizing an answer...`
                  : 'AI is studying your notes and synthesizing an answer with Ollama...'}
              </span>
            </div>
          </div>
        )}

        {/* Error Banner in Chat */}
        {errorMessage && (
          <div className="alert alert-danger" style={{ margin: '0.5rem 0' }}>
            <AlertCircle size={20} style={{ flexShrink: 0 }} />
            <div style={{ flex: 1, fontSize: '0.875rem' }}>
              <strong>Error:</strong> {errorMessage}
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              style={{ background: 'transparent', color: 'inherit', fontWeight: 'bold' }}
            >
              ×
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form Bar */}
      <div className="chat-input-bar-container">
        <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="chat-input-form">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder={
              documents.length === 0
                ? 'Upload a document first to start asking study questions...'
                : selectedDocument
                ? `Ask a question about ${selectedDocument.filename}... (Press Enter to send, Shift+Enter for new line)`
                : 'Ask a question across all study documents... (Press Enter to send, Shift+Enter for new line)'
            }
            value={inputQuestion}
            onChange={(e) => setInputQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading || documents.length === 0}
            rows={1}
          />

          <button
            type="submit"
            className="send-btn"
            disabled={!inputQuestion.trim() || isLoading || documents.length === 0}
            title="Send Question"
          >
            {isLoading ? <Loader2 size={20} className="spinner" /> : <Send size={20} />}
          </button>
        </form>
      </div>
    </div>
  );
}
