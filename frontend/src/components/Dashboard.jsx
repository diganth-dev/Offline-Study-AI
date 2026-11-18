import React from 'react';
import { FileText, MessageSquare, UploadCloud, BookOpen, Layers, ArrowRight, Lightbulb, CheckCircle2, FileQuestion } from 'lucide-react';

export default function Dashboard({
  documents = [],
  queryCount = 0,
  setActiveTab,
  onSelectDocumentForChat,
  isBackendHealthy,
}) {
  const totalDocs = documents.length;
  const totalChunks = documents.reduce((sum, doc) => sum + (doc.total_chunks || 0), 0);
  const totalPages = documents.reduce((sum, doc) => sum + (doc.total_pages || 0), 0);

  return (
    <div className="page-container">
      {/* Welcome Banner */}
      <div className="welcome-banner">
        <div className="welcome-text">
          <h1>Welcome to Offline Study AI! 🎓</h1>
          <p>
            Your private, offline learning companion. Upload your class notes, textbook chapters, or lecture slides,
            and study with an AI assistant that answers questions strictly based on your documents.
          </p>
        </div>
        <div style={{ display: 'none' }} className="d-md-block">
          <BookOpen size={64} style={{ opacity: 0.15 }} />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="stats-grid">
        <div className="card stat-card">
          <div className="stat-icon blue">
            <FileText size={26} />
          </div>
          <div>
            <div className="stat-label">Uploaded Documents</div>
            <div className="stat-value">{totalDocs}</div>
          </div>
        </div>

        <div className="card stat-card">
          <div className="stat-icon green">
            <Layers size={26} />
          </div>
          <div>
            <div className="stat-label">Study Chunks / Pages</div>
            <div className="stat-value">{totalChunks} <span style={{ fontSize: '1rem', fontWeight: 500, color: '#64748b' }}>({totalPages} pgs)</span></div>
          </div>
        </div>

        <div className="card stat-card">
          <div className="stat-icon purple">
            <MessageSquare size={26} />
          </div>
          <div>
            <div className="stat-label">Questions Asked</div>
            <div className="stat-value">{queryCount}</div>
          </div>
        </div>
      </div>

      {/* Split section: Recent Docs & Quick Actions */}
      <div className="dashboard-split">
        {/* Left: Recent Documents */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a' }}>Recent Study Materials</h3>
            <button
              onClick={() => setActiveTab('documents')}
              style={{ background: 'transparent', color: '#2563eb', fontWeight: 600, fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
            >
              View all <ArrowRight size={16} />
            </button>
          </div>

          {documents.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2.5rem 1rem', color: '#64748b' }}>
              <FileText size={44} style={{ margin: '0 auto 0.75rem auto', color: '#94a3b8' }} />
              <p style={{ fontWeight: 600, color: '#334155', marginBottom: '0.25rem' }}>No study documents yet</p>
              <p style={{ fontSize: '0.875rem', marginBottom: '1.25rem' }}>Upload your first lecture notes or PDF to start studying.</p>
              <button
                className="btn-primary"
                onClick={() => setActiveTab('documents')}
              >
                <UploadCloud size={18} /> Upload PDF Notes
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {documents.slice(0, 4).map((doc) => (
                <div
                  key={doc.filename}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.85rem 1rem',
                    backgroundColor: '#f8fafc',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: 0 }}>
                    <div style={{ width: '36px', height: '36px', borderRadius: '8px', backgroundColor: '#eff6ff', color: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <FileText size={20} />
                    </div>
                    <div style={{ minWidth: 0 }}>
                      <p style={{ fontWeight: 600, fontSize: '0.925rem', color: '#0f172a', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {doc.filename}
                      </p>
                      <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
                        {doc.total_pages} {doc.total_pages === 1 ? 'page' : 'pages'} • {doc.total_chunks} indexed chunks
                      </p>
                    </div>
                  </div>

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
                    title={`Chat with AI about ${doc.filename}`}
                  >
                    Chat with AI
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Quick Actions & Tips */}
        <div className="quick-actions-box">
          <div className="card">
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a', marginBottom: '1rem' }}>
              Quick Actions
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <button
                className="action-card-btn"
                onClick={() => setActiveTab('documents')}
              >
                <div className="action-left">
                  <div style={{ width: '40px', height: '40px', borderRadius: '8px', backgroundColor: '#eff6ff', color: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <UploadCloud size={20} />
                  </div>
                  <div>
                    <div className="action-btn-title">Upload New Document</div>
                    <div className="action-btn-desc">Add PDF notes or textbook chapters</div>
                  </div>
                </div>
                <ArrowRight size={18} color="#94a3b8" />
              </button>

              <button
                className="action-card-btn"
                onClick={() => setActiveTab('chat')}
              >
                <div className="action-left">
                  <div style={{ width: '40px', height: '40px', borderRadius: '8px', backgroundColor: '#f3e8ff', color: '#9333ea', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <MessageSquare size={20} />
                  </div>
                  <div>
                    <div className="action-btn-title">Ask Study Questions</div>
                    <div className="action-btn-desc">Chat with AI grounded in your materials</div>
                  </div>
                </div>
                <ArrowRight size={18} color="#94a3b8" />
              </button>
              <button
                className="action-card-btn"
                onClick={() => setActiveTab('questions')}
              >
                <div className="action-left">
                  <div style={{ width: '40px', height: '40px', borderRadius: '8px', backgroundColor: '#fff7ed', color: '#ea580c', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <FileQuestion size={20} />
                  </div>
                  <div>
                    <div className="action-btn-title">Important Questions</div>
                    <div className="action-btn-desc">Exam-focused Short, Long &amp; Conceptual questions</div>
                  </div>
                </div>
                <ArrowRight size={18} color="#94a3b8" />
              </button>
            </div>
          </div>

          {/* Student Study Tips Card */}
          <div className="card" style={{ backgroundColor: '#fffbeb', borderColor: '#fef3c7' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: '#b45309', fontWeight: 700 }}>
              <Lightbulb size={20} />
              <span>Study Tips for Students</span>
            </div>
            <ul style={{ paddingLeft: '1.25rem', fontSize: '0.85rem', color: '#78350f', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <li>Ask for summaries: <em>"Summarize Chapter 1 in 3 bullet points."</em></li>
              <li>Ask for comparisons: <em>"What is the difference between supervised and unsupervised learning?"</em></li>
              <li>Look at the citations below the answer to see the exact page number in your textbook!</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
