import React from 'react';
import {
  LayoutDashboard,
  FileText,
  MessageSquare,
  ShieldCheck,
  BookOpen,
  Sparkles,
  FileQuestion,
  X,
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, isOpen, setIsOpen, isBackendHealthy }) {
  const mainNavItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'documents', label: 'My Documents', icon: FileText },
  ];

  const aiStudyTools = [
    { id: 'chat', label: 'AI Study Chat', icon: MessageSquare },
    { id: 'summarizer', label: 'AI Summarizer', icon: Sparkles },
    { id: 'questions', label: 'Important Questions', icon: FileQuestion },
  ];

  return (
    <>
      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="brand-icon-box">
            <BookOpen size={22} />
          </div>
          <div style={{ flex: 1 }}>
            <div className="brand-title">Offline Study AI</div>
            <div className="brand-subtitle">Personal Study Assistant</div>
          </div>
          {isOpen && (
            <button
              className="mobile-close-btn"
              onClick={() => setIsOpen(false)}
              style={{ background: 'transparent', color: '#94a3b8', padding: '0.25rem' }}
            >
              <X size={20} />
            </button>
          )}
        </div>

        <nav className="sidebar-nav">
          {mainNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab(item.id);
                  setIsOpen(false);
                }}
              >
                <Icon size={20} />
                <span>{item.label}</span>
              </button>
            );
          })}

          <div className="sidebar-section-title">AI Study Tools</div>

          {aiStudyTools.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab(item.id);
                  setIsOpen(false);
                }}
              >
                <Icon size={20} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>System Status:</span>
            <span className={`status-badge ${isBackendHealthy ? 'online' : 'offline'}`}>
              <span className="status-dot"></span>
              {isBackendHealthy ? 'Online' : 'Offline'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.4rem', color: '#94a3b8', fontSize: '0.75rem' }}>
            <ShieldCheck size={16} color="#34d399" />
            <span>100% Private &amp; Local</span>
          </div>
        </div>
      </aside>

      {/* Backdrop for mobile drawer */}
      {isOpen && (
        <div
          onClick={() => setIsOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            zIndex: 35,
          }}
        />
      )}
    </>
  );
}
