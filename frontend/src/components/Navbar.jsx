import React from 'react';
import { Menu, Wifi, WifiOff } from 'lucide-react';

export default function Navbar({ activeTab, setIsSidebarOpen, isBackendHealthy }) {
  const titles = {
    dashboard: 'Study Dashboard',
    documents: 'My Study Documents',
    summarizer: 'AI Study Summarizer',
    chat: 'AI Study Chat',
    questions: 'Important Questions Generator',
  };

  const subtitles = {
    dashboard: 'Welcome to your private study assistant. Review your notes and start learning.',
    documents: 'Upload, manage, and index your PDF lectures, textbooks, and notes.',
    summarizer: 'Generate complete, structured AI study guides with Overview, Topics, Points & Key Takeaways.',
    chat: 'Ask questions and get instant, factual answers based on your uploaded documents.',
    questions: 'Generate exam-focused Short-Answer, Long-Answer, and Conceptual questions covering your entire document.',
  };

  return (
    <header className="top-navbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <button
          className="mobile-menu-btn"
          onClick={() => setIsSidebarOpen(true)}
          aria-label="Open sidebar"
        >
          <Menu size={24} />
        </button>
        <div>
          <h2 className="navbar-page-title">{titles[activeTab] || 'Offline Study AI'}</h2>
          <p style={{ fontSize: '0.8rem', color: '#64748b', display: 'none' }} className="d-sm-block">
            {subtitles[activeTab]}
          </p>
        </div>
      </div>

      <div className="navbar-right">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
          {isBackendHealthy ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#059669', fontWeight: 600 }}>
              <Wifi size={16} /> Backend Connected
            </span>
          ) : (
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#dc2626', fontWeight: 600 }}>
              <WifiOff size={16} /> Backend Offline
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
