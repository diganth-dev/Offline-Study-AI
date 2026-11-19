import React from 'react';
import { Menu, Wifi, WifiOff, Server, ServerOff, Download, User, LogIn, LogOut } from 'lucide-react';

export default function Navbar({
  activeTab,
  setIsSidebarOpen,
  isBackendHealthy,
  isOnline = true,
  canInstall = false,
  onInstallClick,
  currentUser = null,
  onOpenAuth,
  onLogout,
}) {
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
        {/* PWA Install Button */}
        {canInstall && (
          <button
            className="navbar-install-btn"
            onClick={onInstallClick}
            title="Install Offline Study AI as a standalone application on your device"
          >
            <Download size={15} />
            <span className="install-btn-text">Install App</span>
          </button>
        )}

        {/* Dual Status Indicators: Network & Local Backend */}
        <div className="status-indicators-group">
          {/* 1. Network / Internet Status */}
          {isOnline ? (
            <span
              className="status-pill status-pill-online"
              title="Internet connected. Frontend application shell synced."
            >
              <Wifi size={14} />
              <span className="pill-text">PWA Online</span>
            </span>
          ) : (
            <span
              className="status-pill status-pill-offline"
              title="Internet disconnected. Frontend shell loaded from local PWA cache."
            >
              <WifiOff size={14} />
              <span className="pill-text">PWA Offline</span>
            </span>
          )}

          {/* 2. AI Backend Status */}
          {isBackendHealthy ? (
            <span
              className="status-pill status-pill-backend-ok"
              title="Local FastAPI backend (http://127.0.0.1:8000) is connected."
            >
              <Server size={14} />
              <span className="pill-text">Backend Connected</span>
            </span>
          ) : (
            <span
              className="status-pill status-pill-backend-err"
              title="Local FastAPI backend is unreachable at http://127.0.0.1:8000."
            >
              <ServerOff size={14} />
              <span className="pill-text">Backend Offline</span>
            </span>
          )}
        </div>

        {/* User Authentication Badge / Sign In Button */}
        <div className="navbar-auth-section">
          {currentUser ? (
            <div className="navbar-user-badge" title={`Signed in as ${currentUser.username} (${currentUser.email || ''})`}>
              <div className="navbar-user-info">
                <User size={15} className="user-icon" />
                <span className="navbar-username">{currentUser.username}</span>
              </div>
              <button
                className="navbar-logout-btn"
                onClick={onLogout}
                title="Log out"
                aria-label="Log out"
              >
                <LogOut size={14} />
                <span className="logout-text">Logout</span>
              </button>
            </div>
          ) : (
            <button
              className="navbar-login-btn"
              onClick={onOpenAuth}
              title="Sign in or create account to isolate your documents"
            >
              <LogIn size={15} />
              <span>Sign In</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
