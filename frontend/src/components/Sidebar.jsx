import React from 'react';
import {
  LayoutDashboard,
  FileText,
  MessageSquare,
  ShieldCheck,
  BookOpen,
  Sparkles,
  FileQuestion,
  Download,
  X,
  User,
  LogIn,
  LogOut,
} from 'lucide-react';

export default function Sidebar({
  activeTab,
  setActiveTab,
  isOpen,
  setIsOpen,
  isBackendHealthy,
  isOnline = true,
  canInstall = false,
  onInstallClick,
  currentUser = null,
  onOpenAuth,
  onLogout,
}) {
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
              aria-label="Close sidebar"
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
          {/* Account Section */}
          {currentUser ? (
            <div className="sidebar-account-box">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', minWidth: 0 }}>
                <div className="sidebar-avatar">
                  <User size={16} />
                </div>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ fontWeight: 600, color: '#f8fafc', fontSize: '0.85rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {currentUser.username}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {currentUser.email || 'Private Account'}
                  </div>
                </div>
              </div>
              <button
                className="sidebar-logout-btn"
                onClick={onLogout}
                title="Log out of this account"
              >
                <LogOut size={13} />
                <span>Sign Out</span>
              </button>
            </div>
          ) : (
            <button
              className="sidebar-login-btn"
              onClick={() => {
                onOpenAuth();
                setIsOpen(false);
              }}
              title="Sign in to save documents to your private library"
            >
              <LogIn size={15} />
              <span>Sign In / Register</span>
            </button>
          )}

          {canInstall && (
            <button
              className="sidebar-install-btn"
              onClick={onInstallClick}
              title="Install Offline Study AI on your device"
            >
              <Download size={16} />
              <span>Install App</span>
            </button>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: canInstall ? '0.4rem' : '0' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>App Shell:</span>
              <span className={`status-badge ${isOnline ? 'online' : 'cached'}`}>
                <span className="status-dot"></span>
                {isOnline ? 'Online' : 'PWA Cached'}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>AI Backend:</span>
              <span className={`status-badge ${isBackendHealthy ? 'online' : 'offline'}`}>
                <span className="status-dot"></span>
                {isBackendHealthy ? 'Connected' : 'Offline'}
              </span>
            </div>
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
