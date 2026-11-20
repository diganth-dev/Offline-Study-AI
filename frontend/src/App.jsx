import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import Documents from './components/Documents';
import Chat from './components/Chat';
import Summarizer from './components/Summarizer';
import ImportantQuestions from './components/ImportantQuestions';
import OfflineIndicator from './components/OfflineIndicator';
import AuthScreen from './components/AuthScreen';
import {
  checkBackendHealth,
  fetchIndexedDocuments,
  getStoredUser,
  getStoredToken,
  fetchCurrentUser,
  logoutUser,
  isTokenExpired,
  clearAuthSession,
} from './api';
import { BookOpen } from 'lucide-react';

export default function App() {
  // Support PWA shortcuts via ?tab=... parameter
  const [activeTab, setActiveTab] = useState(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const tab = params.get('tab');
      if (tab && ['dashboard', 'documents', 'chat', 'summarizer', 'questions'].includes(tab)) {
        return tab;
      }
    } catch {
      // Fallback
    }
    return 'dashboard';
  });

  // User Authentication State - restore securely from localStorage subject to token expiry
  const [currentUser, setCurrentUser] = useState(() => {
    const token = getStoredToken();
    if (!token || isTokenExpired(token)) {
      if (token && isTokenExpired(token)) {
        clearAuthSession();
      }
      return null;
    }
    return getStoredUser();
  });
  const [isCheckingAuth, setIsCheckingAuth] = useState(() => {
    const token = getStoredToken();
    return Boolean(token && !isTokenExpired(token));
  });

  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isBackendHealthy, setIsBackendHealthy] = useState(false);
  const [isOnline, setIsOnline] = useState(() => (typeof navigator !== 'undefined' ? navigator.onLine : true));
  const [wasOffline, setWasOffline] = useState(false);
  const [documents, setDocuments] = useState(() => {
    try {
      const user = getStoredUser();
      if (user && user.user_id) {
        const cached = localStorage.getItem(`study_ai_cached_docs_${user.user_id}`);
        return cached ? JSON.parse(cached) : [];
      }
    } catch {
      // ignore
    }
    return [];
  });
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [queryCount, setQueryCount] = useState(() => {
    try {
      const saved = localStorage.getItem('study_ai_query_count');
      return saved ? parseInt(saved, 10) : 0;
    } catch {
      return 0;
    }
  });

  // PWA Install prompt state
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [isInstalled, setIsInstalled] = useState(() => {
    if (typeof window === 'undefined') return false;
    return (
      window.matchMedia('(display-mode: standalone)').matches ||
      window.navigator.standalone === true
    );
  });

  // Verify backend health
  const checkHealth = useCallback(async () => {
    try {
      await checkBackendHealth();
      setIsBackendHealthy(true);
    } catch {
      setIsBackendHealthy(false);
    }
  }, []);

  // Fetch indexed documents list scoped to authenticated user
  const refreshDocuments = useCallback(async () => {
    const token = getStoredToken();
    if (!token || isTokenExpired(token)) {
      setDocuments([]);
      setSelectedDocument(null);
      return;
    }

    try {
      const data = await fetchIndexedDocuments();
      const docs = data.documents || [];
      setDocuments(docs);

      // Cache documents locally for offline viewing
      try {
        const user = getStoredUser();
        if (user && user.user_id) {
          localStorage.setItem(`study_ai_cached_docs_${user.user_id}`, JSON.stringify(docs));
        }
      } catch {
        // ignore
      }

      // Clean up selected document if it was deleted
      setSelectedDocument((prev) => {
        if (!prev) return null;
        const exists = docs.some((d) => d.document_id === prev.document_id);
        return exists ? prev : null;
      });
    } catch (err) {
      console.warn('Could not load documents list:', err);
      // Fallback: restore documents from offline cache if available
      try {
        const user = getStoredUser();
        if (user && user.user_id) {
          const cached = localStorage.getItem(`study_ai_cached_docs_${user.user_id}`);
          if (cached) {
            setDocuments(JSON.parse(cached));
          }
        }
      } catch {
        // ignore
      }
    }
  }, []);

  // Handle Logout - clears session and redirects directly to Login/Register screen
  const handleLogout = useCallback(() => {
    try {
      const user = getStoredUser();
      if (user && user.user_id) {
        localStorage.removeItem(`study_ai_cached_docs_${user.user_id}`);
      }
    } catch {
      // ignore
    }
    logoutUser();
    setCurrentUser(null);
    setDocuments([]);
    setSelectedDocument(null);
    setActiveTab('dashboard');
    setIsCheckingAuth(false);
  }, []);

  // Handle Authentication Success - redirects to Dashboard
  const handleAuthSuccess = (user) => {
    setCurrentUser(user);
    setActiveTab('dashboard');
    refreshDocuments();
  };

  // Session validation and recovery on mount
  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      setIsCheckingAuth(false);
      return;
    }

    // Check client-side token expiration
    if (isTokenExpired(token)) {
      handleLogout();
      setIsCheckingAuth(false);
      return;
    }

    setIsCheckingAuth(true);

    fetchCurrentUser()
      .then((profile) => {
        if (profile) {
          setCurrentUser(profile);
          setIsBackendHealthy(true);
          refreshDocuments();
        } else {
          handleLogout();
        }
      })
      .catch((err) => {
        // Clear token ONLY when the backend explicitly returns an authentication failure (HTTP 401 or 403)
        if (err && (err.status === 401 || err.status === 403)) {
          handleLogout();
        } else {
          // Network error or backend offline: preserve the existing login session
          setIsBackendHealthy(false);
          const cachedUser = getStoredUser();
          if (cachedUser) {
            setCurrentUser(cachedUser);
          }
        }
      })
      .finally(() => {
        setIsCheckingAuth(false);
      });
  }, [handleLogout, refreshDocuments]);

  // Global listener for unauthorized (expired or invalid token) responses
  useEffect(() => {
    const handleUnauthorized = () => {
      handleLogout();
    };

    window.addEventListener('study-ai:unauthorized', handleUnauthorized);
    return () => window.removeEventListener('study-ai:unauthorized', handleUnauthorized);
  }, [handleLogout]);

  // Monitor online / offline network state
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setWasOffline(true);
      checkHealth();
      const token = getStoredToken();
      if (token) {
        if (isTokenExpired(token)) {
          handleLogout();
        } else {
          fetchCurrentUser()
            .then((profile) => {
              if (profile) setCurrentUser(profile);
            })
            .catch((err) => {
              if (err && (err.status === 401 || err.status === 403)) {
                handleLogout();
              }
            });
        }
      }
      refreshDocuments();
      setTimeout(() => setWasOffline(false), 4000);
    };

    const handleOffline = () => {
      setIsOnline(false);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [checkHealth, refreshDocuments, handleLogout]);

  // Monitor PWA installation readiness
  useEffect(() => {
    const handleBeforeInstall = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
    };

    const handleAppInstalled = () => {
      setIsInstalled(true);
      setDeferredPrompt(null);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstall);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstall);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  // Trigger PWA Installation prompt
  const handleInstallApp = async () => {
    if (!deferredPrompt) {
      alert(
        'To install Offline Study AI on this device, open your browser menu (⋮ or Share) and choose "Install Offline Study AI" or "Add to Home Screen".'
      );
      return;
    }
    try {
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      if (outcome === 'accepted') {
        setIsInstalled(true);
        setDeferredPrompt(null);
      }
    } catch (err) {
      console.warn('Error during PWA installation:', err);
    }
  };

  // Initial load
  useEffect(() => {
    checkHealth();
    if (currentUser) {
      refreshDocuments();
    }

    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth, refreshDocuments, currentUser]);

  const handleSelectDocumentForChat = (doc) => {
    setSelectedDocument(doc);
    setActiveTab('chat');
  };

  const handleSelectDocumentForSummarize = (doc) => {
    setSelectedDocument(doc);
    setActiveTab('summarizer');
  };

  const handleSelectDocumentForQuestions = (doc) => {
    setSelectedDocument(doc);
    setActiveTab('questions');
  };

  const handleClearSelectedDocument = () => {
    setSelectedDocument(null);
  };

  const handleQuestionAsked = () => {
    setQueryCount((prev) => {
      const newCount = prev + 1;
      try {
        localStorage.setItem('study_ai_query_count', newCount.toString());
      } catch {
        // ignore
      }
      return newCount;
    });
  };

  const canInstall = Boolean(deferredPrompt && !isInstalled);

  // 1. Initial Authentication Check Screen
  if (isCheckingAuth) {
    return (
      <div className="auth-splash-screen">
        <div className="auth-brand-icon-box" style={{ width: 56, height: 56, borderRadius: 14 }}>
          <BookOpen size={30} />
        </div>
        <div style={{ marginTop: '1.25rem', fontWeight: 700, fontSize: '1.25rem', color: '#0f172a' }}>
          Offline Study AI
        </div>
        <div style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '0.35rem' }}>
          Restoring secure study session...
        </div>
      </div>
    );
  }

  // 2. Mandatory Authentication Gate: Neither Dashboard nor Sidebar is shown to unauthenticated users
  if (!currentUser) {
    return (
      <AuthScreen
        onAuthSuccess={handleAuthSuccess}
        isBackendHealthy={isBackendHealthy}
        isOnline={isOnline}
      />
    );
  }

  // 3. Fully Authenticated Application Shell
  return (
    <div className="app-layout">
      {/* Dark Navy Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
        isBackendHealthy={isBackendHealthy}
        isOnline={isOnline}
        canInstall={canInstall}
        onInstallClick={handleInstallApp}
        currentUser={currentUser}
        onLogout={handleLogout}
      />

      {/* Main Wrapper */}
      <div className="main-wrapper">
        <Navbar
          activeTab={activeTab}
          setIsSidebarOpen={setIsSidebarOpen}
          isBackendHealthy={isBackendHealthy}
          isOnline={isOnline}
          canInstall={canInstall}
          onInstallClick={handleInstallApp}
          currentUser={currentUser}
          onLogout={handleLogout}
        />

        {/* Global Offline & Installation Indicators */}
        <OfflineIndicator
          isOnline={isOnline}
          wasOffline={wasOffline}
          isBackendHealthy={isBackendHealthy}
          deferredPrompt={deferredPrompt}
          isInstalled={isInstalled}
          onInstallClick={handleInstallApp}
        />

        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          {activeTab === 'dashboard' && (
            <Dashboard
              documents={documents}
              queryCount={queryCount}
              setActiveTab={setActiveTab}
              onSelectDocumentForChat={handleSelectDocumentForChat}
              isBackendHealthy={isBackendHealthy}
              currentUser={currentUser}
            />
          )}

          {activeTab === 'documents' && (
            <Documents
              documents={documents}
              onRefreshDocuments={refreshDocuments}
              setActiveTab={setActiveTab}
              onSelectDocumentForChat={handleSelectDocumentForChat}
              onSelectDocumentForSummarize={handleSelectDocumentForSummarize}
              onSelectDocumentForQuestions={handleSelectDocumentForQuestions}
              isBackendHealthy={isBackendHealthy}
              isOnline={isOnline}
              currentUser={currentUser}
            />
          )}

          {activeTab === 'summarizer' && (
            <Summarizer
              documents={documents}
              selectedDocument={selectedDocument}
              onSelectDocument={setSelectedDocument}
              setActiveTab={setActiveTab}
              onSelectDocumentForChat={handleSelectDocumentForChat}
              isBackendHealthy={isBackendHealthy}
              isOnline={isOnline}
              currentUser={currentUser}
            />
          )}

          {activeTab === 'questions' && (
            <ImportantQuestions
              documents={documents}
              selectedDocument={selectedDocument}
              onSelectDocument={setSelectedDocument}
              setActiveTab={setActiveTab}
              onSelectDocumentForChat={handleSelectDocumentForChat}
              isBackendHealthy={isBackendHealthy}
              isOnline={isOnline}
              currentUser={currentUser}
            />
          )}

          {activeTab === 'chat' && (
            <Chat
              documents={documents}
              selectedDocument={selectedDocument}
              onSelectDocument={setSelectedDocument}
              onClearSelectedDocument={handleClearSelectedDocument}
              onQuestionAsked={handleQuestionAsked}
              setActiveTab={setActiveTab}
              isBackendHealthy={isBackendHealthy}
              isOnline={isOnline}
              currentUser={currentUser}
            />
          )}
        </main>
      </div>
    </div>
  );
}
