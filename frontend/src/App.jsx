import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import Documents from './components/Documents';
import Chat from './components/Chat';
import Summarizer from './components/Summarizer';
import ImportantQuestions from './components/ImportantQuestions';
import { checkBackendHealth, fetchIndexedDocuments } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isBackendHealthy, setIsBackendHealthy] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [queryCount, setQueryCount] = useState(() => {
    try {
      const saved = localStorage.getItem('study_ai_query_count');
      return saved ? parseInt(saved, 10) : 0;
    } catch {
      return 0;
    }
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

  // Fetch indexed documents list
  const refreshDocuments = useCallback(async () => {
    try {
      const data = await fetchIndexedDocuments();
      const docs = data.documents || [];
      setDocuments(docs);

      // Clean up selected document if it was deleted
      setSelectedDocument((prev) => {
        if (!prev) return null;
        const exists = docs.some((d) => d.document_id === prev.document_id);
        return exists ? prev : null;
      });
    } catch (err) {
      console.warn('Could not load documents list:', err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    checkHealth();
    refreshDocuments();

    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth, refreshDocuments]);

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

  return (
    <div className="app-layout">
      {/* Dark Navy Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
        isBackendHealthy={isBackendHealthy}
      />

      {/* Main Wrapper */}
      <div className="main-wrapper">
        <Navbar
          activeTab={activeTab}
          setIsSidebarOpen={setIsSidebarOpen}
          isBackendHealthy={isBackendHealthy}
        />

        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          {activeTab === 'dashboard' && (
            <Dashboard
              documents={documents}
              queryCount={queryCount}
              setActiveTab={setActiveTab}
              onSelectDocumentForChat={handleSelectDocumentForChat}
              isBackendHealthy={isBackendHealthy}
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
            />
          )}

          {activeTab === 'summarizer' && (
            <Summarizer
              documents={documents}
              selectedDocument={selectedDocument}
              onSelectDocument={setSelectedDocument}
              setActiveTab={setActiveTab}
              onSelectDocumentForChat={handleSelectDocumentForChat}
            />
          )}

          {activeTab === 'questions' && (
            <ImportantQuestions
              documents={documents}
              selectedDocument={selectedDocument}
              onSelectDocument={setSelectedDocument}
              setActiveTab={setActiveTab}
              onSelectDocumentForChat={handleSelectDocumentForChat}
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
            />
          )}
        </main>
      </div>
    </div>
  );
}
