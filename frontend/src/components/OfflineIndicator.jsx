import React, { useState } from 'react';
import { WifiOff, ServerOff, Download, AlertTriangle, CheckCircle2, X } from 'lucide-react';

export default function OfflineIndicator({
  isOnline,
  wasOffline,
  isBackendHealthy,
  deferredPrompt,
  isInstalled,
  onInstallClick,
}) {
  const [dismissBackendWarning, setDismissBackendWarning] = useState(false);

  return (
    <div className="pwa-status-container">
      {/* 1. Browser Offline Banner (High Priority) */}
      {!isOnline && (
        <div className="offline-banner network-offline" role="alert">
          <div className="banner-content">
            <WifiOff size={18} className="banner-icon" />
            <div>
              <strong>Offline Mode:</strong> Internet disconnected. The Offline Study AI shell is active from local PWA cache.
            </div>
          </div>
        </div>
      )}

      {/* 2. Restored Connection Toast */}
      {wasOffline && isOnline && (
        <div className="offline-banner network-restored" role="status">
          <div className="banner-content">
            <CheckCircle2 size={18} className="banner-icon text-success" />
            <div>
              <strong>Back Online:</strong> Internet connection has been restored.
            </div>
          </div>
        </div>
      )}

      {/* 3. Backend Offline Warning Banner */}
      {!isBackendHealthy && !dismissBackendWarning && (
        <div className="offline-banner backend-offline" role="alert">
          <div className="banner-content">
            <ServerOff size={18} className="banner-icon text-danger" />
            <div style={{ flex: 1 }}>
              <strong>AI Backend Server Unreachable:</strong> Cannot connect to local FastAPI backend (<code>http://127.0.0.1:8000</code>).
              <div style={{ fontSize: '0.8rem', marginTop: '0.2rem', color: '#64748b' }}>
                The frontend shell is working offline. To enable AI questions, summarization, and PDF indexing, run:{' '}
                <code style={{ background: '#e2e8f0', padding: '1px 5px', borderRadius: '4px' }}>python -m uvicorn backend.main:app --reload</code>
              </div>
            </div>
            <button
              onClick={() => setDismissBackendWarning(true)}
              className="banner-close-btn"
              title="Dismiss warning"
              aria-label="Dismiss warning"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      {/* 4. Desktop / Android Install Prompt Bar (Optional top banner if prompt available) */}
      {deferredPrompt && !isInstalled && (
        <div className="offline-banner install-prompt-banner">
          <div className="banner-content">
            <Download size={18} className="banner-icon text-primary" />
            <div style={{ flex: 1 }}>
              <strong>Install Offline Study AI:</strong> Install this app on your device for standalone offline study and fast access.
            </div>
            <button onClick={onInstallClick} className="btn-install-banner">
              <Download size={15} /> Install App
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
