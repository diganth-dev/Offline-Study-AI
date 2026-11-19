import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  BookOpen,
  LogIn,
  UserPlus,
  KeyRound,
  Mail,
  Lock,
  User,
  AlertCircle,
  CheckCircle2,
  ArrowLeft,
  ShieldCheck,
  Server,
  ServerOff,
  Wifi,
  WifiOff,
} from 'lucide-react';
import {
  registerUser,
  loginUser,
  loginWithGoogle,
  requestForgotPassword,
  resetUserPassword,
} from '../api';

export default function AuthScreen({
  onAuthSuccess,
  isBackendHealthy = true,
  isOnline = true,
}) {
  const [mode, setMode] = useState('login'); // 'login' | 'register' | 'forgot'
  const [usernameOrEmail, setUsernameOrEmail] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // Forgot password state
  const [forgotEmail, setForgotEmail] = useState('');
  const [resetTokenOrCode, setResetTokenOrCode] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const [resetStep, setResetStep] = useState(1); // 1: request, 2: enter code & new pw

  // Google OAuth & Account Chooser state
  const [googleClientId, setGoogleClientId] = useState(() => {
    return (
      import.meta.env.VITE_GOOGLE_CLIENT_ID ||
      (typeof localStorage !== 'undefined' ? localStorage.getItem('study_ai_google_client_id') : '') ||
      ''
    );
  });
  const [showGoogleChooser, setShowGoogleChooser] = useState(false);
  const [googleAccounts, setGoogleAccounts] = useState(() => {
    try {
      const stored = localStorage.getItem('study_ai_google_accounts');
      if (stored) {
        return JSON.parse(stored);
      }
    } catch {
      // ignore
    }
    return [
      {
        name: 'Deepan K',
        email: 'deeek@gmail.com',
        avatarColor: '#EA4335',
      },
      {
        name: 'Student Demo',
        email: 'student@gmail.com',
        avatarColor: '#4285F4',
      },
    ];
  });
  const [showAddAccountInput, setShowAddAccountInput] = useState(false);
  const [customGoogleEmail, setCustomGoogleEmail] = useState('');
  const [customGoogleName, setCustomGoogleName] = useState('');
  const [showLiveOAuthConfig, setShowLiveOAuthConfig] = useState(false);
  const [inputClientId, setInputClientId] = useState('');
  const [isGsiLoaded, setIsGsiLoaded] = useState(false);

  const googleNativeBtnRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  const resetAlerts = () => {
    setError(null);
    setSuccessMessage(null);
  };

  const switchMode = (newMode) => {
    setMode(newMode);
    resetAlerts();
    if (newMode === 'forgot') {
      setResetStep(1);
    }
  };

  // ----------------------------------------------------
  // Google Sign-In & Account Handler
  // ----------------------------------------------------
  const handleSelectGoogleProfile = useCallback(
    async ({ email, name, google_id, credential }) => {
      setIsLoading(true);
      resetAlerts();
      try {
        const cleanEmail = (email || '').trim();
        const displayName = (name || cleanEmail.split('@')[0]).trim();
        const res = await loginWithGoogle({
          email: cleanEmail,
          name: displayName,
          google_id: google_id || `g_${cleanEmail.replace(/[^a-zA-Z0-9]/g, '_')}`,
          credential,
        });

        // Save account into remembered accounts list in localStorage
        setGoogleAccounts((prev) => {
          const exists = prev.some((a) => a.email.toLowerCase() === cleanEmail.toLowerCase());
          const updated = exists
            ? prev
            : [
                ...prev,
                {
                  name: displayName,
                  email: cleanEmail,
                  avatarColor: '#10b981',
                },
              ];
          try {
            localStorage.setItem('study_ai_google_accounts', JSON.stringify(updated));
          } catch {
            // ignore
          }
          return updated;
        });

        setShowGoogleChooser(false);
        setSuccessMessage(`Signed in with Google as ${res.user.username}!`);
        setTimeout(() => {
          onAuthSuccess(res.user);
        }, 350);
      } catch (err) {
        setError(err.message || 'Google authentication failed.');
      } finally {
        setIsLoading(false);
      }
    },
    [onAuthSuccess]
  );

  // ----------------------------------------------------
  // Google Identity Services (GSI) Initialization
  // ----------------------------------------------------
  const handleGoogleCredentialResponse = useCallback(
    async (response) => {
      if (!response || !response.credential) return;

      setIsLoading(true);
      resetAlerts();
      try {
        const res = await loginWithGoogle({ credential: response.credential });
        setSuccessMessage(`Signed in with Google as ${res.user.username}!`);
        setTimeout(() => {
          onAuthSuccess(res.user);
        }, 350);
      } catch (err) {
        setError(err.message || 'Google authentication failed.');
      } finally {
        setIsLoading(false);
      }
    },
    [onAuthSuccess]
  );

  useEffect(() => {
    // Check if Google script has loaded
    const checkGsi = () => {
      if (typeof window !== 'undefined' && window.google?.accounts?.id) {
        setIsGsiLoaded(true);
        return true;
      }
      return false;
    };

    if (checkGsi()) {
      if (googleClientId && window.google?.accounts?.id) {
        try {
          window.google.accounts.id.initialize({
            client_id: googleClientId,
            callback: handleGoogleCredentialResponse,
            auto_select: false,
            cancel_on_tap_outside: true,
          });

          // Render Google's native button if container is mounted
          if (googleNativeBtnRef.current) {
            googleNativeBtnRef.current.innerHTML = '';
            window.google.accounts.id.renderButton(googleNativeBtnRef.current, {
              theme: 'outline',
              size: 'large',
              text: 'continue_with',
              shape: 'rectangular',
              logo_alignment: 'center',
              width: 380,
            });
          }

          // Trigger One Tap account prompt on page load
          window.google.accounts.id.prompt();
        } catch (e) {
          console.warn('Google Identity initialization notice:', e);
        }
      }
    } else {
      const interval = setInterval(() => {
        if (checkGsi()) {
          clearInterval(interval);
        }
      }, 300);
      return () => clearInterval(interval);
    }
  }, [googleClientId, handleGoogleCredentialResponse]);

  // Trigger Google Account Chooser
  const handleContinueWithGoogle = () => {
    resetAlerts();

    // If client ID is present and GIS OAuth2 is available, trigger native Google popup
    if (googleClientId && typeof window !== 'undefined' && window.google?.accounts?.oauth2) {
      try {
        const tokenClient = window.google.accounts.oauth2.initTokenClient({
          client_id: googleClientId,
          scope: 'email profile openid',
          callback: async (tokenResponse) => {
            if (tokenResponse && tokenResponse.access_token) {
              setIsLoading(true);
              try {
                const profileRes = await fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
                  headers: { Authorization: `Bearer ${tokenResponse.access_token}` },
                });
                const profile = await profileRes.json();
                await handleSelectGoogleProfile({
                  email: profile.email,
                  name: profile.name,
                  google_id: profile.sub,
                  credential: tokenResponse.access_token,
                });
              } catch (err) {
                setError(err.message || 'Google profile retrieval failed.');
              } finally {
                setIsLoading(false);
              }
            } else if (tokenResponse && tokenResponse.error) {
              setError(`Google sign-in cancelled: ${tokenResponse.error}`);
            }
          },
        });

        // Opens official Google popup with account selector
        tokenClient.requestAccessToken({ prompt: 'select_account' });
        return;
      } catch (err) {
        console.warn('Google OAuth token client exception, falling back to account chooser:', err);
      }
    }

    // Automatically display Google Account Chooser
    setShowAddAccountInput(false);
    setShowGoogleChooser(true);
  };

  const handleCustomGoogleAccountSubmit = (e) => {
    e.preventDefault();
    if (!customGoogleEmail.trim()) {
      setError('Please enter your Google account email.');
      return;
    }
    handleSelectGoogleProfile({
      email: customGoogleEmail.trim(),
      name: customGoogleName.trim() || undefined,
    });
  };

  const handleSaveGoogleClientId = (e) => {
    e.preventDefault();
    const id = inputClientId.trim();
    if (!id) {
      setError('Please provide a valid Google Client ID.');
      return;
    }
    try {
      localStorage.setItem('study_ai_google_client_id', id);
    } catch {
      // ignore
    }
    setGoogleClientId(id);
    setShowLiveOAuthConfig(false);
    setShowGoogleChooser(false);
    setSuccessMessage('Google Client ID linked! Launching official Google account chooser...');
    setTimeout(() => {
      // Trigger Google popup immediately
      if (typeof window !== 'undefined' && window.google?.accounts?.oauth2) {
        try {
          const client = window.google.accounts.oauth2.initTokenClient({
            client_id: id,
            scope: 'email profile openid',
            callback: async (tokRes) => {
              if (tokRes && tokRes.access_token) {
                const profileRes = await fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
                  headers: { Authorization: `Bearer ${tokRes.access_token}` },
                });
                const profile = await profileRes.json();
                await handleSelectGoogleProfile({
                  email: profile.email,
                  name: profile.name,
                  google_id: profile.sub,
                  credential: tokRes.access_token,
                });
              }
            },
          });
          client.requestAccessToken({ prompt: 'select_account' });
        } catch {
          // ignore
        }
      }
    }, 400);
  };

  // ----------------------------------------------------
  // Login & Registration
  // ----------------------------------------------------
  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    resetAlerts();

    const identifier = usernameOrEmail.trim();
    if (!identifier || !password) {
      setError('Please enter your username/email and password.');
      return;
    }

    setIsLoading(true);
    try {
      const res = await loginUser(identifier, password);
      setSuccessMessage(`Welcome back, ${res.user.username}!`);
      setTimeout(() => {
        onAuthSuccess(res.user);
      }, 350);
    } catch (err) {
      setError(err.message || 'Login failed. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegisterSubmit = async (e) => {
    e.preventDefault();
    resetAlerts();

    const u = username.trim();
    const em = email.trim();

    if (!u || !em || !password) {
      setError('Please fill in all registration fields.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters long.');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match. Please re-enter.');
      return;
    }

    setIsLoading(true);
    try {
      const res = await registerUser(u, em, password);
      setSuccessMessage(`Account created successfully! Welcome, ${res.user.username}.`);
      setTimeout(() => {
        onAuthSuccess(res.user);
      }, 450);
    } catch (err) {
      setError(err.message || 'Registration failed. Please choose another username/email.');
    } finally {
      setIsLoading(false);
    }
  };

  // ----------------------------------------------------
  // Forgot Password
  // ----------------------------------------------------
  const handleForgotRequest = async (e) => {
    e.preventDefault();
    resetAlerts();

    if (!forgotEmail.trim()) {
      setError('Please enter your account email address.');
      return;
    }

    setIsLoading(true);
    try {
      const res = await requestForgotPassword(forgotEmail.trim());
      setSuccessMessage(res.message);
      // Clean reset code state: user must check their email (Requirement 14)
      setResetTokenOrCode('');
      setResetStep(2);
    } catch (err) {
      setError(err.message || 'Could not send verification email. Please check your email address.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetSubmit = async (e) => {
    e.preventDefault();
    resetAlerts();

    const code = resetTokenOrCode.trim();
    if (!code) {
      setError('Please enter the 6-digit verification code from your email.');
      return;
    }
    if (!newPassword || newPassword.length < 6) {
      setError('New password must be at least 6 characters long.');
      return;
    }
    if (newPassword !== confirmNewPassword) {
      setError('Passwords do not match. Please re-enter.');
      return;
    }

    setIsLoading(true);
    try {
      const res = await resetUserPassword(code, newPassword, forgotEmail.trim());
      setSuccessMessage(res.message || 'Password successfully updated! You can now sign in.');
      setTimeout(() => {
        setUsernameOrEmail(forgotEmail);
        switchMode('login');
      }, 1800);
    } catch (err) {
      setError(err.message || 'Failed to reset password. Please check your verification code.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-screen-layout">
      {/* Background Decor Elements */}
      <div className="auth-decor-circle-1" />
      <div className="auth-decor-circle-2" />

      <div className="auth-screen-card">
        {/* Brand Header */}
        <div className="auth-screen-brand">
          <div className="auth-brand-icon-box">
            <BookOpen size={28} />
          </div>
          <h1 className="auth-brand-title">Offline Study AI</h1>
          <p className="auth-brand-subtitle">
            Personal AI Learning Assistant with Isolated ChromaDB Study Vaults
          </p>
        </div>

        {/* Backend Unreachable Alert */}
        {!isBackendHealthy && (
          <div className="auth-screen-backend-warning">
            <ServerOff size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontWeight: 600 }}>Backend Connection Required</div>
              <div style={{ fontSize: '0.78rem', marginTop: '2px' }}>
                Start your local Python FastAPI server:
                <code style={{ display: 'block', marginTop: '4px', backgroundColor: 'rgba(0,0,0,0.06)', padding: '2px 4px', borderRadius: '4px' }}>
                  python -m uvicorn backend.main:app --reload
                </code>
              </div>
            </div>
          </div>
        )}

        {/* Mode Switcher Tabs (Only for login and register) */}
        {mode !== 'forgot' && (
          <div className="auth-screen-tabs">
            <button
              type="button"
              className={`auth-screen-tab-btn ${mode === 'login' ? 'active' : ''}`}
              onClick={() => switchMode('login')}
            >
              <LogIn size={16} />
              <span>Sign In</span>
            </button>
            <button
              type="button"
              className={`auth-screen-tab-btn ${mode === 'register' ? 'active' : ''}`}
              onClick={() => switchMode('register')}
            >
              <UserPlus size={16} />
              <span>Create Account</span>
            </button>
          </div>
        )}

        {/* Forgot Password Header */}
        {mode === 'forgot' && (
          <div className="auth-forgot-header">
            <button
              type="button"
              className="auth-back-link"
              onClick={() => switchMode('login')}
            >
              <ArrowLeft size={16} />
              <span>Back to Sign In</span>
            </button>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a', marginTop: '0.5rem' }}>
              Reset Your Password
            </h2>
            <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
              {resetStep === 1
                ? 'Enter your registered email address to receive your verification code.'
                : 'Enter your 6-digit code and choose a new password.'}
            </p>
          </div>
        )}

        {/* Alerts */}
        {error && (
          <div className="auth-screen-alert error">
            <AlertCircle size={18} style={{ flexShrink: 0 }} />
            <span>{error}</span>
          </div>
        )}

        {successMessage && (
          <div className="auth-screen-alert success">
            <CheckCircle2 size={18} style={{ flexShrink: 0 }} />
            <span>{successMessage}</span>
          </div>
        )}

        {/* Google Account Chooser Modal (Displays authentic Google Accounts selection) */}
        {showGoogleChooser && (
          <div className="google-chooser-overlay" onClick={() => setShowGoogleChooser(false)}>
            <div className="google-chooser-card" onClick={(e) => e.stopPropagation()}>
              {/* Header */}
              <div className="google-chooser-header">
                <svg viewBox="0 0 24 24" width="28" height="28" className="google-icon">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                </svg>
                <h2 className="google-chooser-title">Choose an account</h2>
                <p className="google-chooser-subtitle">to continue to <strong>Offline Study AI</strong></p>
              </div>

              {/* Account List */}
              <div className="google-accounts-list">
                {googleAccounts.map((account, index) => (
                  <button
                    key={index}
                    type="button"
                    className="google-account-item"
                    onClick={() => handleSelectGoogleProfile({ email: account.email, name: account.name })}
                    disabled={isLoading}
                    title={`Sign in as ${account.email}`}
                  >
                    <div
                      className="google-account-avatar"
                      style={{ backgroundColor: account.avatarColor || '#2563eb' }}
                    >
                      {account.name.charAt(0).toUpperCase()}
                    </div>
                    <div className="google-account-details">
                      <div className="google-account-name">{account.name}</div>
                      <div className="google-account-email">{account.email}</div>
                    </div>
                  </button>
                ))}

                {/* Option to use another Google account */}
                {!showAddAccountInput ? (
                  <button
                    type="button"
                    className="google-account-item"
                    onClick={() => setShowAddAccountInput(true)}
                  >
                    <div className="google-account-avatar add-icon">
                      <UserPlus size={18} />
                    </div>
                    <div className="google-account-details">
                      <div className="google-account-name">Use another Google account</div>
                    </div>
                  </button>
                ) : (
                  <form onSubmit={handleCustomGoogleAccountSubmit} className="google-add-account-form">
                    <input
                      type="email"
                      placeholder="Enter Google email (e.g. user@gmail.com)"
                      value={customGoogleEmail}
                      onChange={(e) => setCustomGoogleEmail(e.target.value)}
                      required
                      autoFocus
                    />
                    <input
                      type="text"
                      placeholder="Display Name (optional)"
                      value={customGoogleName}
                      onChange={(e) => setCustomGoogleName(e.target.value)}
                    />
                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.25rem' }}>
                      <button
                        type="button"
                        className="auth-cancel-btn"
                        onClick={() => setShowAddAccountInput(false)}
                      >
                        Back
                      </button>
                      <button
                        type="submit"
                        className="auth-screen-submit-btn"
                        style={{ flex: 1, margin: 0 }}
                        disabled={isLoading}
                      >
                        {isLoading ? 'Signing in...' : 'Sign In'}
                      </button>
                    </div>
                  </form>
                )}
              </div>

              {/* Link Official Google Cloud OAuth Client ID (collapsible) */}
              <div className="google-chooser-cloud-config">
                <button
                  type="button"
                  className="google-cloud-toggle-btn"
                  onClick={() => setShowLiveOAuthConfig(!showLiveOAuthConfig)}
                >
                  <span>{showLiveOAuthConfig ? '▲ Hide Google Cloud Settings' : '⚙️ Link live Google Cloud OAuth Client ID'}</span>
                </button>
                {showLiveOAuthConfig && (
                  <form onSubmit={handleSaveGoogleClientId} style={{ marginTop: '0.65rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      To trigger Google's native browser popup window, enter your Web Client ID from Google Cloud Console:
                    </div>
                    <input
                      type="text"
                      placeholder="123456789-xxx.apps.googleusercontent.com"
                      value={inputClientId}
                      onChange={(e) => setInputClientId(e.target.value)}
                      style={{ padding: '0.5rem 0.75rem', fontSize: '0.8rem', border: '1px solid #cbd5e1', borderRadius: '6px' }}
                    />
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button type="submit" className="auth-screen-submit-btn" style={{ margin: 0, padding: '0.45rem', flex: 1, fontSize: '0.8rem' }}>
                        Save &amp; Open Live Google Popup
                      </button>
                    </div>
                  </form>
                )}
              </div>

              {/* Footer Note */}
              <div className="google-chooser-footer">
                To continue, Google will share your name, email address, and profile picture with Offline Study AI.
              </div>

              <div style={{ textAlign: 'center', marginTop: '0.75rem' }}>
                <button
                  type="button"
                  className="auth-cancel-btn"
                  style={{ width: '100%' }}
                  onClick={() => setShowGoogleChooser(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Google Continue Button (for login and register) */}
        {mode !== 'forgot' && (
          <div style={{ marginBottom: '1.25rem' }}>
            {/* Native Google Identity Services Button Container (when Client ID is loaded) */}
            {googleClientId && isGsiLoaded ? (
              <div
                ref={googleNativeBtnRef}
                style={{ width: '100%', display: 'flex', justifyContent: 'center', marginBottom: '0.5rem' }}
              />
            ) : (
              <button
                type="button"
                className="google-signin-btn"
                onClick={handleContinueWithGoogle}
                disabled={isLoading || !isBackendHealthy}
                title="Sign in with your Google account"
              >
                <svg viewBox="0 0 24 24" width="18" height="18" className="google-icon">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                </svg>
                <span>Continue with Google</span>
              </button>
            )}

            <div className="auth-screen-divider">
              <span>or continue with email</span>
            </div>
          </div>
        )}

        {/* 1. SIGN IN FORM */}
        {mode === 'login' && (
          <form onSubmit={handleLoginSubmit} className="auth-screen-form">
            <div className="auth-screen-field">
              <label htmlFor="screen-login-id">Username or Email</label>
              <div className="auth-screen-input-wrapper">
                <User size={18} className="auth-screen-input-icon" />
                <input
                  id="screen-login-id"
                  type="text"
                  placeholder="Enter your username or email"
                  value={usernameOrEmail}
                  onChange={(e) => setUsernameOrEmail(e.target.value)}
                  autoComplete="username"
                  required
                />
              </div>
            </div>

            <div className="auth-screen-field">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label htmlFor="screen-login-pw">Password</label>
                <button
                  type="button"
                  className="auth-inline-link"
                  onClick={() => switchMode('forgot')}
                >
                  Forgot password?
                </button>
              </div>
              <div className="auth-screen-input-wrapper">
                <Lock size={18} className="auth-screen-input-icon" />
                <input
                  id="screen-login-pw"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              className="auth-screen-submit-btn"
              disabled={isLoading || !isBackendHealthy}
            >
              <LogIn size={18} />
              <span>{isLoading ? 'Signing in...' : 'Sign In to Dashboard'}</span>
            </button>
          </form>
        )}

        {/* 2. CREATE ACCOUNT FORM */}
        {mode === 'register' && (
          <form onSubmit={handleRegisterSubmit} className="auth-screen-form">
            <div className="auth-screen-field">
              <label htmlFor="screen-reg-username">Username</label>
              <div className="auth-screen-input-wrapper">
                <User size={18} className="auth-screen-input-icon" />
                <input
                  id="screen-reg-username"
                  type="text"
                  placeholder="Choose a username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  required
                />
              </div>
              <span className="auth-screen-hint">3-30 letters, numbers, hyphens or underscores</span>
            </div>

            <div className="auth-screen-field">
              <label htmlFor="screen-reg-email">Email Address</label>
              <div className="auth-screen-input-wrapper">
                <Mail size={18} className="auth-screen-input-icon" />
                <input
                  id="screen-reg-email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                  required
                />
              </div>
            </div>

            <div className="auth-screen-field">
              <label htmlFor="screen-reg-password">Password</label>
              <div className="auth-screen-input-wrapper">
                <Lock size={18} className="auth-screen-input-icon" />
                <input
                  id="screen-reg-password"
                  type="password"
                  placeholder="At least 6 characters"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                />
              </div>
            </div>

            <div className="auth-screen-field">
              <label htmlFor="screen-reg-confirm">Confirm Password</label>
              <div className="auth-screen-input-wrapper">
                <KeyRound size={18} className="auth-screen-input-icon" />
                <input
                  id="screen-reg-confirm"
                  type="password"
                  placeholder="Re-enter password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              className="auth-screen-submit-btn"
              disabled={isLoading || !isBackendHealthy}
            >
              <UserPlus size={18} />
              <span>{isLoading ? 'Creating account...' : 'Create Account & Access AI'}</span>
            </button>
          </form>
        )}

        {/* 3. FORGOT PASSWORD FLOW */}
        {mode === 'forgot' && resetStep === 1 && (
          <form onSubmit={handleForgotRequest} className="auth-screen-form">
            <div className="auth-screen-field">
              <label htmlFor="forgot-email">Registered Account Email</label>
              <div className="auth-screen-input-wrapper">
                <Mail size={18} className="auth-screen-input-icon" />
                <input
                  id="forgot-email"
                  type="email"
                  placeholder="Enter your registered email (e.g. name@gmail.com)"
                  value={forgotEmail}
                  onChange={(e) => setForgotEmail(e.target.value)}
                  required
                />
              </div>
              <span className="auth-screen-hint">
                A 6-digit verification code will be sent to your Gmail / email inbox.
              </span>
            </div>

            <button
              type="submit"
              className="auth-screen-submit-btn"
              disabled={isLoading || !isBackendHealthy}
            >
              <KeyRound size={18} />
              <span>{isLoading ? 'Sending verification code...' : 'Send Verification Code to Email'}</span>
            </button>
          </form>
        )}

        {mode === 'forgot' && resetStep === 2 && (
          <form onSubmit={handleResetSubmit} className="auth-screen-form">
            <div className="auth-screen-field">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label htmlFor="reset-code">6-Digit Verification Code</label>
                <button
                  type="button"
                  className="auth-inline-link"
                  onClick={() => {
                    setResetStep(1);
                    resetAlerts();
                  }}
                >
                  Resend code
                </button>
              </div>
              <div className="auth-screen-input-wrapper">
                <KeyRound size={18} className="auth-screen-input-icon" />
                <input
                  id="reset-code"
                  type="text"
                  placeholder="Enter 6-digit code from email (e.g. 583921)"
                  maxLength={6}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  value={resetTokenOrCode}
                  onChange={(e) => setResetTokenOrCode(e.target.value.replace(/[^0-9]/g, ''))}
                  required
                  autoFocus
                />
              </div>
              <span className="auth-screen-hint">
                Check <strong>{forgotEmail}</strong>. Valid for 15 minutes. Single-use only.
              </span>
            </div>

            <div className="auth-screen-field">
              <label htmlFor="reset-new-pw">New Password</label>
              <div className="auth-screen-input-wrapper">
                <Lock size={18} className="auth-screen-input-icon" />
                <input
                  id="reset-new-pw"
                  type="password"
                  placeholder="Minimum 6 characters"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                />
              </div>
            </div>

            <div className="auth-screen-field">
              <label htmlFor="reset-confirm-pw">Confirm New Password</label>
              <div className="auth-screen-input-wrapper">
                <Lock size={18} className="auth-screen-input-icon" />
                <input
                  id="reset-confirm-pw"
                  type="password"
                  placeholder="Re-enter new password"
                  value={confirmNewPassword}
                  onChange={(e) => setConfirmNewPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              className="auth-screen-submit-btn"
              disabled={isLoading || !isBackendHealthy}
            >
              <CheckCircle2 size={18} />
              <span>{isLoading ? 'Updating password...' : 'Update Password & Sign In'}</span>
            </button>
          </form>
        )}

        {/* Footer Security Badges */}
        <div className="auth-screen-footer">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', color: '#10b981', fontSize: '0.78rem', fontWeight: 600 }}>
            <ShieldCheck size={16} />
            <span>Strict ChromaDB Multi-User Data Isolation</span>
          </div>

          <div className="auth-status-bar">
            <span className={`status-pill ${isOnline ? 'status-pill-online' : 'status-pill-offline'}`}>
              {isOnline ? <Wifi size={13} /> : <WifiOff size={13} />}
              <span>{isOnline ? 'PWA Online' : 'PWA Offline Cache'}</span>
            </span>

            <span className={`status-pill ${isBackendHealthy ? 'status-pill-backend-ok' : 'status-pill-backend-err'}`}>
              {isBackendHealthy ? <Server size={13} /> : <ServerOff size={13} />}
              <span>{isBackendHealthy ? 'AI Backend Connected' : 'Backend Offline'}</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
