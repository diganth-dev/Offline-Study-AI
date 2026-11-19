import React, { useState } from 'react';
import { LogIn, UserPlus, X, AlertCircle, CheckCircle2, User, Mail, Lock, KeyRound } from 'lucide-react';
import { registerUser, loginUser } from '../api';

export default function AuthModal({
  isOpen,
  onClose,
  onAuthSuccess,
  initialMode = 'login',
}) {
  const [mode, setMode] = useState(initialMode); // 'login' or 'register'
  const [usernameOrEmail, setUsernameOrEmail] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState(null);

  if (!isOpen) return null;

  const resetForm = () => {
    setError(null);
    setSuccessMessage(null);
  };

  const switchMode = (newMode) => {
    setMode(newMode);
    resetForm();
  };

  const handleFillDemo = (name = 'student_demo') => {
    if (mode === 'login') {
      setUsernameOrEmail(name);
      setPassword('study123');
    } else {
      setUsername(name);
      setEmail(`${name}@study.local`);
      setPassword('study123');
      setConfirmPassword('study123');
    }
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (mode === 'register') {
      if (!username.trim() || !email.trim() || !password) {
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
        const result = await registerUser(username.trim(), email.trim(), password);
        setSuccessMessage(`Account created! Welcome, ${result.user.username}.`);
        setTimeout(() => {
          onAuthSuccess(result.user);
          onClose();
        }, 500);
      } catch (err) {
        setError(err.message || 'Registration failed. Please try again.');
      } finally {
        setIsLoading(false);
      }
    } else {
      // Login mode
      if (!usernameOrEmail.trim() || !password) {
        setError('Please enter your username/email and password.');
        return;
      }

      setIsLoading(true);
      try {
        const result = await loginUser(usernameOrEmail.trim(), password);
        setSuccessMessage(`Welcome back, ${result.user.username}!`);
        setTimeout(() => {
          onAuthSuccess(result.user);
          onClose();
        }, 400);
      } catch (err) {
        setError(err.message || 'Login failed. Check your credentials.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="auth-modal-overlay" onClick={onClose}>
      <div
        className="auth-modal-container"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-modal-title"
      >
        {/* Modal Header */}
        <div className="auth-modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div className="auth-modal-badge">
              {mode === 'login' ? <LogIn size={20} /> : <UserPlus size={20} />}
            </div>
            <div>
              <h3 id="auth-modal-title" className="auth-modal-title">
                {mode === 'login' ? 'Sign In to Study AI' : 'Create an Account'}
              </h3>
              <p className="auth-modal-subtitle">
                {mode === 'login'
                  ? 'Access your private study documents and isolated AI chats'
                  : 'Get your own isolated ChromaDB library and encrypted workspace'}
              </p>
            </div>
          </div>
          <button
            className="auth-modal-close-btn"
            onClick={onClose}
            aria-label="Close dialog"
          >
            <X size={20} />
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="auth-tab-group">
          <button
            type="button"
            className={`auth-tab-btn ${mode === 'login' ? 'active' : ''}`}
            onClick={() => switchMode('login')}
          >
            <LogIn size={15} />
            <span>Sign In</span>
          </button>
          <button
            type="button"
            className={`auth-tab-btn ${mode === 'register' ? 'active' : ''}`}
            onClick={() => switchMode('register')}
          >
            <UserPlus size={15} />
            <span>Create Account</span>
          </button>
        </div>

        {/* Alerts */}
        {error && (
          <div className="auth-alert error">
            <AlertCircle size={18} style={{ flexShrink: 0 }} />
            <span>{error}</span>
          </div>
        )}

        {successMessage && (
          <div className="auth-alert success">
            <CheckCircle2 size={18} style={{ flexShrink: 0 }} />
            <span>{successMessage}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          {mode === 'login' ? (
            <>
              <div className="auth-form-field">
                <label htmlFor="auth-login-identifier">Username or Email</label>
                <div className="auth-input-wrapper">
                  <User size={18} className="auth-input-icon" />
                  <input
                    id="auth-login-identifier"
                    type="text"
                    placeholder="Enter your username or email"
                    value={usernameOrEmail}
                    onChange={(e) => setUsernameOrEmail(e.target.value)}
                    autoComplete="username"
                    required
                  />
                </div>
              </div>

              <div className="auth-form-field">
                <label htmlFor="auth-login-password">Password</label>
                <div className="auth-input-wrapper">
                  <Lock size={18} className="auth-input-icon" />
                  <input
                    id="auth-login-password"
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    autoComplete="current-password"
                    required
                  />
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="auth-form-field">
                <label htmlFor="auth-reg-username">Username</label>
                <div className="auth-input-wrapper">
                  <User size={18} className="auth-input-icon" />
                  <input
                    id="auth-reg-username"
                    type="text"
                    placeholder="e.g. alice, study_bob"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    autoComplete="username"
                    required
                  />
                </div>
                <span className="auth-field-hint">3-30 letters, numbers, hyphens, or underscores</span>
              </div>

              <div className="auth-form-field">
                <label htmlFor="auth-reg-email">Email Address</label>
                <div className="auth-input-wrapper">
                  <Mail size={18} className="auth-input-icon" />
                  <input
                    id="auth-reg-email"
                    type="email"
                    placeholder="you@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    autoComplete="email"
                    required
                  />
                </div>
              </div>

              <div className="auth-form-field">
                <label htmlFor="auth-reg-password">Password</label>
                <div className="auth-input-wrapper">
                  <Lock size={18} className="auth-input-icon" />
                  <input
                    id="auth-reg-password"
                    type="password"
                    placeholder="Minimum 6 characters"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    autoComplete="new-password"
                    required
                  />
                </div>
              </div>

              <div className="auth-form-field">
                <label htmlFor="auth-reg-confirm">Confirm Password</label>
                <div className="auth-input-wrapper">
                  <KeyRound size={18} className="auth-input-icon" />
                  <input
                    id="auth-reg-confirm"
                    type="password"
                    placeholder="Re-enter password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    autoComplete="new-password"
                    required
                  />
                </div>
              </div>
            </>
          )}

          <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            <button
              type="submit"
              className="auth-submit-btn"
              disabled={isLoading}
            >
              {isLoading ? (
                <span>Please wait...</span>
              ) : mode === 'login' ? (
                <>
                  <LogIn size={18} />
                  <span>Sign In</span>
                </>
              ) : (
                <>
                  <UserPlus size={18} />
                  <span>Create Account</span>
                </>
              )}
            </button>

            {/* Quick Demo Helper Button for fast testing */}
            <button
              type="button"
              className="auth-demo-btn"
              onClick={() => handleFillDemo('student_demo')}
              title="Autofill demo credentials for quick local testing"
            >
              Fill Demo Credentials (student_demo)
            </button>
          </div>
        </form>

        {/* Modal Footer Note */}
        <div className="auth-modal-footer-note">
          <p>
            🔒 Passwords are hashed with bcrypt (rounds=12). All document vectors and queries are cryptographically isolated to your account.
          </p>
        </div>
      </div>
    </div>
  );
}
