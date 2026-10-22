import { useState, useEffect } from 'react';
import { checkHealth } from './api';
import MessageAnalysis from './components/MessageAnalysis';
import UrlAnalysis from './components/UrlAnalysis';
import './index.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('message');
  const [health, setHealth] = useState({ loading: true });

  // Poll health once on mount
  useEffect(() => {
    checkHealth()
      .then(data => setHealth({ ...data, loading: false, error: null }))
      .catch(err => setHealth({ loading: false, error: err.message }));
  }, []);

  const healthStatus = health.loading
    ? 'loading'
    : health.error
    ? 'offline'
    : 'online';

  return (
    <div className="app-wrapper">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon" aria-hidden="true">🛡</div>
            <span className="logo-text">Phishing Detector XAI</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {/* Model status */}
            <div className="health-bar" title="API / model status">
              <div className={`health-dot ${healthStatus}`} />
              {health.loading
                ? <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Connecting…</span>
                : health.error
                ? <span style={{ color: 'var(--accent-red)', fontSize: '0.75rem' }}>Backend offline</span>
                : <>
                    <span style={{ color: health.message_model_loaded ? 'var(--accent-green)' : 'var(--accent-orange)', fontSize: '0.75rem' }}>
                      MSG: {health.message_model_loaded ? '✓' : '✗'}
                    </span>
                    <span style={{ color: health.url_model_loaded ? 'var(--accent-green)' : 'var(--accent-orange)', fontSize: '0.75rem' }}>
                      URL: {health.url_model_loaded ? '✓' : '✗'}
                    </span>
                  </>
              }
            </div>
            <span className="header-badge">CSET312 · Bennett University</span>
          </div>
        </div>
      </header>

      {/* ── Main content ────────────────────────────────────────── */}
      <main className="main-content">
        <div className="alert-error" style={{ marginBottom: '1.5rem' }} role="note">
          <span>⚠</span>
          <span>
            <strong>COLLEGE PROJECT PROTOTYPE</strong> — this is an educational
            demonstration, not a dependable production security control. Verify
            important messages and URLs through official channels.
          </span>
        </div>

        {/* Model-not-loaded warning */}
        {!health.loading && !health.error && (!health.message_model_loaded || !health.url_model_loaded) && (
          <div className="alert-error" style={{ marginBottom: '1.5rem' }}>
            <span>⚠</span>
            <span>
              {!health.message_model_loaded && 'Message model not trained. '}
              {!health.url_model_loaded && 'URL model not trained. '}
              Run <code style={{ fontFamily: 'JetBrains Mono, monospace', background: 'rgba(0,0,0,0.3)', padding: '1px 6px', borderRadius: '4px' }}>
                python scripts/train_message_model.py
              </code> and <code style={{ fontFamily: 'JetBrains Mono, monospace', background: 'rgba(0,0,0,0.3)', padding: '1px 6px', borderRadius: '4px' }}>
                python scripts/train_url_model.py
              </code> first.
            </span>
          </div>
        )}

        {/* Prototype / non-real-data warning */}
        {!health.loading && !health.error && health.prototype && (
          <div className="alert-error" style={{ marginBottom: '1.5rem' }}>
            <span>⚠</span>
            <span>
              <strong>PROTOTYPE</strong> — the message model was trained on{' '}
              <strong>{health.message_dataset_provenance}</strong> data, not the real
              Mendeley SMS dataset. Treat the results below as a development
              demonstration, not an evaluation.
            </span>
          </div>
        )}

        {/* Tab selector */}
        <nav className="tabs" role="tablist" aria-label="Analysis mode">
          <button
            className={`tab-btn${activeTab === 'message' ? ' active' : ''}`}
            role="tab"
            aria-selected={activeTab === 'message'}
            aria-controls="panel-message"
            id="tab-message"
            onClick={() => setActiveTab('message')}
          >
            <span aria-hidden="true">💬</span> Message Analysis
          </button>
          <button
            className={`tab-btn${activeTab === 'url' ? ' active' : ''}`}
            role="tab"
            aria-selected={activeTab === 'url'}
            aria-controls="panel-url"
            id="tab-url"
            onClick={() => setActiveTab('url')}
          >
            <span aria-hidden="true">🔗</span> URL Analysis
          </button>
        </nav>

        {/* Tab panels */}
        <div id="panel-message" role="tabpanel" aria-labelledby="tab-message" hidden={activeTab !== 'message'}>
          <MessageAnalysis />
        </div>
        <div id="panel-url" role="tabpanel" aria-labelledby="tab-url" hidden={activeTab !== 'url'}>
          <UrlAnalysis />
        </div>
      </main>
    </div>
  );
}
