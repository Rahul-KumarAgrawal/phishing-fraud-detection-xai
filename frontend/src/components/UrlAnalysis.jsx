import { useState } from 'react';
import { analyzeUrl } from '../api';
import ExplanationPanel from './ExplanationPanel';

const MAX_URL_LEN = 4096;

// Features to highlight prominently in the feature summary card
const KEY_FEATURES = [
  'is_https', 'is_ip_host', 'is_url_shortener', 'suspicious_token_count',
  'url_length', 'subdomain_count', 'dot_count', 'has_at_symbol',
  'has_percent_encoding', 'path_depth', 'query_param_count',
];

export default function UrlAnalysis() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showAllFeatures, setShowAllFeatures] = useState(false);

  async function handleAnalyze() {
    const trimmed = url.trim();
    if (!trimmed) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const data = await analyzeUrl(trimmed);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setUrl('');
    setResult(null);
    setError(null);
    setShowAllFeatures(false);
  }

  const isSuspicious = result && result.label === 1;
  const verdictClass = result ? (isSuspicious ? 'suspicious' : 'legitimate') : '';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Input card */}
      <div className="card">
        <h2 className="card-title">
          <span aria-hidden="true">🔗</span> URL Phishing Analysis
        </h2>

        <label htmlFor="url-input" style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>
          Enter a URL to analyse using static feature extraction (destination is never visited):
        </label>
        <input
          id="url-input"
          className="input"
          type="url"
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleAnalyze(); }}
          placeholder="https://example.com/path?query=value"
          maxLength={MAX_URL_LEN}
          disabled={loading}
          aria-label="URL to analyse"
          autoComplete="off"
          spellCheck="false"
        />
        <div className="char-count">{url.length} / {MAX_URL_LEN.toLocaleString()}</div>

        <div className="btn-row">
          <button
            id="analyze-url-btn"
            className="btn btn-primary"
            onClick={handleAnalyze}
            disabled={loading || !url.trim()}
            aria-busy={loading}
          >
            {loading ? <><div className="spinner" aria-hidden="true" /> Analysing…</> : '🔍 Analyse URL'}
          </button>
          <button
            id="clear-url-btn"
            className="btn btn-secondary"
            onClick={handleClear}
            disabled={loading}
          >
            ✕ Clear
          </button>
        </div>

        {/* Sample URLs */}
        <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', alignSelf: 'center' }}>Try:</span>
          {URL_SAMPLES.map((s, i) => (
            <button
              key={i}
              className="btn btn-secondary"
              style={{ fontSize: '0.72rem', padding: '3px 10px' }}
              onClick={() => { setUrl(s.url); setResult(null); setError(null); }}
              title={s.label}
            >
              {s.label}
            </button>
          ))}
        </div>

        <div style={{ marginTop: '0.75rem', padding: '0.6rem 0.875rem', background: 'rgba(56,139,253,0.06)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(56,139,253,0.15)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          🔒 The URL destination is <strong style={{ color: 'var(--text-secondary)' }}>never visited, fetched, or resolved.</strong> Analysis uses only the URL string itself.
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="alert-error" role="alert">
          <span aria-hidden="true">⚠</span>
          <span><strong>Error:</strong> {error}</span>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="card" style={{ padding: '1.5rem' }}>
          <h3 className="card-title" style={{ marginBottom: '0.75rem' }}>
            <span aria-hidden="true">📊</span> URL Result
          </h3>

          {/* URL display */}
          <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem', color: 'var(--accent-blue)', wordBreak: 'break-all', background: 'var(--bg-secondary)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', marginBottom: '0.75rem' }}>
            {result.url}
          </div>

          {/* Verdict banner */}
          <div className={`result-banner ${verdictClass}`} role="status" aria-live="polite">
            <div className={`result-label ${verdictClass}`}>
              {isSuspicious
                ? <><span aria-hidden="true">🚨</span> Suspicious</>
                : <><span aria-hidden="true">✅</span> Likely Legitimate</>
              }
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '4px' }}>
              <span className={`score-pill ${verdictClass}`}>
                score: {(result.score * 100).toFixed(1)}%
              </span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                threshold: {(result.threshold * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Key feature chips */}
          {result.features && (
            <div style={{ marginTop: '1.25rem' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span aria-hidden="true">⚙</span> Key Extracted Features
              </div>
              <div className="feature-grid">
                {KEY_FEATURES.map(name => (
                  result.features[name] !== undefined && (
                    <div key={name} className="feature-chip">
                      <span className="feature-chip-name">{name.replace(/_/g, ' ')}</span>
                      <span className="feature-chip-val">
                        {typeof result.features[name] === 'number' && !Number.isInteger(result.features[name])
                          ? result.features[name].toFixed(3)
                          : String(result.features[name])
                        }
                      </span>
                    </div>
                  )
                ))}
              </div>

              {/* Toggle all features */}
              <button
                className="btn btn-secondary"
                style={{ marginTop: '0.75rem', fontSize: '0.75rem', padding: '4px 12px' }}
                onClick={() => setShowAllFeatures(v => !v)}
                aria-expanded={showAllFeatures}
                aria-controls="all-features-panel"
              >
                {showAllFeatures ? '▲ Hide all features' : `▼ Show all ${Object.keys(result.features).length} features`}
              </button>

              {showAllFeatures && (
                <div id="all-features-panel" className="feature-grid" style={{ marginTop: '0.6rem' }}>
                  {Object.entries(result.features)
                    .filter(([name]) => !KEY_FEATURES.includes(name))
                    .map(([name, val]) => (
                      <div key={name} className="feature-chip">
                        <span className="feature-chip-name">{name.replace(/_/g, ' ')}</span>
                        <span className="feature-chip-val">
                          {typeof val === 'number' && !Number.isInteger(val) ? val.toFixed(3) : String(val)}
                        </span>
                      </div>
                    ))
                  }
                </div>
              )}
            </div>
          )}

          {/* SHAP explanation */}
          <ExplanationPanel explanation={result.explanation} type="url" />

          {/* Disclaimer */}
          <div className="disclaimer" role="note" style={{ marginTop: '1.25rem' }}>
            <span aria-hidden="true">ℹ</span>
            <span>{result.disclaimer}</span>
          </div>
        </div>
      )}
    </div>
  );
}

const URL_SAMPLES = [
  { label: '🟩 Legit', url: 'https://www.google.com/search?q=example' },
  { label: '🟥 Suspicious', url: 'http://192.168.1.1/login?verify=true&account=update' },
  { label: '🔗 Shortener', url: 'http://bit.ly/3xampleLink' },
  { label: '🟧 Long path', url: 'http://secure-login.paypal-verify.suspicious-domain.com/account/verify/confirm?token=abc123' },
];
