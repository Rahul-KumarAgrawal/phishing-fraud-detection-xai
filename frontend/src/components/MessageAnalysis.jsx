import { useState } from 'react';
import { analyzeMessage } from '../api';
import ExplanationPanel from './ExplanationPanel';

const MAX_LEN = 10_000;

export default function MessageAnalysis() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleAnalyze() {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const data = await analyzeMessage(text.trim());
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setText('');
    setResult(null);
    setError(null);
  }

  const isSuspicious = result && result.message_label === 1;
  const verdictClass = result ? (isSuspicious ? 'suspicious' : 'legitimate') : '';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Input card */}
      <div className="card">
        <h2 className="card-title">
          <span aria-hidden="true">💬</span> Message / Email Analysis
        </h2>

        <label htmlFor="msg-input" style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>
          Paste an SMS or email message to analyse for phishing or fraud:
        </label>
        <textarea
          id="msg-input"
          className="textarea"
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder={'Paste a message here…\n\nExample: "URGENT: Your bank account has been suspended. Click http://bit.ly/claimNow to verify."'}
          maxLength={MAX_LEN}
          disabled={loading}
          aria-label="Message text for analysis"
          aria-describedby="msg-charcount"
        />
        <div id="msg-charcount" className="char-count">{text.length} / {MAX_LEN.toLocaleString()}</div>

        <div className="btn-row">
          <button
            id="analyze-message-btn"
            className="btn btn-primary"
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            aria-busy={loading}
          >
            {loading ? <><div className="spinner" aria-hidden="true" /> Analysing…</> : '🔍 Analyse Message'}
          </button>
          <button
            id="clear-message-btn"
            className="btn btn-secondary"
            onClick={handleClear}
            disabled={loading}
          >
            ✕ Clear
          </button>
        </div>

        {/* Sample inputs */}
        <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', alignSelf: 'center' }}>Try:</span>
          {SAMPLES.map((s, i) => (
            <button
              key={i}
              className="btn btn-secondary"
              style={{ fontSize: '0.72rem', padding: '3px 10px' }}
              onClick={() => { setText(s.text); setResult(null); setError(null); }}
              title={s.label}
            >
              {s.label}
            </button>
          ))}
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
            <span aria-hidden="true">📊</span> Message Result
          </h3>

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
                score: {(result.message_score * 100).toFixed(1)}%
              </span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                threshold: {(result.message_threshold * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* SHAP explanation */}
          <ExplanationPanel explanation={result.explanation} type="message" />

          {/* Detected URLs */}
          {result.detected_urls && result.detected_urls.length > 0 && (
            <div className="url-results-section">
              <div className="url-results-title">
                <span aria-hidden="true">🔗</span>
                URLs Detected in Message ({result.detected_urls.length})
              </div>
              {result.detected_urls.map((u, i) => (
                <div key={i} className="url-item">
                  <div style={{ flex: 1 }}>
                    <div className="url-text">{u.url}</div>
                    <div style={{ marginTop: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className={`score-pill ${u.label === 1 ? 'suspicious' : 'legitimate'}`} style={{ fontSize: '0.72rem', padding: '2px 8px' }}>
                        {u.label_name}
                      </span>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                        score: {(u.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                URL predictions are independent from the message prediction.
              </p>
            </div>
          )}

          {/* Disclaimer */}
          <div className="disclaimer" role="note">
            <span aria-hidden="true">ℹ</span>
            <span>{result.disclaimer}</span>
          </div>
        </div>
      )}
    </div>
  );
}

const SAMPLES = [
  {
    label: '🟥 Smishing',
    text: 'URGENT: Your account will be suspended. Verify immediately at http://bit.ly/secure-verify-now or lose access!',
  },
  {
    label: '🟩 Ham',
    text: 'Hey, are you free for lunch tomorrow? The team meeting was rescheduled to Thursday.',
  },
  {
    label: '🔗 URL msg',
    text: 'Congratulations! You have been selected. Click http://free-prize.example.com/claim to redeem your gift card!',
  },
];
