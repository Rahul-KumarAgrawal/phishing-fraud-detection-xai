/**
 * ExplanationPanel
 * Renders SHAP feature contributions in a compact, visual bar chart format.
 *
 * Each row shows two distinct numbers:
 *   - `feature_value`: the feature's actual value for this input (e.g. a URL
 *     feature count, or a token's TF-IDF weight).
 *   - `shap_value`: the SHAP contribution towards `explained_output`.
 * The panel header states which model output the contributions add up to and in
 * which units, so log-odds contributions are never read as probabilities.
 */
export default function ExplanationPanel({ explanation, type = 'message' }) {
  if (!explanation) return null;

  const {
    explanation_type,
    units,
    explained_output,
    base_value,
    top_positive_features,
    top_negative_features,
  } = explanation;

  const allFeatures = [
    ...top_positive_features.map(f => ({ ...f, dir: 'positive' })),
    ...top_negative_features.map(f => ({ ...f, dir: 'negative' })),
  ];

  if (allFeatures.length === 0) return null;

  const maxAbs = Math.max(
    0,
    ...allFeatures.map(f => Math.abs(f.shap_value ?? 0))
  );

  const unitsLabel = units === 'log_odds'
    ? 'log-odds units'
    : units === 'probability'
    ? 'probability units'
    : '';

  return (
    <div className="explanation-panel">
      <div className="explanation-header">
        <span aria-hidden="true">🧠</span>
        SHAP Explanation — <span style={{ color: 'var(--accent-blue)' }}>{explanation_type}</span>
        <span style={{ marginLeft: 'auto', fontWeight: 400, fontSize: '0.75rem' }}>
          {top_positive_features.length} risk factors ·{' '}
          {top_negative_features.length} reducing factors
        </span>
      </div>

      <div style={{
        padding: '0 1rem 0.4rem',
        fontSize: '0.72rem',
        color: 'var(--text-muted, #9ca3af)',
      }}>
        Contributions explain <strong>{explained_output || 'the model output'}</strong>
        {unitsLabel ? ` (${unitsLabel})` : ''}
        {typeof base_value === 'number' ? ` · baseline ${base_value.toFixed(3)}` : ''}
        {units === 'log_odds' ? ' · values are log-odds, not probabilities' : ''}
      </div>

      {top_positive_features.length > 0 && (
        <>
          <div style={{
            padding: '0.5rem 1rem 0.25rem',
            fontSize: '0.73rem',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--accent-red)',
            fontWeight: 600,
          }}>
            ↑ Increases risk
          </div>
          <div className="explanation-body" style={{ paddingTop: 0 }}>
            {top_positive_features.slice(0, 8).map((f, i) => (
              <FeatureRow key={i} feature={f} dir="positive" maxAbs={maxAbs} type={type} />
            ))}
          </div>
        </>
      )}

      {top_negative_features.length > 0 && (
        <>
          <div style={{
            padding: '0.5rem 1rem 0.25rem',
            fontSize: '0.73rem',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--accent-green)',
            fontWeight: 600,
          }}>
            ↓ Reduces risk
          </div>
          <div className="explanation-body" style={{ paddingTop: 0 }}>
            {top_negative_features.slice(0, 6).map((f, i) => (
              <FeatureRow key={i} feature={f} dir="negative" maxAbs={maxAbs} type={type} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function FeatureRow({ feature, dir, maxAbs }) {
  const shapVal = feature.shap_value ?? 0;
  const barWidth = maxAbs > 0 ? Math.abs(shapVal) / maxAbs * 100 : 0;
  const rawValue = feature.feature_value;
  const displayValue =
    rawValue === undefined || rawValue === null
      ? ''
      : typeof rawValue === 'boolean'
      ? (rawValue ? '1' : '0')
      : typeof rawValue === 'number'
      ? String(Math.round(rawValue * 10000) / 10000)
      : String(rawValue).slice(0, 20);
  // Keep very small contributions visible instead of rendering them as +0.0000.
  const abs = Math.abs(shapVal);
  const shapText =
    (shapVal >= 0 ? '+' : '') +
    (abs === 0 ? '0.0000' : abs >= 0.0001 ? shapVal.toFixed(4) : shapVal.toExponential(2));

  return (
    <div className={`feature-row ${dir}`} title={feature.description || feature.feature}>
      <span className="feature-name" title={feature.feature}>
        {feature.feature.replace(/_/g, '_\u200b')}
      </span>
      <div className="feature-bar-wrap">
        <div
          className={`feature-bar ${dir}`}
          style={{ width: `${barWidth.toFixed(1)}%` }}
        />
      </div>
      {displayValue && (
        <span className="feature-value" title={`Feature value for this input: ${displayValue}`}>
          value={displayValue}
        </span>
      )}
      <span className="feature-value" title="SHAP contribution to the explained output">
        {shapText}
      </span>
    </div>
  );
}
