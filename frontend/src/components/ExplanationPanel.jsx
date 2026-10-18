/**
 * ExplanationPanel
 * Renders SHAP feature contributions in a compact, visual bar chart format.
 */
export default function ExplanationPanel({ explanation, type = 'message' }) {
  if (!explanation) return null;

  const { explanation_type, top_positive_features, top_negative_features, disclaimer } = explanation;

  const allFeatures = [
    ...top_positive_features.map(f => ({ ...f, dir: 'positive' })),
    ...top_negative_features.map(f => ({ ...f, dir: 'negative' })),
  ];

  if (allFeatures.length === 0) return null;

  const maxAbs = Math.max(
    ...allFeatures.map(f => Math.abs(f.shap_value ?? (typeof f.value === 'number' ? f.value : 0)))
  );

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
  const displayValue = typeof feature.value === 'boolean'
    ? (feature.value ? '1' : '0')
    : feature.value !== undefined && feature.value !== null
    ? String(feature.value).slice(0, 20)
    : '';

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
        <span className="feature-value" title={`Value: ${displayValue}`}>
          val={displayValue}
        </span>
      )}
      <span className="feature-value">
        {shapVal >= 0 ? '+' : ''}{shapVal.toFixed(3)}
      </span>
    </div>
  );
}
