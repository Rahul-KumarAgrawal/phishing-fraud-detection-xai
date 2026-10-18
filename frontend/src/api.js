// API base URL
// In dev: empty string lets Vite's proxy route /api/* to localhost:8000
// In prod: set VITE_API_URL to the backend's base URL
const BASE_URL = import.meta.env.VITE_API_URL || '';

/**
 * Check API health and model status.
 * @returns {{ status, message_model_loaded, url_model_loaded }}
 */
export async function checkHealth() {
  const resp = await fetch(`${BASE_URL}/api/health`);
  if (!resp.ok) throw new Error('Backend unreachable');
  return resp.json();
}

/**
 * Analyze a text message for phishing/fraud.
 * @param {string} message
 */
export async function analyzeMessage(message) {
  const resp = await fetch(`${BASE_URL}/api/analyze/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
  return data;
}

/**
 * Analyze a URL for phishing.
 * @param {string} url
 */
export async function analyzeUrl(url) {
  const resp = await fetch(`${BASE_URL}/api/analyze/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
  return data;
}
