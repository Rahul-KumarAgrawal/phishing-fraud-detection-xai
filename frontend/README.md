# Frontend – Phishing & Fraud Detection XAI

React + Vite frontend for the CSET312 mini-project.

## Prerequisites

- Node.js >= 18 (tested on 22.x)
- Backend running at `http://localhost:8000`

## Setup & Run

```powershell
# From repo root, enter the frontend directory
cd frontend

# Install dependencies
npm install

# Start dev server (hot-reload)
npm run dev
```

The app opens at **http://localhost:5173**.

## Build for production

```powershell
npm run build
# Output goes to frontend/dist/
```

## Environment

Create `frontend/.env` to override the API base URL:

```
VITE_API_URL=http://localhost:8000
```

## Structure

```
frontend/
+-- src/
|   +-- App.jsx               # Root component with tab navigation
|   +-- api.js                # API client (fetch wrappers)
|   +-- index.css             # Design system
|   +-- main.jsx              # React entry point
|   +-- components/
|       +-- MessageAnalysis.jsx   # Message / SMS analysis tab
|       +-- UrlAnalysis.jsx       # URL analysis tab
|       +-- ExplanationPanel.jsx  # SHAP visualization component
+-- index.html
+-- vite.config.js
```

## Notes

- All predictions come from the running backend -- no hardcoded results.
- Submitted text and URLs are never stored in the browser or sent to third parties.
- The URL destination is never fetched or visited.
