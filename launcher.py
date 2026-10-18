"""
launcher.py
===========
Starts the FastAPI backend and React frontend in separate windows,
auto-trains models if missing, and opens the browser.

Usage:  Double-click START_APP.bat  (which calls this script)
"""

import os
import sys
import time
import pathlib
import subprocess
import webbrowser

# ── Config ─────────────────────────────────────────────────────────
REPO = pathlib.Path(__file__).resolve().parent
PYTHON = sys.executable
BACKEND_PORT = 8000
FRONTEND_PORT = 5173

os.environ["PYTHONIOENCODING"] = "utf-8"

print()
print("  ====================================================")
print("  Phishing and Fraud Detection XAI  -  CSET312")
print("  Bennett University")
print("  ====================================================")
print()


# ── Helpers ────────────────────────────────────────────────────────
def run(args, label):
    print(f"  [{label}] Running: {' '.join(str(a) for a in args)}")
    result = subprocess.run(args, cwd=str(REPO))
    if result.returncode != 0:
        print(f"  [ERROR] {label} failed (exit {result.returncode})")
        input("  Press Enter to exit...")
        sys.exit(1)


def check(cmd, name):
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"  [ERROR] {name} not found. Please install it first.")
        input("  Press Enter to exit...")
        sys.exit(1)


# ── Checks ─────────────────────────────────────────────────────────
check(["node", "--version"], "Node.js")

# ── Auto-train models if missing ───────────────────────────────────
msg_model = REPO / "models" / "message_model.joblib"
url_model = REPO / "models" / "url_model.joblib"

if not msg_model.exists():
    print("  [INFO] Message model not found - training now ...")
    run([PYTHON, str(REPO / "scripts" / "train_message_model.py")], "Train MSG")
    print()

if not url_model.exists():
    print("  [INFO] URL model not found - training now (~30s) ...")
    run([PYTHON, str(REPO / "scripts" / "train_url_model.py")], "Train URL")
    print()

# ── Install npm deps if missing ────────────────────────────────────
frontend = REPO / "frontend"
if not (frontend / "node_modules").exists():
    print("  [INFO] Installing frontend npm packages ...")
    subprocess.run(["npm", "install"], cwd=str(frontend), shell=True)
    print()

# ── Start backend in a new window ──────────────────────────────────
print(f"  [1/2] Starting backend on port {BACKEND_PORT} ...")
backend_cmd = f'"{PYTHON}" -m uvicorn fraud_phishing.api.main:app --port {BACKEND_PORT} --reload'
subprocess.Popen(
    f'start "Backend - FastAPI" cmd /k "set PYTHONIOENCODING=utf-8 & cd /d {str(REPO)} & {backend_cmd}"',
    shell=True,
)
time.sleep(5)

# ── Start frontend in a new window ─────────────────────────────────
print(f"  [2/2] Starting frontend on port {FRONTEND_PORT} ...")
subprocess.Popen(
    f'start "Frontend - React" cmd /k "cd /d {str(frontend)} & npm run dev"',
    shell=True,
)
time.sleep(3)

# ── Open browser ───────────────────────────────────────────────────
url = f"http://localhost:{FRONTEND_PORT}"
print(f"  Opening browser at {url} ...")
webbrowser.open(url)

print()
print("  ====================================================")
print("  App is running!")
print()
print(f"  Frontend :  http://localhost:{FRONTEND_PORT}")
print(f"  Backend  :  http://localhost:{BACKEND_PORT}")
print(f"  API Docs :  http://localhost:{BACKEND_PORT}/docs")
print()
print("  Close the Backend and Frontend windows to stop.")
print("  ====================================================")
print()
input("  Press Enter to close this launcher window...")
