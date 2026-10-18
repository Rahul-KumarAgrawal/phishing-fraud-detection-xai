@echo off
setlocal
title Phishing Fraud Detector XAI

set "PY=python"
if exist "D:\Python311\python.exe" set "PY=D:\Python311\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"

"%PY%" "%~dp0launcher.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with code %ERRORLEVEL%.
    pause
)
