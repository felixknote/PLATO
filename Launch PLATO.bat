@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m plato.cli gui
if errorlevel 1 pause
