@echo off
setlocal
cd /d "%~dp0\.."
where py >nul 2>nul
if %errorlevel%==0 (set PYTHON=py) else (set PYTHON=python)
%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install -r requirements.txt
pause
