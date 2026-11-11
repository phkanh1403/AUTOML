@echo off
setlocal
cd /d "%~dp0\.."
where py >nul 2>nul
if %errorlevel%==0 (
  set PYTHON=py
) else (
  set PYTHON=python
)
%PYTHON% -c "import fastapi, uvicorn, pandas, sklearn, skimage, PIL" >nul 2>nul
if errorlevel 1 (
  echo Dependencies are missing. Installing requirements.txt...
  %PYTHON% -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Installation failed. Check your Python installation and Internet connection.
    pause
    exit /b 1
  )
)
echo Starting the backend at http://127.0.0.1:8000 ...
start "AutoML Backend" cmd /k "%PYTHON% -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:8000
echo The backend is running in the AutoML Backend window.
