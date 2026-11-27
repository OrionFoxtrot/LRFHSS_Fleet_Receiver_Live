@echo off
setlocal
cd /d "%~dp0"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "PYTHONDONTWRITEBYTECODE=1"
where python3 >nul 2>nul
if errorlevel 1 (
  echo ERROR: python3 is unavailable. Install Python and make python3 accessible on PATH.
  exit /b 1
)
python3 --version >nul 2>nul
if errorlevel 1 (
  echo ERROR: python3 could not start. Check the Python installation and Windows app aliases.
  exit /b 1
)
if exist "%~dp0requirements-benchmarks.txt" (
  python3 -u -m pip install -r requirements.txt -r requirements-benchmarks.txt %*
) else (
  python3 -u -m pip install -r requirements.txt %*
)
exit /b %errorlevel%
