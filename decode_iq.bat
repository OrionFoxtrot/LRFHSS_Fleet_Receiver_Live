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
if "%~1"=="" (
  echo Drag an IQ file onto this launcher or supply its path.
  exit /b 1
)
python3 -u src\lrfhss_live.py --throughput -v --file %*
exit /b %errorlevel%
