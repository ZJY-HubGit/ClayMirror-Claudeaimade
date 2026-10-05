@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
title ClayMirror - NPU clay mirror
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"
set "VPY=%ROOT%.venv\Scripts\python.exe"

echo.
echo   ==============================================
echo     ClayMirror  -  real-time camera clay toy
echo     Intel NPU first (OpenVINO), GPU/CPU fallback
echo   ==============================================
echo.

if exist "%VPY%" goto deps

echo  [1/3] First run: looking for Python 3.10+ ...
call :findpy
if defined PY goto mkvenv
echo  Python 3.10+ not found. Trying to install Python 3.12 with winget ...
where winget >nul 2>nul || goto nopython
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY312%" set PY="%PY312%"
if not defined PY call :findpy
if not defined PY goto nopython

:mkvenv
echo  [2/3] Creating virtual environment with %PY% ...
%PY% -m venv "%ROOT%.venv"
if not exist "%VPY%" goto venvfail

:deps
if exist "%ROOT%.venv\deps-ok.txt" goto run
echo  [3/3] Installing packages (first run only, about 100 MB) ...
"%VPY%" -m pip install --disable-pip-version-check -q --upgrade pip
"%VPY%" -m pip install --disable-pip-version-check -r "%ROOT%requirements.txt"
if not errorlevel 1 goto depsok
echo  Retrying with a PyPI mirror ...
"%VPY%" -m pip install --disable-pip-version-check -r "%ROOT%requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto pipfail
:depsok
echo ok>"%ROOT%.venv\deps-ok.txt"

:run
echo  Starting ... the browser will open http://localhost:8848
echo  (first start compiles the models for the NPU, later starts are fast)
echo.
"%VPY%" "%ROOT%server\server.py" %*
echo.
echo  ClayMirror stopped.
pause
exit /b 0

:findpy
set "PY="
for %%V in (3.12 3.13 3.11 3.10 3.14) do call :trypy %%V
if defined PY exit /b 0
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"
exit /b 0

:trypy
if defined PY exit /b 0
py -%1 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY=py -%1"
exit /b 0

:nopython
echo.
echo  Python was not found. Please install Python 3.12 from
echo     https://www.python.org/downloads/
echo  (tick "Add python.exe to PATH"), then double-click start.bat again.
start "" https://www.python.org/downloads/
pause
exit /b 1

:venvfail
echo.
echo  Could not create the virtual environment (.venv). Delete the .venv folder and try again.
pause
exit /b 1

:pipfail
echo.
echo  Package installation failed. Check the network connection and run start.bat again.
pause
exit /b 1
