@echo off
rem ===========================================================================
rem  DriveRL launcher for cmd.exe / Explorer double-click.
rem
rem    run.cmd            backend only (serves the built frontend on :8000)
rem    run.cmd --dev      backend + Vite dev server (open :5173)
rem    run.cmd --build    build the frontend first, then serve on :8000
rem
rem  !! ASCII ONLY IN THIS FILE !!
rem  cmd.exe parses .cmd files using the console code page (cp932 on Japanese
rem  Windows), not UTF-8. UTF-8 Japanese text turns into mojibake and each
rem  broken line is then executed as a command, producing a flood of
rem  "'...' is not recognized as an internal or external command".
rem  Keep every byte in this file ASCII. Japanese explanations live in README.md.
rem
rem  PowerShell users: run.ps1 does the same thing.
rem  Both are thin wrappers around backend\run.py.
rem ===========================================================================
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%backend\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo.
    echo   venv not found: backend\.venv
    echo.
    echo   Create it first:
    echo     py -3.13 -m venv backend\.venv
    echo     backend\.venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%PYTHON%" "%ROOT%backend\run.py" %*
set "CODE=%ERRORLEVEL%"

rem Keep the window open on failure so a double-click user can read why.
if not "%CODE%"=="0" pause
endlocal & exit /b %CODE%
