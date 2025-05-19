@echo off
setlocal enabledelayedexpansion

REM Check for administrator privileges
openfiles >nul 2>&1
if '%errorlevel%' NEQ '0' (
    echo Requesting administrative privileges...
    powershell start-process '%0' -verb runas
    exit /b
)

REM Define the folder path to delete
set targetFolder=%SystemRoot%\system32\screenshots

REM Define the path to current script
set scriptPath=%~f0

REM Check if the target folder exists
if exist "%targetFolder%" (
    echo Deleting folder "%targetFolder%"...
    
    REM Attempt to delete folder
    rd /S /Q "%targetFolder%"
    
    REM Check if the deletion was successful
    if exist "%targetFolder%" (
        echo Deletion failed. Please check if the folder is in use.
    ) else (
        echo Deletion successful.
    )
) else (
    echo Target folder does not exist: %targetFolder%
)

REM Delete this script itself
del "%~f0"

endlocal
exit /b
