@echo off
setlocal

:: Activate virtual environment
call venv\Scripts\activate

:: Set CUDA and Python environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=.

:: Parse arguments
set MODE=train
set HYPEROPT=0
set CHECKPOINT=

:parse_args
if "%~1"=="" goto :execute
if /i "%~1"=="train" set MODE=train
if /i "%~1"=="evaluate" set MODE=evaluate
if /i "%~1"=="generate" set MODE=generate
if /i "%~1"=="explain" set MODE=explain
if /i "%~1"=="--hyperopt" set HYPEROPT=1
if /i "%~1"=="--checkpoint" set CHECKPOINT=%~2 & shift
shift
goto :parse_args

:execute
echo Running SFIN in %MODE% mode...

:: Create command string
set CMD=python sfin.py --mode %MODE%

if %HYPEROPT%==1 (
    set CMD=%CMD% --hyperopt
)

if not "%CHECKPOINT%"=="" (
    set CMD=%CMD% --checkpoint "%CHECKPOINT%"
)

:: Execute command
echo Executing: %CMD%
%CMD%

:: Deactivate virtual environment
pause
call venv\Scripts\deactivate
pause
endlocal