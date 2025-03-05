@echo off
setlocal EnableDelayedExpansion

:: Activate virtual environment
call venv\Scripts\activate

:: Set CUDA and Python environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=.

:: Default settings
set MODE=train
set HYPEROPT=0
set CHECKPOINT=

:: Display mode selection menu
echo.
echo ========== SFIN RUNNER ==========
echo.
echo Select execution mode:
echo 1. Train (with optimized settings)
echo 2. Evaluate
echo 3. Generate
echo 4. Explain
echo 5. Custom training settings
echo.
set /p CHOICE="Enter option (1-5): "

if "%CHOICE%"=="1" (
    set MODE=train
    set USE_OPTIMIZED=1
) else if "%CHOICE%"=="2" (
    set MODE=evaluate
    set USE_OPTIMIZED=0
) else if "%CHOICE%"=="3" (
    set MODE=generate
    set USE_OPTIMIZED=0
) else if "%CHOICE%"=="4" (
    set MODE=explain
    set USE_OPTIMIZED=0
) else if "%CHOICE%"=="5" (
    set MODE=train
    set USE_OPTIMIZED=0
) else (
    echo Invalid choice. Defaulting to optimized training.
    set MODE=train
    set USE_OPTIMIZED=1
)

echo.
:: Ask about hyperparameter optimization
set /p HYPEROPT_CHOICE="Use hyperparameter optimization? (y/n, default: n): "
if /i "%HYPEROPT_CHOICE%"=="y" set HYPEROPT=1

:: Ask about checkpoint loading
set /p CHECKPOINT_CHOICE="Load from checkpoint? (y/n, default: n): "
if /i "%CHECKPOINT_CHOICE%"=="y" (
    echo.
    echo Enter the path to checkpoint file:
    echo (e.g., checkpoints\sfin_epoch_3.pt)
    set /p CHECKPOINT="> "
)

:: Create base command string
set CMD=python sfin.py --mode %MODE%

:: Add hyperopt flag if selected
if %HYPEROPT%==1 (
    set CMD=%CMD% --hyperopt
)

:: Add checkpoint if provided
if not "%CHECKPOINT%"=="" (
    set CMD=%CMD% --checkpoint "%CHECKPOINT%"
)

:: Add optimized training settings if selected
if %USE_OPTIMIZED%==1 if "%MODE%"=="train" (
    set CMD=%CMD% --learning_rate 7e-5 --epochs 10 --model_dim 768 --batch_size 8
    :: Note: Additional optimal settings identified but not added as command line args:
    :: --depth 7 --dropout 0.05 --interference_type classical --collapse_type entanglement --use_hierarchical
)

:: Custom training settings
if "%CHOICE%"=="5" (
    echo.
    echo ===== Custom Training Settings =====
    
    set /p EPOCHS="Epochs (default: 3): "
    if not "!EPOCHS!"=="" set CMD=%CMD% --epochs !EPOCHS!
    
    set /p LR="Learning rate (default: 5e-5): "
    if not "!LR!"=="" set CMD=%CMD% --learning_rate !LR!
    
    set /p BATCH="Batch size (default: auto): "
    if not "!BATCH!"=="" set CMD=%CMD% --batch_size !BATCH!
    
    set /p MODEL_DIM="Model dimension (default: 768): "
    if not "!MODEL_DIM!"=="" set CMD=%CMD% --model_dim !MODEL_DIM!
    
    set /p SEED="Random seed (default: 42): "
    if not "!SEED!"=="" set CMD=%CMD% --seed !SEED!
    
    set /p ENABLE_MEMORY="Enable memory? (y/n, default: n): "
    if /i "!ENABLE_MEMORY!"=="y" set CMD=%CMD% --enable_memory
)

:: Display the constructed command
echo.
echo ===== EXECUTION COMMAND =====
echo %CMD%
echo =============================
echo.

:: Confirm execution
set /p CONFIRM="Execute this command? (y/n, default: y): "
if /i not "%CONFIRM%"=="n" (
    echo.
    echo Executing command...
    echo.
    %CMD%
) else (
    echo.
    echo Execution cancelled.
)

:: Deactivate virtual environment
call venv\Scripts\deactivate
pause
endlocal