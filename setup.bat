@echo off
echo Setting up SFIN environment with Python 3.10 and CUDA 12.4...

:: Check if Python 3.10 is installed
python --version 2>NUL | findstr /C:"Python 3.10" >NUL
if errorlevel 1 (
    echo Error: Python 3.10 not found. Please install Python 3.10 before continuing.
    exit /b 1
)

:: Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt
pause
:: Create necessary directories
echo Creating project directories...
mkdir checkpoints
mkdir visualizations
mkdir runs
mkdir optuna

echo.
echo Environment setup complete!
echo.
echo To activate the environment, run: venv\Scripts\activate
echo To start training, run: run.bat train
echo.
pause