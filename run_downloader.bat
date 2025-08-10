@echo off
echo ========================================
echo    Market Data Downloader
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated successfully
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Virtual environment created and activated
)

echo.
echo Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo Testing setup...
python test_setup.py

echo.
echo Running small test download...
python test_download.py

echo.
echo Starting full data download...
python data_downloader.py

echo.
echo Download completed!
pause 