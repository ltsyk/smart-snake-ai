@echo off
REM Windows CUDA setup script for Snake RL project
echo Setting up Snake RL environment with CUDA support...

REM Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo Python not found or PyTorch not installed. Running basic setup first...
    call setup_windows.bat
)

REM Activate virtual environment
call venv\Scripts\activate

echo.
echo Checking CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count()); print('Current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'None')"

REM Install CUDA version of PyTorch (CUDA 11.8 compatible)
echo.
echo Installing CUDA version of PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other requirements
pip install -r requirements.txt

echo.
echo CUDA setup complete!
echo.
echo Verifying CUDA installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo.
echo To activate the environment, run:
echo   venv\Scripts\activate
echo.
pause