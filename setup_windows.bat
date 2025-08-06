@echo off
REM Windows setup script for Snake RL project
echo Setting up Snake RL environment on Windows...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install CPU version by default
echo Installing CPU version of PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install other requirements
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate
echo.
echo To install CUDA version instead, run:
echo   setup_cuda.bat
echo.
pause