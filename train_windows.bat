@echo off
REM Windows training script for Snake RL project
echo Starting Snake RL training on Windows...

REM Activate virtual environment
call venv\Scripts\activate

REM Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo.
echo Select training mode:
echo 1. Quick CPU training (200 episodes)
echo 2. Full CPU training (2000 episodes)
echo 3. Quick CUDA training (200 episodes)
echo 4. Full CUDA training (5000 episodes)
echo 5. Custom training
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Running quick CPU training...
    python train/train_improved.py --episodes 200 --hidden_size 128 --log_interval 10
) else if "%choice%"=="2" (
    echo Running full CPU training...
    python train/train_improved.py --episodes 2000 --hidden_size 256 --log_interval 50
) else if "%choice%"=="3" (
    echo Running quick CUDA training...
    python train/train_improved.py --episodes 200 --hidden_size 128 --log_interval 10 --use_cuda
) else if "%choice%"=="4" (
    echo Running full CUDA training...
    python train/train_improved.py --episodes 5000 --hidden_size 256 --log_interval 50 --use_cuda --use_prioritized_replay
) else if "%choice%"=="5" (
    echo.
    echo Example custom commands:
    echo python train/train_improved.py --episodes 1000 --use_cuda --hidden_size 256 --lr 0.001 --use_prioritized_replay
    echo.
    echo Available parameters:
    echo   --episodes: Number of training episodes
    echo   --use_cuda: Use GPU acceleration
    echo   --hidden_size: Network hidden layer size (128, 256, 512)
    echo   --lr: Learning rate (0.001, 0.0001)
    echo   --use_prioritized_replay: Use prioritized experience replay
    echo   --batch_size: Batch size for training (64, 128, 256)
    echo.
    pause
    exit /b
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b
)

echo.
echo Training completed! Check the models/ directory for saved models.
echo.
echo To demo the trained model, run:
echo   demo_windows.bat
echo.
pause