@echo off
REM Windows demo script for Snake RL project
echo Snake RL Demo on Windows...

REM Activate virtual environment
call venv\Scripts\activate

echo.
echo Select demo mode:
echo 1. Original model demo
echo 2. Improved model demo
echo 3. Best model demo
echo 4. Console demo (no graphics)
echo 5. Model performance evaluation
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    if exist "models\dqn_snake.pt" (
        echo Running original model demo...
        python demo\demo.py --model_path models\dqn_snake.pt
    ) else (
        echo Original model not found. Please train first.
    )
) else if "%choice%"=="2" (
    if exist "models\improved_dqn_snake.pt" (
        echo Running improved model demo...
        python demo_improved.py --model_path models\improved_dqn_snake.pt
    ) else (
        echo Improved model not found. Please train first.
    )
) else if "%choice%"=="3" (
    if exist "models\best_dqn_snake.pt" (
        echo Running best model demo...
        python demo_improved.py --model_path models\best_dqn_snake.pt
    ) else (
        echo Best model not found. Please train first.
    )
) else if "%choice%"=="4" (
    echo Select model for console demo:
    echo 1. Original model
    echo 2. Improved model
    set /p model_choice="Enter choice (1-2): "
    
    if "!model_choice!"=="1" (
        if exist "models\dqn_snake.pt" (
            python demo_console.py --model_path models\dqn_snake.pt --max_steps 500
        ) else (
            echo Original model not found.
        )
    ) else if "!model_choice!"=="2" (
        if exist "models\improved_dqn_snake.pt" (
            python evaluate_improved.py --improved_model models\improved_dqn_snake.pt --episodes 10
        ) else (
            echo Improved model not found.
        )
    )
) else if "%choice%"=="5" (
    echo Running model evaluation...
    if exist "models\improved_dqn_snake.pt" (
        python evaluate_improved.py --improved_model models\improved_dqn_snake.pt --episodes 50 --compare
    ) else (
        echo No models found for evaluation. Please train first.
    )
) else (
    echo Invalid choice. Exiting.
)

echo.
pause