@echo off
setlocal
python --version
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Setup complete. If TensorFlow installed successfully, run: run_exploration.bat
pause
