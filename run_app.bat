@echo off
cd /d "%~dp0"
echo Starting Fashion Product Recommendation System...
call venv\Scripts\activate
streamlit run app/streamlit_app.py
pause
