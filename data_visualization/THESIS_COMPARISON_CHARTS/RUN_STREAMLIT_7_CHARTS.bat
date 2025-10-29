@echo off
echo ===================================
echo   STREAMLIT - 7 CHARTS
echo ===================================
echo.
echo Ten skrypt uruchamia aplikacje Streamlit
echo zawierajaca wszystkie 7 wykresow:
echo.
echo   1. Execution Time
echo   2. Operation Breakdown
echo   3. Memory Usage (DP)
echo   4. Scalability
echo   5. Training Time (ML)
echo   6. Inference Speed (ML)
echo   7. Memory Usage (ML)
echo.
echo ===================================
echo.
echo Starting Streamlit...
echo Otwieranie w przegladarce: http://localhost:8501
echo.
echo UWAGA: Nie zamykaj tego okna!
echo        Aby zatrzymac serwer: Ctrl+C
echo.
echo ===================================
echo.

streamlit run STREAMLIT_7_CHARTS.py

echo.
echo.
echo ===================================
echo   Serwer Streamlit zatrzymany
echo ===================================
pause




