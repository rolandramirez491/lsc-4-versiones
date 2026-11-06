@echo off
rem Reproduce el experimento de las 4 configuraciones y lo compara con los resultados originales.
rem Uso:  reproducir.bat                         (usa el banco incluido: lo normal)
rem       reproducir.bat "C:\ruta\datos.json"    (además rehace el banco desde LSC-54 y comprueba que es idéntico)
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0codigo"

echo === 1) Entorno (versiones y GPU) ===
python verificar.py entorno

echo.
echo === 2) Datos de entrada y banco (huellas SHA-256) ===
python verificar.py datos || (echo LOS DATOS NO SON LOS ORIGINALES & pause & exit /b 1)

if not "%~1"=="" (
  echo.
  echo === 2b) Rehaciendo el banco desde los datasets oficiales ===
  if exist "..\dataset 90 senas (reconstruido)" rmdir /s /q "..\dataset 90 senas (reconstruido)"
  python crear_dataset_90.py "%~1" || (pause & exit /b 1)
  python verificar.py reconstruido || (echo EL BANCO REHECHO NO ES IGUAL & pause & exit /b 1)
)

echo.
echo === 3) Experimento: B0, M1, M2, M3 (5 vueltas, semilla 42; unos 20-30 min con GPU) ===
python experimento_4.py || (pause & exit /b 1)

echo.
echo === 4) Comparacion con los resultados originales ===
python verificar.py resultados
echo.
echo Graficas y tablas nuevas en la carpeta "resultados".
pause
