@echo off
rem Abre la camara para probar las 4 versiones haciendo senas (teclas 1-4 cambian de version, ESC sale).
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0codigo"
if not exist "..\modelos\cuatro_versiones.pt" python versiones.py entrenar
python probar_en_vivo.py %*
pause
