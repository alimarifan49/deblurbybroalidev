@echo off
echo ================================
echo   Build DeblurApp by BroaliDEV
echo ================================

:: Aktifkan venv
call venv\Scripts\activate

:: Hapus build lama biar bersih
rmdir /s /q build
rmdir /s /q dist

:: Compile ke exe dengan folder (onedir)
pyinstaller --onedir --noconsole --name "DeblurApp_by_BroaliDEV" app.py

echo ================================
echo Build selesai!
echo Hasil ada di folder: dist\DeblurApp_by_BroaliDEV
echo Jalankan: dist\DeblurApp_by_BroaliDEV\DeblurApp_by_BroaliDEV.exe
echo ================================
pause
