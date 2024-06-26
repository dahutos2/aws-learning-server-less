@echo off
setlocal

set /p imagePath="Enter the path to the image file: "
set /p fileName="Enter the file name to save as: "
set /p apiKey="Enter your API Key: "

powershell -File .\call_result_api.ps1 -imagePath "%imagePath%" -fileName "%fileName%" -apiKey "%apiKey%"

endlocal
pause