@echo off
set /p imagePath="Enter the path to the image file: "
set /p fileName="Enter the file name to save as: "
set /p apiKey="Enter your API Key: "

:: 画像をbase64に変換する
for /f "delims=" %%A in ('certutil -encode %imagePath% temp.b64 ^& findstr /v /c:- temp.b64') do set base64Image=%%A

:: APIのURLを設定する
set apiUrl=https://{YOUR_API_ID}.execute-api.ap-northeast-1.amazonaws.com/prod/result

:: リクエストを送信し、レスポンスをキャプチャする
for /f "delims=" %%A in ('curl -X POST %apiUrl% -H "x-api-key: %apiKey%" -H "Content-Type: application/json" -d "{\"image\": \"%base64Image%\", \"filename\": \"%fileName%\"}"') do set response=%%A

:: レスポンスから結果を取り出す
for /f "tokens=1,2 delims=:" %%A in ("!response:{=!") do (
    if "%%A"=="\"result\"" (
        set result=%%B
        set result=!result:~1,-2!
        echo Result: !result!
    )
)

:: クリーンアップ
del temp.b64
pause