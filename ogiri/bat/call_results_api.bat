@echo off
set API_KEY=YOUR_API_KEY
set API_URL=https://{YOUR_API_ID}.execute-api.ap-northeast-1.amazonaws.com/prod/results

curl -X GET %API_URL% -H "x-api-key: %API_KEY%" -o result.json

echo API Response:
type result.json
pause