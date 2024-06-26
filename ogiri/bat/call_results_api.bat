@echo off
setlocal

set /p API_KEY="Enter your API Key: "
set API_URL=https://{CloudFrontのドメイン名(.cloudfront.netで終わる)}/prod/results

curl -X GET %API_URL% -H "x-api-key: %API_KEY%" -o result.json

echo API Response:
type result.json

endlocal
pause