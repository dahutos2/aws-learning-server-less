@echo off
setlocal

REM S3バケット名とCSVファイルのパスを設定
set AWS_BUCKET=ogiri-training-data-bucket
set CSV_FILE=path\to\your\train_data.csv

REM CSVファイルをS3にアップロード
aws s3 cp %CSV_FILE% s3://%AWS_BUCKET%/

endlocal