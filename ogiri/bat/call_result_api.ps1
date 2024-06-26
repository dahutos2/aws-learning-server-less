param (
   [string]$imagePath,
   [string]$fileName,
   [string]$apiKey
)

# プリサインドURLを取得するエンドポイント
$getPresignedUrlApi = "https://{CloudFrontのドメイン名(.cloudfront.netで終わる)}/prod/presigned-url"

# 画像処理と結果取得をリクエストするエンドポイント
$processImageApi = "https://{CloudFrontのドメイン名(.cloudfront.netで終わる)}/prod/result"

# リクエストデータを作成
$jsonData = @{
   filename = $fileName
} | ConvertTo-Json

try {
   # プリサインドURLを取得するリクエストを送信
   $response = Invoke-RestMethod -Uri $getPresignedUrlApi -Method Post -Headers @{ "x-api-key" = $apiKey } -ContentType "application/json" -Body $jsonData

   # プリサインドURLとファイル名を取得
   $uploadUrl = $response.UploadUrl

   if (-not $uploadUrl) {
      throw "Failed to retrieve presigned URL."
   }

   # 画像ファイルを読み込み
   $fileContent = [IO.File]::ReadAllBytes($imagePath)

   # 画像を一度にアップロード
   Invoke-RestMethod -Uri $uploadUrl -Method Put -ContentType "image/jpeg" -Body $fileContent
   Write-Output "Image uploaded successfully to S3"

   # 画像処理と結果取得をリクエスト
   $response = Invoke-RestMethod -Uri $processImageApi -Method Post -Headers @{ "x-api-key" = $apiKey } -ContentType "application/json" -Body $jsonData

   # 結果を表示
   Write-Output "Processing result:"
   Write-Output $response
}
catch {
   Write-Output "Error occurred: $_"
}