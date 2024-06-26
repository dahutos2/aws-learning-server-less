import json
import boto3


def lambda_handler(event, context):
    s3_client = boto3.client("s3")
    bucket_name = "ogiri-images-bucket"

    try:
        # リクエストボディからファイル名を取得
        body = json.loads(event["body"])
        filename = body["filename"]

        # プリサインドURLを生成
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket_name,
                "Key": filename,
                "ContentType": "image/jpeg",
            },
            ExpiresIn=3600,
        )

        return {"statusCode": 200, "body": json.dumps({"UploadUrl": presigned_url})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
