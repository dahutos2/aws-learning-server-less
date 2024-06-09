import json
import boto3
import base64
from botocore.exceptions import ClientError
import logging

# CloudWatch Logs の設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
rekognition = boto3.client("rekognition")
sagemaker_runtime = boto3.client("sagemaker-runtime")


def lambda_handler(event, context):
    try:
        # 画像データを取得
        body = json.loads(event["body"])
        image_data = base64.b64decode(body["image"])
        image_key = body["filename"]

        # 公開する画像のURLを生成
        cloudfront_domain = "your-cloudfront-domain.cloudfront.net"
        image_url = f"https://{cloudfront_domain}/{image_key}"

        # DynamoDBに既にデータが存在するか確認
        table = dynamodb.Table("OgiriResultsTable")
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("ImageKey").eq(
                image_key
            )
        )
        items = response.get("Items", [])
        labels = []
        confidences = []
        detected_text = ""

        if items:
            # 既に存在するデータがあれば、最初のアイテムのラベルを使用
            labels = items[0]["Labels"]
            confidences = items[0].get("Confidences", [])
            detected_text = items[0].get("DetectedText", "")
        else:
            # 画像をS3に保存
            bucket_name = "ogiri-images-bucket"
            s3.put_object(Bucket=bucket_name, Key=image_key, Body=image_data)

            # Rekognitionを使用して画像を分析
            rekognition_response = rekognition.detect_labels(
                Image={"S3Object": {"Bucket": bucket_name, "Name": image_key}},
                MaxLabels=10,
            )
            labels = [label["Name"] for label in rekognition_response["Labels"]]
            confidences = [
                label["Confidence"] for label in rekognition_response["Labels"]
            ]

            # 画像内のテキストを検出
            text_detection_response = rekognition.detect_text(
                Image={"S3Object": {"Bucket": bucket_name, "Name": image_key}}
            )
            detected_texts = [
                text["DetectedText"]
                for text in text_detection_response["TextDetections"]
            ]
            detected_text = " ".join(detected_texts) if detected_texts else ""

        # SageMakerエンドポイントを呼び出して予測
        sagemaker_input = {
            "image": base64.b64encode(image_data).decode("utf-8"),
            "labels": labels,
            "confidences": confidences,
            "detected_text": detected_text,
        }
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName="ogiri-endpoint",
            ContentType="application/json",
            Body=json.dumps(sagemaker_input),
        )

        result = json.loads(response["Body"].read().decode())

        # DynamoDBに保存
        item = {
            "ImageKey": {"S": image_key},
            "ImageUrl": {"S": image_url},
            "Labels": {"L": [{"S": label} for label in labels]},
            "Confidences": {"L": [{"N": str(conf)} for conf in confidences]},
            "Result": {"S": json.dumps(result)},
        }

        if detected_text:
            item["DetectedText"] = {"S": detected_text}

        table.put_item(Item=item)

        return {"statusCode": 200, "body": json.dumps({"result": result})}

    except ClientError as e:
        logger.error(f"クライアントエラー: {str(e)}")
        return {"statusCode": 500, "body": json.dumps(f"クライアントエラー: {str(e)}")}
    except Exception as e:
        logger.error(f"予期せぬ例外: {str(e)}")
        return {"statusCode": 500, "body": json.dumps(f"予期せぬ例外: {str(e)}")}
