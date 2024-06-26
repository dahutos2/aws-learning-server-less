import json
import boto3
from botocore.exceptions import ClientError
import logging
from googletrans import Translator
from decimal import Decimal

# CloudWatch Logs の設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
rekognition = boto3.client("rekognition")
sagemaker_runtime = boto3.client("sagemaker-runtime")

# Google翻訳のインスタンスを作成
translator = Translator()
MAX_LENGTH = 3


def translate_labels_to_japanese(labels):
    return [translator.translate(label, src="en", dest="ja").text for label in labels]


def lambda_handler(event, context):
    try:
        # リクエストボディからファイル名を取得
        body = json.loads(event["body"])
        image_key = body["filename"]

        # 公開する画像のURLを生成
        cloudfront_domain = "d1p47yp2l3zlz9.cloudfront.net"
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

        if items:
            # 既に存在するデータがあれば、最初のアイテムのラベルを使用
            labels = items[0]["Labels"]
            confidences = [
                float(confidence) for confidence in items[0].get("Confidences", [])
            ]
        else:
            # Rekognitionを使用して画像を分析
            rekognition_response = rekognition.detect_labels(
                Image={
                    "S3Object": {"Bucket": "ogiri-images-bucket", "Name": image_key}
                },
                MaxLabels=10,
            )
            en_labels = [label["Name"] for label in rekognition_response["Labels"]]
            confidences = [
                float(label["Confidence"]) for label in rekognition_response["Labels"]
            ]

            # ラベルを日本語に翻訳
            labels = translate_labels_to_japanese(en_labels)

        # SageMakerエンドポイントを呼び出して予測
        sagemaker_input = {
            "labels": labels[:MAX_LENGTH],
            "confidences": confidences[:MAX_LENGTH],
        }

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName="ogiri-endpoint",
            ContentType="application/json",
            Body=json.dumps(sagemaker_input),
        )
        result = json.loads(response["Body"].read().decode())

        # 結果を整形
        result_text = result.get("generated_text", "")

        # DynamoDBに保存
        item = {
            "ImageKey": image_key,
            "ImageUrl": image_url,
            "Labels": labels[:MAX_LENGTH],
            "Confidences": [Decimal(confidence) for confidence in confidences][
                :MAX_LENGTH
            ],
            "Result": result_text,
        }

        table.put_item(Item=item)
        return {"statusCode": 200, "body": json.dumps({"Result": result_text})}
    except ClientError as e:
        logger.error(f"クライアントエラー: {str(e)}")
        return {"statusCode": 500, "body": json.dumps(f"クライアントエラー: {str(e)}")}
    except Exception as e:
        logger.error(f"予期せぬ例外: {str(e)}")
        return {"statusCode": 500, "body": json.dumps(f"予期せぬ例外: {str(e)}")}
