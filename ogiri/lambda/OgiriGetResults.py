import json
import boto3
import logging

# CloudWatch Logsの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dynamodb = boto3.resource("dynamodb")


def lambda_handler(event, context):
    try:
        table_name = "OgiriResultsTable"
        table = dynamodb.Table(table_name)

        # DynamoDBから全てのデータを取得
        response = table.scan()

        if "Items" in response:
            items = response["Items"]
            results = [
                {"ImageUrl": item["ImageUrl"], "Result": item["Result"]}
                for item in items
            ]
            return {"statusCode": 200, "body": json.dumps(results)}
        else:
            return {
                "statusCode": 404,
                "body": json.dumps({"message": "データが見つかりません。"}),
            }
    except Exception as e:
        logger.error(f"予期せぬ例外: {str(e)}")
        return {"statusCode": 500, "body": json.dumps(f"予期せぬ例外: {str(e)}")}
