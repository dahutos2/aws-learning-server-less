import json
import boto3
import csv
import requests
from datetime import datetime
import uuid
from botocore.exceptions import NoCredentialsError, ClientError
import logging
from decimal import Decimal

# CloudWatch Logsの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
rekognition = boto3.client("rekognition")
dynamodb = boto3.resource("dynamodb")
sagemaker = boto3.client("sagemaker")
events = boto3.client("events")


def lambda_handler(event, context):
    try:
        # イベントからバケット名とオブジェクトキーを取得
        bucket = event.get("bucket")
        key = event.get("key")

        if not bucket or not key:
            raise ValueError("バケットとキーはイベントで指定してください。")

        # S3からCSVファイルを取得
        csv_file = s3.get_object(Bucket=bucket, Key=key)
        csv_content = csv_file["Body"].read().decode("utf-8").splitlines()
        csv_reader = csv.reader(csv_content)

        # 先頭行(タイトル行)がスキップする
        next(csv_reader)

        # DynamoDBに学習用のデータを登録する
        table = dynamodb.Table("OgiriTrainingDataTable")

        logger.info("学習用のデータの登録を開始します。")
        for row in csv_reader:
            image_url, expected_result = row

            try:
                image_key = image_url.split("/")[-1]
                image_path = f"images/{image_key}"  # images ディレクトリに配置

                # DynamoDBに既にデータが存在するか確認
                response = table.get_item(
                    Key={"ImageKey": image_key, "ExpectedResult": expected_result}
                )
                if "Item" in response:
                    logger.info(f"{image_url}は既に処理されています。")
                    continue

                # Rekognitionを使用して画像を分析する
                rekognition_labels = None

                # 同じ画像が存在するかを確認する
                labels_response = table.query(
                    KeyConditionExpression=boto3.dynamodb.conditions.Key("ImageKey").eq(
                        image_key
                    )
                )

                # 同じ画像が存在する場合は、DynamoDBの分析結果を使用する
                if labels_response["Items"]:
                    logger.info(f"{image_key}は画像のみ登録済みです。")
                    rekognition_labels = labels_response["Items"][0]["Labels"]
                else:
                    # 画像が存在しない場合は登録する

                    # グローバルなURLから画像をダウンロード
                    image_data = requests.get(image_url).content

                    # ダウンロードした画像をS3にアップロード
                    s3.put_object(Bucket=bucket, Key=image_path, Body=image_data)
                    logger.info(f"S3に{image_key}を登録しました。")

                    # 画像を分析
                    rekognition_response = rekognition.detect_labels(
                        Image={"S3Object": {"Bucket": bucket, "Name": image_path}},
                        MaxLabels=10,
                    )
                    rekognition_labels = [
                        {
                            "Name": label["Name"],
                            "Confidence": Decimal(str(label["Confidence"])),
                        }
                        for label in rekognition_response["Labels"]
                    ]

                # DynamoDBに画像URLと期待結果を保存
                table = dynamodb.Table("OgiriTrainingDataTable")
                item = {
                    "ImageKey": image_key,
                    "ExpectedResult": expected_result,
                    "Labels": rekognition_labels,
                }

                table.put_item(Item=item)
                logger.info(f"{image_key}をDBに登録しました。")

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"{image_url}からの画像のダウロードに失敗しました: {str(e)}"
                )
            except ClientError as e:
                logger.error(f"{image_url}の登録に失敗しました: {str(e)}")

        # SageMakerトレーニングジョブの作成
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        training_job_name = f"ogiri-training-job-{now}"
        response = sagemaker.create_training_job(
            TrainingJobName=training_job_name,
            HyperParameters={
                "batch_size": "32",
                "epochs": "10",
                "learning_rate": "0.001",
                "sagemaker_submit_directory": "s3://ogiri-training-data-bucket/training-code/training_code.tar.gz",
            },
            AlgorithmSpecification={
                "TrainingImage": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310",
                "MetricDefinitions": [
                    {"Name": "validation:error", "Regex": "validation:error=(.*)"}
                ],
                "TrainingInputMode": "File",
                "EnableSageMakerMetricsTimeSeries": True,
            },
            RoleArn="arn:aws:iam::765231401377:role/SageMakerOgiriTrainingJobRole",
            InputDataConfig=[
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://ogiri-training-data-bucket/images/",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "application/json",
                    "InputMode": "File",
                }
            ],
            OutputDataConfig={
                "S3OutputPath": "s3://ogiri-training-data-bucket/output/"
            },
            ResourceConfig={
                "InstanceType": "ml.m5.large",
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 86400},
            Environment={
                "DYNAMODB_TABLE_NAME": "OgiriTrainingDataTable",
                "AWS_REGION": "ap-northeast-1",
                "SAGEMAKER_PROGRAM": "training_script.py",
            },
        )

        logger.info("トレーニングジョブの開始に成功しました。")

        # CloudWatch Event ルールを動的に作成
        rule_name = f"OgiriTrainingJobCompletionRule-{now}"
        event_pattern = {
            "source": ["aws.sagemaker"],
            "detail-type": ["SageMaker Training Job State Change"],
            "detail": {
                "TrainingJobName": [training_job_name],
                "TrainingJobStatus": ["Completed"],
            },
        }

        # ルールを作成
        events.put_rule(
            Name=rule_name, EventPattern=json.dumps(event_pattern), State="ENABLED"
        )

        # ルールにターゲットとしてLambda関数を追加
        target_id = f"target-{uuid.uuid4()}"
        events.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    "Id": target_id,
                    "Arn": "arn:aws:lambda:ap-northeast-1:765231401377:function:EndOgiriTrainingJob",
                }
            ],
        )

        logger.info(
            f"CloudWatch Event Rule {rule_name} を作成し、Lambdaターゲットを設定しました。"
        )

        return {
            "statusCode": 200,
            "body": json.dumps("トレーニングジョブの開始に成功しました。"),
        }
    except NoCredentialsError:
        logger.error("権限がないです。")
        return {"statusCode": 403, "body": "権限がないです。"}
    except Exception as e:
        logger.error(f"予期せぬ例外: {str(e)}")
        return {"statusCode": 500, "body": json.dumps(f"予期せぬ例外: {str(e)}")}
