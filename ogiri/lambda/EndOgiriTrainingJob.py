import json
import boto3
import logging

# CloudWatch Logsの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sagemaker = boto3.client("sagemaker")


def lambda_handler(event, context):
    try:
        training_job_name = event["detail"]["TrainingJobName"]

        # トレーニングジョブの完了を確認
        if event["detail"]["TrainingJobStatus"] != "Completed":
            logger.error(
                f"トレーニングジョブ{training_job_name}が正常に完了しませんでした。"
            )
            return {
                "statusCode": 400,
                "body": json.dumps("トレーニングが正常に完了しませんでした。"),
            }

        # モデルの作成
        model_name = "ogiri-model"
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.0.0-gpu-py310",
                "ModelDataUrl": f"s3://ogiri-training-data-bucket/output/{training_job_name}/output/model.tar.gz",
                "Environment": {
                    "SAGEMAKER_SUBMIT_DIRECTORY": "s3://ogiri-training-data-bucket/training-code/training_code.tar.gz"
                },
                "entry_point": "inference.py",
            },
            ExecutionRoleArn="arn:aws:iam::765231401377:role/SageMakerOgiriTrainingJobRole",
        )
        logger.info("モデルの作成に成功しました。")

        # エンドポイント構成の作成
        endpoint_config_name = "ogiri-endpoint-config"
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": "ml.m5.large",
                    "InitialInstanceCount": 1,
                }
            ],
        )
        logger.info("エンドポイント構成の作成に成功しました。")

        # エンドポイントの作成
        endpoint_name = "ogiri-endpoint"
        sagemaker.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        logger.info("エンドポイントの作成に成功しました。")

        return {
            "statusCode": 200,
            "body": json.dumps("モデルとエンドポイントの作成に成功しました。"),
        }
    except Exception as e:
        logger.error(f"予期せぬ例外: {str(e)}")
        return {"statusCode": 500, "body": json.dumps(f"予期せぬ例外: {str(e)}")}
