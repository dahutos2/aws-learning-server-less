import subprocess
import sys
import shutil

# requirements.txtをインストール
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "/opt/ml/code/requirements.txt"]
)

import logging
import boto3
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# CloudWatch Logsの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# データセットクラスの定義
class OgiriDataset(Dataset):
    def __init__(self, dynamodb_table_name, dynamodb_client):
        self.dynamodb_table_name = dynamodb_table_name
        self.dynamodb_client = dynamodb_client

        # DynamoDBからデータをロード
        self.data = self._load_data_from_dynamodb()
        logger.info(f"DynamoDBから{len(self.data)}個のデータをロードしました。")

    def _load_data_from_dynamodb(self):
        # DynamoDBのデータをページネーションで取得
        paginator = self.dynamodb_client.get_paginator("scan")
        response_iterator = paginator.paginate(TableName=self.dynamodb_table_name)
        data = []
        for page in response_iterator:
            data.extend(page["Items"])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 画像とラベルを取得
        item = self.data[idx]
        expected_result = item["ExpectedResult"]["S"]
        labels = item["Labels"]["L"]
        confidences = item["Confidences"]["L"]
        labels, confidences = self.extract_labels_and_confidences(labels, confidences)
        elements = "\n".join(
            [
                f"- {label}: 信頼度{confidence:.2f}"
                for label, confidence in zip(labels, confidences)
            ]
        )
        input_text = (
            "以下は、画像内に写っている要素とその信頼度です。\n"
            "この情報を元に、一言で面白い大喜利をしてください。\n"
            f"{elements}\n"
            "答え: "
        )
        return input_text, expected_result

    def extract_labels_and_confidences(self, labels, confidences):
        labels = [label["S"] for label in labels]
        confidences = [float(confidence["N"]) for confidence in confidences]
        return labels, confidences


# データの前処理とトークナイズ
def preprocess_and_tokenize(dataset, tokenizer):
    inputs, targets = [], []
    for input_text, expected_result in dataset:
        inputs.append(input_text)
        targets.append(expected_result)

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"], max_length=128, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data = pd.DataFrame({"input": inputs, "target": targets})
    tokenized_data = data.apply(tokenize_function, axis=1)
    return tokenized_data


# トレーニング関数の修正
def train(args):
    dynamodb_table_name = os.environ["DYNAMODB_TABLE_NAME"]

    # デフォルトでは、東京リージョンを使用する
    region_name = os.environ.get("AWS_REGION", "ap-northeast-1")
    dynamodb_client = boto3.client("dynamodb", region_name=region_name)

    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    # データセットとトークナイズ
    dataset = OgiriDataset(dynamodb_table_name, dynamodb_client)
    tokenizer = AutoTokenizer.from_pretrained(
        "rinna/japanese-gpt2-medium", use_fast=False
    )
    tokenizer.do_lower_case = True
    tokenized_data = preprocess_and_tokenize(dataset, tokenizer)
    logger.info("データセットとトークナイズの読み込みを終了しました。")

    # モデルの定義
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("モデルの読み込みを終了しました。")

    # DataCollatorの設定
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # トレーニングの設定
    training_args = TrainingArguments(
        output_dir="/opt/ml/checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        logging_dir="/opt/ml/logs",
        logging_steps=500,
        save_steps=1000,
        save_total_limit=5,
        fp16=True if torch.cuda.is_available() else False,
    )

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data,  # データが少ない場合、trainと同じデータを使用
        data_collator=data_collator,
    )

    logger.info("トレーニングを開始しました。")
    trainer.train()

    # モデルの保存
    model.save_pretrained("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")

    # inference.py と requirements.txt を /opt/ml/model ディレクトリにコピー
    shutil.copy("/opt/ml/code/inference.py", "/opt/ml/model/inference.py")
    shutil.copy("/opt/ml/code/requirements.txt", "/opt/ml/model/requirements.txt")
    logger.info("トレーニングが完了し、モデルを保存しました")


if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    train(args)
