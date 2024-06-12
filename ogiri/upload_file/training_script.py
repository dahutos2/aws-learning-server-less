import subprocess
import sys

# requirements.txtをインストール
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "/opt/ml/code/requirements.txt"]
)

import logging
import boto3
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
from model import EncoderCNN, DecoderRNN

# CloudWatch Logsの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定数の定義
EMBEDDING_DIM = 256


# データセットクラスの定義
class OgiriDataset(Dataset):
    def __init__(self, training_data_dir, dynamodb_table_name, dynamodb_client):
        self.training_data_dir = training_data_dir
        self.dynamodb_table_name = dynamodb_table_name
        self.dynamodb_client = dynamodb_client
        # DynamoDBからデータをロード
        self.data = self._load_data_from_dynamodb()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # 画像サイズの変更
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),  # 正規化
            ]
        )
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
        image_key = item["ImageKey"]["S"]
        image_path = os.path.join(self.training_data_dir, image_key)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        expected_result = item["ExpectedResult"]["S"]
        labels = item["Labels"]["L"]  # Rekognitionのラベル情報
        label_texts, confidences = self.extract_labels_and_confidences(labels)

        return image, expected_result, label_texts, confidences

    def extract_labels_and_confidences(self, labels):
        label_texts = [label["M"]["Name"]["S"] for label in labels]
        confidences = [float(label["M"]["Confidence"]["N"]) for label in labels]
        return label_texts, confidences


# カスタムcollate関数の定義
def collate_fn(batch):
    # 画像データをバッチとして結合
    images, expected_results, label_texts, confidences = zip(*batch)
    images = torch.stack(images, 0)

    # パディングを使用して confidences を揃える
    max_len = max([len(conf) for conf in confidences])
    padded_confidences = [conf + [0.0] * (max_len - len(conf)) for conf in confidences]
    padded_confidences = torch.tensor(padded_confidences)

    # その他のデータをリスト形式で保持
    return images, list(expected_results), list(label_texts), padded_confidences


# トレーニング関数の修正
def train(args):
    # トレーニングデータのディレクトリとDynamoDBテーブル名を設定
    training_data_dir = "/opt/ml/input/data/training"
    dynamodb_table_name = os.environ["DYNAMODB_TABLE_NAME"]

    # デフォルトでは、東京リージョンを使用する
    region_name = os.environ.get("AWS_REGION", "ap-northeast-1")
    dynamodb_client = boto3.client("dynamodb", region_name=region_name)

    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    # データセットとデータローダの設定
    dataset = OgiriDataset(training_data_dir, dynamodb_table_name, dynamodb_client)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # モデルの定義
    encoder = EncoderCNN(EMBEDDING_DIM)
    decoder = DecoderRNN(EMBEDDING_DIM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    decoder.to(device)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # 最適化関数の定義
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
    )

    # トレーニング開始
    logger.info("トレーニングを開始します")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} started")
        for i, (
            images,
            expected_results,
            label_texts,
            confidences,
        ) in enumerate(dataloader):
            images = images.to(device)

            # 特徴量を抽出
            features = encoder(images)

            # ラベルテキストをトークナイズしてインデックス化
            label_indices = [
                decoder.tokenizer.encode(text, add_special_tokens=False)
                for sublist in label_texts
                for text in sublist
            ]
            # ラベルのインデックスをトークンの埋め込みに変換し、デバイスに移動
            label_embeddings = decoder.model.transformer.wte(
                torch.tensor(label_indices).to(device)
            )

            # 信頼度をテンソルに変換し、最大ラベル長にパディングしてラベル埋め込みのサイズに拡張
            max_label_len = label_embeddings.size(1)
            confidences_tensor = confidences.to(device).unsqueeze(-1)
            if confidences_tensor.size(1) < max_label_len:
                padded_confidences = torch.zeros(
                    confidences_tensor.size(0), max_label_len, device=device
                )
                padded_confidences[:, : confidences_tensor.size(1)] = confidences_tensor
                confidences_tensor = padded_confidences.unsqueeze(-1).expand(
                    -1, max_label_len, label_embeddings.size(2)
                )
            else:
                confidences_tensor = confidences_tensor.expand(
                    -1, max_label_len, label_embeddings.size(2)
                )

            # 特徴量、ラベル埋め込み、信頼度、検出テキスト埋め込みを結合
            combined_features = torch.cat(
                (features, label_embeddings, confidences_tensor), dim=1
            )
            combined_features = decoder.fc(combined_features)
            optimizer.zero_grad()

            # デコーダーの出力を取得
            outputs = decoder(
                combined_features, expected_results, label_texts, confidences
            )
            inputs = decoder.tokenizer(
                expected_results, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(device)

            # 損失を計算
            loss = criterion(
                outputs.logits.view(-1, outputs.logits.size(-1)), inputs.view(-1)
            )
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

    # モデルの保存
    torch.save(encoder.state_dict(), "/opt/ml/model/encoder.ckpt")
    torch.save(decoder.state_dict(), "/opt/ml/model/decoder.ckpt")
    logger.info("トレーニングが完了し、モデルを保存しました")


if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    train(args)
