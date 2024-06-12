import csv
import torch
import torch.nn as nn
import json
import io
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model import EncoderCNN, DecoderRNN
import requests

# 定数の定義
EMBEDDING_DIM = 256


# データセットクラスの定義
class OgiriDataset(Dataset):
    def __init__(self, csv_file):
        self.data = self._load_data_from_csv(csv_file)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # 画像サイズの変更
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),  # 正規化
            ]
        )
        print(f"CSVから{len(self.data)}個のデータをロードしました。")

    def _load_data_from_csv(self, csv_file):
        data = []
        with open(csv_file, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                row["labels"] = json.loads(row["labels"])
                row["confidences"] = json.loads(row["confidences"])
                data.append(row)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 画像とラベルを取得
        item = self.data[idx]
        image_url = item["image_url"]
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image = self.transform(image)
        expected_result = item["expected_result"]
        label_texts = item["labels"]
        confidences = item["confidences"]

        return image, expected_result, label_texts, confidences


# カスタムcollate関数の定義
def collate_fn(batch):
    # 画像データをバッチとして結合
    images, expected_results, label_texts, confidences = zip(*batch)

    # 画像テンソルを結合
    images = torch.stack(images, 0)

    # パディングを使用して confidences を揃える
    max_len = max([len(conf) for conf in confidences])
    padded_confidences = torch.zeros((len(confidences), max_len))
    for i, conf in enumerate(confidences):
        padded_confidences[i, : len(conf)] = torch.tensor(conf)

    return images, list(expected_results), list(label_texts), padded_confidences


# トレーニング関数の修正
def train():
    # トレーニングデータのCSVファイルパスを設定
    csv_file_path = "./augmented_data.csv"

    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001

    print("データセットとデータローダの設定を開始しました。")

    # データセットとデータローダの設定
    dataset = OgiriDataset(csv_file_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    print("データセットとデータローダの設定を終了しました。")

    # モデルの定義
    encoder = EncoderCNN(EMBEDDING_DIM)
    decoder = DecoderRNN(EMBEDDING_DIM)

    print("モデルの定義を完了しました。")
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
    print("トレーニングを開始します")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started")
        for i, (
            images,
            expected_results,
            label_texts,
            confidences,
        ) in enumerate(dataloader):
            images = images.to(device)

            # 特徴量を抽出
            features = encoder(images)

            print(f"expected_results {expected_results}, len {len(expected_results)}")
            print(f"label_texts {label_texts}, len {len(label_texts)}")
            print(f"confidences {len(confidences)}")

            print(f"features {len(features)}")

            # ラベルテキストをトークナイズしてインデックス化
            label_embeddings = []
            for sublist in label_texts:
                sublist_embeddings = decoder.model.transformer.wte(
                    torch.tensor(
                        decoder.tokenizer.encode(
                            " ".join(sublist), add_special_tokens=False
                        )
                    ).to(device)
                )
                label_embeddings.append(sublist_embeddings)

            print(f"label_embeddings {len(label_embeddings)}")

            # 信頼度をパディングして拡張
            max_label_len = max([len(emb) for emb in label_texts])
            padded_confidences = torch.zeros((len(confidences), max_label_len)).to(
                device
            )
            for j, conf in enumerate(confidences):
                padded_confidences[j, : len(conf)] = (
                    torch.tensor(conf).clone().detach().to(device)
                )

            print(f"padded_confidences {len(padded_confidences)}")

            # 信頼度のサイズをラベル埋め込みのサイズに拡張
            expanded_confidences = []
            for j, emb in enumerate(label_embeddings):
                conf = padded_confidences[j, : emb.size(0)]
                conf_expanded = conf.unsqueeze(1).expand(-1, emb.size(1))
                expanded_confidences.append(conf_expanded)

            print(f"expanded_confidences {len(expanded_confidences)}")

            print("features shape:", features.shape)
            print("label_embeddings shape:", [emb.shape for emb in label_embeddings])
            print("padded_confidences shape:", padded_confidences.shape)

            # 特徴量、ラベル埋め込み、信頼度を結合
            combined_features_list = []
            for j, emb in enumerate(label_embeddings):
                emb_len = emb.size(0)  # ラベル埋め込みの長さを取得

                # 特徴量テンソルをラベル埋め込みの長さに合わせて拡張
                features_expanded = features.unsqueeze(1).expand(-1, emb_len, -1)

                # 信頼度テンソルをラベル埋め込みの形状に一致させる
                conf_expanded = (
                    padded_confidences[:, :emb_len]
                    .unsqueeze(2)
                    .expand(-1, -1, emb.size(1))
                )

                # ラベル埋め込みテンソルを特徴量テンソルの形状に一致させる
                emb_expanded = emb.unsqueeze(0).expand(features.size(0), -1, -1)

                # 特徴量、ラベル埋め込み、および信頼度テンソルを結合
                combined_features_list.append(
                    torch.cat(
                        (
                            features_expanded,
                            emb_expanded,
                            conf_expanded,
                        ),
                        dim=2,
                    )
                )

            # 結果の形状確認
            for cf in combined_features_list:
                print(cf.shape)

            print(f"combined_features_list {len(combined_features_list)}")

            # 特徴量のパディングを追加
            max_len = max(emb.size(1) for emb in combined_features_list)
            padded_combined_features = []
            for emb in combined_features_list:
                if emb.size(1) < max_len:
                    padding = torch.zeros(
                        (emb.size(0), max_len - emb.size(1), emb.size(2))
                    ).to(device)
                    padded_combined_features.append(torch.cat([emb, padding], dim=1))
                else:
                    padded_combined_features.append(emb)

            combined_features = torch.cat(padded_combined_features, dim=0)
            combined_features = decoder.fc(combined_features)

            print(f"combined_features {len(combined_features)}")

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
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

    # モデルの保存
    torch.save(encoder.state_dict(), "./output/encoder.ckpt")
    torch.save(decoder.state_dict(), "./output/decoder.ckpt")
    print("トレーニングが完了し、モデルを保存しました")


if __name__ == "__main__":
    train()
