import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# 画像特徴抽出モデル（ResNet）
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # 事前学習済みのResNet50モデルを読み込む
        resnet = models.resnet50(pretrained=True)
        # 最後の全結合層を除去
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # 埋め込み層
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images):
        # 画像から特徴を抽出
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        features = self.dropout(features)
        return features


# GPT-2を使ったキャプション生成
class DecoderRNN(nn.Module):
    def __init__(self, embed_size):
        super(DecoderRNN, self).__init__()
        # GPT-2トークナイザとモデルの読み込み
        self.tokenizer = GPT2Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        self.model = GPT2LMHeadModel.from_pretrained("rinna/japanese-gpt2-medium")
        self.fc = nn.Linear(
            embed_size * 2, embed_size
        )  # 画像特徴、ラベル、信頼度、テキストの結合

    def forward(self, features, captions, label_texts, confidences, detected_text):
        # 文字列をエンコード
        label_indices = self.tokenizer.encode(
            " ".join(label_texts), add_special_tokens=False
        )
        text_indices = self.tokenizer.encode(detected_text, add_special_tokens=False)

        # キャプションをトークナイズし、テンソル形式に変換
        inputs = self.tokenizer(
            captions, return_tensors="pt", padding=True, truncation=True
        )
        # ラベルのインデックスをトークンの埋め込みに変換し、デバイスに移動
        label_embeddings = self.model.transformer.wte(
            torch.tensor(label_indices).to(features.device)
        )
        # 信頼度をテンソルに変換し、ラベル埋め込みのサイズに拡張
        confidences = (
            torch.tensor(confidences)
            .to(features.device)
            .unsqueeze(1)
            .expand(-1, label_embeddings.size(1))
        )
        # 検出されたテキストのインデックスをトークンの埋め込みに変換し、デバイスに移動
        text_embeddings = self.model.transformer.wte(
            torch.tensor(text_indices).to(features.device)
        )
        # 画像特徴、ラベル埋め込み、信頼度、テキスト埋め込みを結合
        combined_features = torch.cat(
            (features, label_embeddings, confidences, text_embeddings), dim=1
        )

        # 結合された特徴を全結合層に入力
        combined_features = self.fc(combined_features)
        # トークナイズされたキャプションを入力にして、モデルで前向き伝播を実行
        outputs = self.model(
            inputs_embeds=combined_features, labels=inputs["input_ids"]
        )
        # 損失と出力ロジットを返す
        return outputs.loss, outputs.logits

    def sample(self, features, label_texts, confidences, detected_text):
        # 文字列をエンコード
        label_indices = self.tokenizer.encode(
            " ".join(label_texts), add_special_tokens=False
        )
        text_indices = self.tokenizer.encode(detected_text, add_special_tokens=False)

        # 特徴から大喜利のテキストを生成
        # 初期シーケンスとして「大喜利:」をエンコードし、テンソル形式に変換してデバイスに移動
        input_ids = self.tokenizer.encode("大喜利:", return_tensors="pt").to(
            features.device
        )
        # ラベルのインデックスをトークンの埋め込みに変換し、デバイスに移動
        label_embeddings = self.model.transformer.wte(
            torch.tensor(label_indices).to(features.device)
        )
        # 信頼度をテンソルに変換し、ラベル埋め込みのサイズに拡張
        confidences = (
            torch.tensor(confidences)
            .to(features.device)
            .unsqueeze(1)
            .expand(-1, label_embeddings.size(1))
        )
        # 検出されたテキストのインデックスをトークンの埋め込みに変換し、デバイスに移動
        text_embeddings = self.model.transformer.wte(
            torch.tensor(text_indices).to(features.device)
        )
        # 画像特徴、ラベル埋め込み、信頼度、テキスト埋め込みを結合
        combined_features = torch.cat(
            (features, label_embeddings, confidences, text_embeddings), dim=1
        )

        # 結合された特徴を全結合層に入力
        combined_features = self.fc(combined_features)
        # input_ids を使用してテキスト生成
        outputs = self.model.generate(
            input_ids=input_ids, max_length=50, num_return_sequences=1
        )
        # 生成されたテキストをデコードして返す
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
