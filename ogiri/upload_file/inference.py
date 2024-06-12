import subprocess
import sys

# requirements.txtをインストール
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "/opt/ml/code/requirements.txt"]
)

import logging
import torch
import os
import json
import base64
import io
from PIL import Image
import torchvision.transforms as transforms
from model import EncoderCNN, DecoderRNN

# CloudWatch Logsの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定数の定義
EMBEDDING_DIM = 256


# モデルのロード
def model_fn(model_dir):
    logger.info("モデルのロード開始")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # エンコーダのロード
    encoder = EncoderCNN(EMBEDDING_DIM)
    encoder_path = os.path.join(model_dir, "encoder.ckpt")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    logger.info(f"エンコーダのロード完了: {encoder_path}")

    # デコーダのロード
    decoder = DecoderRNN(EMBEDDING_DIM)
    decoder_path = os.path.join(model_dir, "decoder.ckpt")
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.to(device)
    decoder.eval()
    logger.info(f"デコーダのロード完了: {decoder_path}")

    return encoder, decoder


# リクエストの前処理
def input_fn(request_body, request_content_type):
    logger.info("リクエストの前処理開始")
    if request_content_type == "application/json":
        request = json.loads(request_body)
        image_data = base64.b64decode(request["image_data"])
        labels = request["labels"]
        confidences = request["confidences"]
        logger.info("リクエストの前処理完了")
        return image_data, labels, confidences
    else:
        error_message = f"予期せぬ入力形式: {request_content_type}"
        logger.error(error_message)
        raise ValueError(error_message)


# 推論の実行
def predict_fn(input_data, model):
    logger.info("推論の実行開始")
    image_data, labels, confidences = input_data
    encoder, decoder = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 画像データをテンソルに変換し、デバイスに移動
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image = transform(image).unsqueeze(0).to(device)
    logger.info("画像データの前処理完了")

    # 画像から特徴を抽出
    features = encoder(image)
    logger.info("画像特徴の抽出完了")

    # ラベルと信頼度をエンコード
    label_texts, confidences = labels, confidences
    label_indices = decoder.tokenizer.encode(
        " ".join(label_texts), add_special_tokens=False
    )

    # ラベルのインデックスをトークンの埋め込みに変換し、デバイスに移動
    label_embeddings = decoder.model.transformer.wte(
        torch.tensor(label_indices).to(device)
    )
    # 信頼度をテンソルに変換し、ラベル埋め込みのサイズに拡張
    confidences = (
        torch.tensor(confidences)
        .to(device)
        .unsqueeze(1)
        .expand(-1, label_embeddings.size(1))
    )

    # 画像特徴、ラベル埋め込み、信頼度、テキスト埋め込みを結合
    combined_features = torch.cat((features, label_embeddings, confidences), dim=1)
    combined_features = decoder.fc(combined_features)
    logger.info("特徴の結合完了")

    # 推論を実行
    input_ids = decoder.tokenizer.encode("大喜利:", return_tensors="pt").to(device)
    outputs = decoder.model.generate(
        input_ids=input_ids,
        inputs_embeds=combined_features,
        max_length=50,
        num_return_sequences=1,
    )
    result = decoder.tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("推論完了")
    return result


# 応答の後処理
def output_fn(prediction, content_type):
    logger.info("応答の後処理開始")
    if content_type == "application/json":
        response = json.dumps({"generated_text": prediction})
        logger.info("応答の後処理完了")
        return response
    else:
        error_message = f"予期せぬ入力形式: {content_type}"
        logger.error(error_message)
        raise ValueError(error_message)
