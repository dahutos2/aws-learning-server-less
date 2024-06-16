import subprocess
import sys

# requirements.txtをインストール
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "/opt/ml/code/requirements.txt"]
)

import logging
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CloudWatch Logsの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# モデルのロード
def model_fn(model_dir):
    logger.info("モデルのロード開始")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    tokenizer.padding_side = "left"  # ここで左パディングを設定
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("モデルのロード完了")
    return tokenizer, model


# リクエストの前処理
def input_fn(request_body, request_content_type):
    logger.info("リクエストの前処理開始")
    if request_content_type == "application/json":
        request = json.loads(request_body)
        labels = request["labels"]
        confidences = request["confidences"]
        # 入力形式を学習時と同様に整形
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
        logger.info("リクエストの前処理完了")
        return input_text
    else:
        error_message = f"予期せぬ入力形式: {request_content_type}"
        logger.error(error_message)
        raise ValueError(error_message)


# 推論の実行
def predict_fn(input_data, model):
    logger.info("推論の実行開始")
    tokenizer, model = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # トークナイザーで入力テキストをトークン化
    inputs = tokenizer(
        input_data, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 推論を実行
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            top_k=50,  # トップKサンプリングを使用
            top_p=0.95,  # トップPサンプリングを使用
            do_sample=True,  # サンプリングを有効にする
        )

    # 出力トークンをデコードしてテキストに変換
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 「答え:」以降の部分を抽出して返す
    response = response.split("答え:")[1].strip() if "答え:" in response else response
    logger.info("推論の実行完了")
    return response


# 応答の後処理
def output_fn(prediction, response_content_type):
    logger.info("応答の後処理開始")
    if response_content_type == "application/json":
        response = json.dumps({"generated_text": prediction})
        logger.info("応答の後処理完了")
        return response
    else:
        error_message = f"予期せぬ出力形式: {response_content_type}"
        logger.error(error_message)
        raise ValueError(error_message)
