import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_path, tokenizer_path):
    # トークナイザーの初期化時に `padding_side='left'` を設定
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizer.padding_side = "left"  # ここで左パディングを設定
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model


def generate_response(
    tokenizer, model, labels, confidences, max_new_tokens=50, top_k=50, top_p=0.95
):
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

    # トークナイザーで入力テキストをトークン化
    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # モデルを使用して出力を生成
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            top_k=top_k,  # トップKサンプリングを使用
            top_p=top_p,  # トップPサンプリングを使用
            do_sample=True,  # サンプリングを有効にする
        )

    # 出力トークンをデコードしてテキストに変換
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 「答え:」以降の部分を抽出して返す
    response = response.split("答え:")[1].strip() if "答え:" in response else response
    return response


def main():
    # ファインチューニングされたモデルとトークナイザーのパス
    model_path = "./output/gpt2_finetuned"
    tokenizer_path = "./output/gpt2_finetuned"

    # モデルとトークナイザーの読み込み
    tokenizer, model = load_model_and_tokenizer(model_path, tokenizer_path)

    # 推論するラベルと信頼度
    labels = ["犬", "猫", "喧嘩"]
    confidences = [0.96, 0.88, 0.87]

    # 応答の生成
    response = generate_response(tokenizer, model, labels, confidences)
    print("入力:", labels, confidences)
    print("出力:", response)


if __name__ == "__main__":
    main()
