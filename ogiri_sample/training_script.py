import csv
import json
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


# データセットクラスの定義
class OgiriDataset(Dataset):
    def __init__(self, csv_file):
        self.data = self._load_data_from_csv(csv_file)
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
        item = self.data[idx]
        expected_result = item["expected_result"]
        labels = item["labels"]
        confidences = item["confidences"]
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


# データの前処理とトークナイズ
def preprocess_and_tokenize(dataset, tokenizer):
    inputs, targets = [], []

    for input_text, expected_result in dataset:
        inputs.append(input_text)
        targets.append(expected_result)

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"], max_length=512, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            examples["target"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data = pd.DataFrame({"input": inputs, "target": targets})
    tokenized_data = data.apply(tokenize_function, axis=1)
    return tokenized_data


# トレーニング関数の修正
def train():
    # トレーニングデータのCSVファイルパスを設定
    csv_file_path = "./augmented_data.csv"

    batch_size = 8
    num_epochs = 3
    learning_rate = 0.00005

    print("データセットとトークナイズを開始しました。")

    # データセットとトークナイズ
    dataset = OgiriDataset(csv_file_path)
    tokenizer = AutoTokenizer.from_pretrained(
        "rinna/japanese-gpt2-medium", use_fast=False
    )
    tokenizer.do_lower_case = True
    tokenized_data = preprocess_and_tokenize(dataset, tokenizer)

    print("データセットとトークナイズを終了しました。")

    # モデルの定義
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

    print("モデルの定義を完了しました。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DataCollatorの設定
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # トレーニングの設定
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        logging_dir="./logs",
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

    # トレーニングの実行
    print("トレーニングを開始します")
    trainer.train()

    # モデルの保存
    model.save_pretrained("./output/gpt2_finetuned")
    tokenizer.save_pretrained("./output/gpt2_finetuned")
    print("トレーニングが完了し、モデルを保存しました")


if __name__ == "__main__":
    train()
