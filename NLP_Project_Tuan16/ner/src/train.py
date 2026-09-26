import numpy as np
# Gọi trực tiếp các công thức đo điểm từ thư viện seqeval để tránh xung đột tên file
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
# Nạp dữ liệu từ data.py
from data import load_and_prepare_data 

def main():
    print("1. Đang nạp dữ liệu từ data.py...")
    tokenized_datasets, tokenizer, label_list = load_and_prepare_data()

    # Lấy 500 câu để chạy thử nghiệm nhanh vòng lặp học tập
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

    print("2. Đang khởi tạo mô hình PhoBERT...")
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Hàm tính điểm F1 dùng trực tiếp thư viện seqeval
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Lọc bỏ các nhãn -100 sinh ra do băm nhỏ từ
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    # 3. Thiết lập thông số huấn luyện
    training_args = TrainingArguments(
        output_dir="../results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,    # Đã cập nhật theo phiên bản mới
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("3. Bắt đầu huấn luyện...")
    trainer.train()
    
    print("4. Chấm điểm mô hình...")
    metrics = trainer.evaluate()
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(metrics)

if __name__ == "__main__":
    main()