import numpy as np
import torch
from torch import nn
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from data import load_and_prepare_data 

def main():
    print("1. Đang nạp dữ liệu từ data.py...")
    tokenized_datasets, tokenizer, label_list = load_and_prepare_data()

    # Sử dụng toàn bộ dữ liệu để huấn luyện Baseline Cải tiến
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

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

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

    # ==========================================
    # KỸ THUẬT CẢI TIẾN: CUSTOM TRAINER
    # ==========================================
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Lấy tổng số nhãn
            num_labels = logits.shape[-1]
            
            # Khởi tạo ma trận trọng số: Mặc định là 1.0
            device = model.device
            loss_weights = torch.ones(num_labels, device=device)
            
            # Tìm vị trí của nhãn 'O' (chữ bình thường) và ép trọng số xuống 0.1
            # Các nhãn thực thể (PER, ORG, LOC) giữ nguyên trọng số cao để ép AI phải tập trung học
            if 'O' in label2id:
                o_index = label2id['O']
                loss_weights[o_index] = 0.1
                
            loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
            
            if labels is not None:
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            else:
                loss = outputs["loss"]
                
            return (loss, outputs) if return_outputs else loss

    # 3. Thiết lập thông số huấn luyện (Fine-tuning Hyperparameters)
    training_args = TrainingArguments(
        output_dir="../results_improved",  # Lưu sang thư mục mới để so sánh với Baseline cũ
        learning_rate=3e-5,                # Điều chỉnh Learning Rate tối ưu hơn
        per_device_train_batch_size=16,    # Tăng batch_size để AI học mượt hơn (Nếu máy báo hết RAM, hạ xuống 8)
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
    )

    # Sử dụng CustomTrainer thay cho Trainer mặc định
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("3. Bắt đầu huấn luyện mô hình Cải tiến (Custom Loss)...")
    trainer.train()
    
    print("4. Chấm điểm mô hình...")
    metrics = trainer.evaluate()
    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CẢI TIẾN ---")
    print(metrics)

if __name__ == "__main__":
    main()