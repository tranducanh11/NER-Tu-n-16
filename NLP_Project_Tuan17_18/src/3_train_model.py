from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

def main():
    print("1. Đang chuẩn bị dữ liệu và Tokenizer...")
    dataset = load_dataset("knkarthick/samsum")
    checkpoint = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Hàm tiền xử lý (gom lại từ bước trước để nạp thẳng vào Trainer)
    prefix = "summarize: "
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("2. Đang xử lý dữ liệu đầu vào...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    print("3. Đang tải kiến trúc mô hình T5-small...")
    # Khác với NER, bài toán sinh văn bản dùng AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # DataCollator đặc thù cho Seq2Seq: tự động đệm (padding) input và target cho bằng nhau
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    print("4. Khởi tạo cấu hình huấn luyện...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="../results_t5",
        eval_strategy="epoch",         # Đánh giá sau mỗi epoch
        learning_rate=2e-5,
        per_device_train_batch_size=4, # Đặt size nhỏ (4) để tránh tràn bộ nhớ card màn hình
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,            # Chỉ lưu 3 bản sao mới nhất để nhẹ ổ cứng
        num_train_epochs=3,            # Huấn luyện 3 vòng
        predict_with_generate=True,    # BẮT BUỘC bật cho bài toán sinh văn bản (Text Generation)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("5. BẮT ĐẦU HUẤN LUYỆN (Quá trình này có thể mất thời gian)...")
    trainer.train()
    
    print("6. ĐANG LƯU MÔ HÌNH HOÀN THIỆN...")
    trainer.save_model("../models/t5_samsum_final")
    print("Hoàn tất! Mô hình đã được lưu tại thư mục 'models/t5_samsum_final'.")

if __name__ == "__main__":
    main()