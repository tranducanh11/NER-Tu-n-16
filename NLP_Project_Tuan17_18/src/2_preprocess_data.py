from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    print("1. Đang nạp bộ dữ liệu SAMSum...")
    dataset = load_dataset("knkarthick/samsum")
    
    print("2. Đang tải Tokenizer của google-t5/t5-small...")
    model_checkpoint = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Kỹ thuật bắt buộc với mô hình T5: Thêm task prefix
    prefix = "summarize: "
    
    # Thiết lập độ dài tối đa để tránh tràn RAM (Out of Memory)
    max_input_length = 512
    max_target_length = 128
    
    def preprocess_function(examples):
        # 1. Gắn tiền tố "summarize: " vào trước mỗi đoạn hội thoại
        inputs = [prefix + doc for doc in examples["dialogue"]]
        
        # 2. Tokenize phần Input (Đoạn chat)
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        
        # 3. Tokenize phần Target (Bản tóm tắt chuẩn)
        # Sử dụng tham số text_target để báo cho Tokenizer biết đây là nhãn (labels)
        labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)
        
        # Gán nhãn vào bộ dữ liệu đầu vào
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    print("3. Bắt đầu băm từ (Tokenize) toàn bộ 14,731 đoạn chat. Vui lòng đợi...")
    # Dùng hàm map để áp dụng Tokenize cho toàn bộ tập dataset
    # batched=True giúp xử lý song song nhiều câu cùng lúc để tăng tốc độ
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    print("\n--- HOÀN TẤT TIỀN XỬ LÝ ---")
    print(tokenized_datasets)
    
    print("\n[Các cột dữ liệu mới được tạo ra trong Train set]:")
    print(tokenized_datasets["train"].column_names)

if __name__ == "__main__":
    main()