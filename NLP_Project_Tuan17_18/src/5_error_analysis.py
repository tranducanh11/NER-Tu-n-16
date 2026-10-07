import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import random

def main():
    print("Đang tải dữ liệu và mô hình để phân tích lỗi...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tải lại mô hình đã fine-tune
    model_path = "../models/t5_samsum_final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
    # Tải tập test
    dataset = load_dataset("knkarthick/samsum")
    test_data = dataset["test"]
    
    # Chọn ngẫu nhiên 3 đoạn chat để phân tích
    random.seed(42) # Cố định seed để kết quả không bị đổi mỗi lần chạy
    sample_indices = random.sample(range(len(test_data)), 3)
    
    print("\n" + "="*60)
    print("PHÂN TÍCH ĐỊNH TÍNH (ERROR ANALYSIS) - 3 MẪU NGẪU NHIÊN")
    print("="*60)
    
    for i, idx in enumerate(sample_indices, 1):
        dialogue = test_data[idx]["dialogue"]
        reference = test_data[idx]["summary"]
        
        # Cho AI tóm tắt
        inputs = tokenizer("summarize: " + dialogue, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
        outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n--- MẪU SỐ {i} ---")
        print(f"🔸 [ĐOẠN CHAT GỐC]:\n{dialogue.strip()}")
        print(f"\n🔹 [CON NGƯỜI TÓM TẮT (Target)]:\n{reference}")
        print(f"🤖 [AI TÓM TẮT (Prediction)]:\n{prediction}")
        print("-" * 60)

if __name__ == "__main__":
    main()