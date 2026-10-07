import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate

def summarize_text(model, tokenizer, text, device):
    """Hàm suy luận (Inference) nhận vào đoạn chat và trả về bản tóm tắt"""
    # 1. Gắn tiền tố và băm từ
    inputs = ["summarize: " + text]
    input_ids = tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
    
    # 2. Đưa vào mô hình để sinh văn bản (Text Generation)
    outputs = model.generate(
        input_ids, 
        max_length=128, 
        num_beams=4,            # Khám phá 4 nhánh từ vựng để chọn ra câu hay nhất
        early_stopping=True
    )
    
    # 3. Dịch ngược các con số thành chữ tiếng Anh
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("1. ĐANG TẢI MÔ HÌNH VỪA FINE-TUNE...")
    # Trỏ đường dẫn vào thư mục đã lưu mô hình ở Bước 3
    model_path = "../models/t5_samsum_final" 
    
    # Ưu tiên dùng Card đồ họa (GPU) nếu có để tăng tốc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=> Đang sử dụng thiết bị tính toán: {device.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
    print("\n" + "="*50)
    print("2. KIỂM TRA SUY LUẬN (INFERENCE) VỚI ĐOẠN CHAT THỰC TẾ")
    print("="*50)
    # Bạn có thể thay thế bằng bất kỳ đoạn chat tiếng Anh nào
    sample_dialogue = """
    Alice: Did you send the report to the boss yet?
    Bob: Just sent it. He said we need to revise the March revenue section.
    Alice: Oh really? When is the deadline for the revision?
    Bob: This afternoon before 5 PM. Are you free to help me double-check the numbers?
    Alice: Sure, send the file over.
    """
    print("[Đoạn chat gốc]:")
    print(sample_dialogue.strip())
    
    summary = summarize_text(model, tokenizer, sample_dialogue, device)
    print("\n[=> AI T5 Tóm tắt]:")
    print(summary)
    
    print("\n" + "="*50)
    print("3. ĐÁNH GIÁ ĐỊNH LƯỢNG BẰNG THANG ĐO ROUGE (TEST SET)")
    print("="*50)
    rouge = evaluate.load("rouge")
    
    dataset = load_dataset("knkarthick/samsum")
    test_data = dataset["test"]
    
    # Lấy 50 mẫu đầu tiên để test nhanh (Để đánh giá toàn bộ 819 câu, bạn đổi thành test_data)
    sample_size = 50
    print(f"Đang tự động đọc và tóm tắt {sample_size} đoạn hội thoại để chấm điểm...")
    
    dialogues = test_data["dialogue"][:sample_size]
    references = test_data["summary"][:sample_size]
    
    predictions = []
    for text in dialogues:
        pred = summarize_text(model, tokenizer, text, device)
        predictions.append(pred)
        
    # Tiến hành đối chiếu bản của AI (predictions) với bản của con người (references)
    results = rouge.compute(predictions=predictions, references=references)
    
    print("\n[KẾT QUẢ ĐIỂM ROUGE CỦA MÔ HÌNH]:")
    print(f"- ROUGE-1 (Trùng lặp từng từ đơn): {results['rouge1'] * 100:.2f}%")
    print(f"- ROUGE-2 (Trùng lặp cụm 2 từ):  {results['rouge2'] * 100:.2f}%")
    print(f"- ROUGE-L (Trùng lặp câu dài nhất): {results['rougeL'] * 100:.2f}%")
    print("\nHoàn tất 100% Pipeline của Project!")

if __name__ == "__main__":
    main()