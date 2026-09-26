from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import os

def main():
    # Sử dụng chữ 'r' (raw string) để khóa định dạng đường dẫn tuyệt đối
    # XÓA ĐƯỜNG DẪN NÀY VÀ DÁN ĐƯỜNG DẪN BẠN VỪA COPY PATH VÀO ĐÂY:
    model_path = r"E:\code\NLP_train_Payt\results\checkpoint-7500"
    
    print(f"Đường dẫn mô hình chuẩn: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ LỖI: Vẫn không tìm thấy thư mục! Hãy kiểm tra lại thao tác Copy Path.")
        return
    else:
        print("✅ Đã tìm thấy thư mục mô hình!")
        
    print("Đang nạp bộ não AI...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        return

    # Khởi tạo ống dẫn xử lý
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Câu văn cần nhận diện
    text = "Quyền trưởng khoa Trần Tiến Công đang lên kế hoạch tổ chức sự kiện cho Khoa AI - PTIT tại Học viện Công nghệ Bưu chính Viễn thông."
    
    print("\n" + "="*50)
    print(f"Câu văn gốc: {text}")
    print("="*50)
    
    # Chạy mô hình
    results = nlp(text)

    # In kết quả
    print("\nCác thực thể phát hiện được:")
    for entity in results:
        print(f"- Từ vựng: {entity['word']:<30} | Nhãn: {entity['entity_group']:<7} | Độ tin cậy: {entity['score']:.4f}")

if __name__ == "__main__":
    main()