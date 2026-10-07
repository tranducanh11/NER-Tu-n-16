from datasets import load_dataset
import pandas as pd

def main():
    print("Đang kết nối tới máy chủ Hugging Face để tải SAMSum...")
    # Tải bộ dữ liệu. Tham số trust_remote_code=True bắt buộc có để tải SAMSum an toàn.
    dataset = load_dataset("knkarthick/samsum")
    
    print("\n" + "="*50)
    print("TỔNG QUAN VỀ BỘ DỮ LIỆU (DATASET DICTIONARY)")
    print("="*50)
    print(dataset)
    
    print("\n" + "="*50)
    print("VÍ DỤ CHI TIẾT TỪ TẬP HUẤN LUYỆN (TRAIN SET)")
    print("="*50)
    # Lấy ví dụ đầu tiên (index = 0)
    sample = dataset['train'][0]
    
    print("[NỘI DUNG ĐOẠN CHAT (INPUT)]:")
    print(sample['dialogue'])
    
    print("\n[BẢN TÓM TẮT CHUẨN CỦA CON NGƯỜI (TARGET)]:")
    print(sample['summary'])
    
    print("\n[ID ĐỊNH DANH ĐOẠN CHAT]:", sample['id'])

if __name__ == "__main__":
    main()