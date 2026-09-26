from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_prepare_data(model_checkpoint="vinai/phobert-base"):
    """
    Hàm này có nhiệm vụ tải dữ liệu và băm chữ thành số (Tokenize) bằng PhoBERT.
    """
    # 1. Tải dữ liệu WikiANN
    dataset = load_dataset("wikiann", "vi")
    
    # 2. Tải công cụ băm từ (Tokenizer) của PhoBERT
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # 3. Hàm đồng bộ nhãn (Tùy chỉnh thủ công cho PhoBERT)
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True, 
            max_length=256
        )
        
        labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            label_ids = []
            # Duyệt qua từng từ gốc trong câu
            for word_idx, word in enumerate(examples["tokens"][i]):
                # Băm từ gốc thành các từ con (sub-words) bằng PhoBERT
                sub_words = tokenizer.tokenize(word)
                
                if len(sub_words) > 0:
                    # Gán nhãn của từ gốc cho sub-word đầu tiên
                    label_ids.append(ner_tags[word_idx])
                    # Gán -100 cho các sub-word bị cắt vỡ phía sau để AI không tính điểm lỗi
                    label_ids.extend([-100] * (len(sub_words) - 1))
            
            # Thêm -100 cho token đặc biệt <s> ở đầu và </s> ở cuối câu
            label_ids = [-100] + label_ids + [-100]
            
            # Cắt bớt nếu câu quá dài (vượt max_length=256)
            if len(label_ids) > 256:
                label_ids = label_ids[:256]
                label_ids[-1] = -100
                
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # 4. Áp dụng băm từ cho toàn bộ tập dữ liệu
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    
    # 5. Lấy danh sách tên nhãn (B-PER, I-PER,...)
    label_list = dataset["train"].features["ner_tags"].feature.names
    
    return tokenized_datasets, tokenizer, label_list

# Đoạn code này chỉ chạy khi bạn mở trực tiếp file data.py để test thử
if __name__ == "__main__":
    print("Đang kết nối tải Tokenizer của PhoBERT...")
    data, tokenizer, labels = load_and_prepare_data()
    print("\n--- KẾT QUẢ ---")
    print(f"Xử lý thành công! Số lượng nhãn NER: {len(labels)}")
    print(f"Danh sách nhãn: {labels}")