import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# 載入訓練好的模型和分詞器
model_name = "Helsinki-NLP/opus-mt-zh-en"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
model.load_state_dict(torch.load("fine_tuned_translation_model.pth"))
'''
# Initialize a new instance of MarianMTModel
model_name = "Helsinki-NLP/opus-mt-zh-en"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
# Load the saved parameters
model.load_state_dict(torch.load("fine_tuned_translation_model.h5"))
'''
# 加載測試資料
test_data = pd.read_csv('test-ZH-nospace.csv')

# 將模型和資料移至適當的設備
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 設定輸出最大長度
max_output_length = 512

# 進行翻譯並生成結果
predictions = []
with torch.no_grad():
    for _, row in test_data.iterrows():
        zh_text = row['txt']
        inputs = tokenizer(zh_text, return_tensors="pt").to(device)
        
        # 進行翻譯
        translated = model.generate(**inputs, max_length=max_output_length)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        predictions.append(translated_text)

# 將結果存入DataFrame
submission_df = pd.DataFrame({'id': test_data['id'], 'txt': predictions})

# 儲存結果到 sub.csv
submission_df.to_csv('sub.csv', index=False)