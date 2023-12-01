import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# 載入預訓練模型和分詞器
model_name = 'Helsinki-NLP/opus-mt-zh-en'  # 預訓練的中翻英模型
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 資料預處理
train_zh = pd.read_csv('train-ZH.csv')
train_tl = pd.read_csv('train-TL.csv')

# 分詞和準備訓練數據
train_encodings = tokenizer(list(train_zh['txt']), padding=True, truncation=True, return_tensors='pt')
train_decodings = tokenizer(list(train_tl['txt']), padding=True, truncation=True, return_tensors='pt')

# 處理 labels 張量，將其前移一個位置作為 labels
train_labels = train_decodings['input_ids'][:, 1:].clone()

train_data = list(zip(train_encodings['input_ids'], train_labels))

# Fine-tune 模型
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 5  # 設置您想要的 epoch 數量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 訓練模型
for epoch in range(num_epochs):
    total_loss = 0.0

    with tqdm(total=len(train_zh), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for i in range(len(train_zh)):
            zh_text = train_zh['txt'][i]
            tl_text = train_tl['txt'][i]

            inputs = tokenizer(zh_text, return_tensors="pt").to(device)
            labels = tokenizer(tl_text, return_tensors="pt").input_ids.squeeze(0).to(device)

            outputs = model(**inputs, labels=labels.unsqueeze(0))
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())
            pbar.update()

    average_loss = total_loss / len(train_zh)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

# 儲存訓練好的模型
torch.save(model.state_dict(), "fine_tuned_translation_model.pth")
 
