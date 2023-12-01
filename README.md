# main
# 台文漢字至台語（台羅拼音）翻譯的機器學習程式  
## 內附檔案
trainex.py:訓練程式檔  
testex.py:測試程式檔  
fine_tuned_translation_model.pth:訓練好的模型  
requirement.txt
採用現有的中翻英模型(Helsinki-NLP/opus-mt-zh-en)進行fine-tuned  
## 程式碼內容
### trainex.py
首先是資料前處理的部分:  
![image](https://github.com/312513016-morrisWW/main/assets/145111464/3d7dd822-5e0b-4fd0-a0da-48087d4b6fed)  
使用 Helsinki-NLP/opus-mt-zh-en 作為預訓練的中文到英文翻譯模型，並且載入對應的分詞器和模型。  
![image](https://github.com/312513016-morrisWW/main/assets/145111464/ebbe300f-c333-441c-b5f2-d2e63c7fb574)  
讀取了兩個 CSV 檔案 (train-ZH.csv 和 train-TL.csv)，分別含有中文和英文的訓練資料。(讀取路徑需要更改)  
![image](https://github.com/312513016-morrisWW/main/assets/145111464/787e422a-6acf-4536-9547-68ac7974b3c5)   
使用分詞器將中文和英文的句子進行分詞並轉換成模型可以接受的格式。將它們轉換成 PyTorch 張量的形式，並準備訓練數據。  
![image](https://github.com/312513016-morrisWW/main/assets/145111464/f4396d56-2eb7-440c-b4f1-02f2be4286e1)  
初始化了一個 AdamW 優化器，設置了訓練的epoch 數量，並檢查了是否有可用的GPU來進行模型訓練。將模型移到GPU或CPU上。  
![image](https://github.com/312513016-morrisWW/main/assets/145111464/09be1d53-9d81-4377-b85a-cbfea25a0d3f)  
使用迴圈對訓練資料進行訓練。在每個 epoch 中，透過 tqdm 來追蹤訓練的進度。在每一個訓練步驟中，從中文和英文訓練資料中取出一對句子，將它們轉換成模型可以接受的格式，並將其送入模型中進行訓練。同時計算並更新模型的梯度，最小化訓練損失。  
![image](https://github.com/312513016-morrisWW/main/assets/145111464/d4adea20-2557-4d75-b147-26f74a4494e3)
在訓練完成後，將訓練好的模型權重儲存為fine_tuned_translation_model.pth。  
### testex.py
![image](https://github.com/312513016-morrisWW/main/assets/145111464/3f046e0f-b767-4059-bcee-e60f2cc67028)  
載入了之前使用 Helsinki-NLP/opus-mt-zh-en 預訓練的中文到英文翻譯模型（MarianMTModel）和對應的分詞器（MarianTokenizer）。接著，使用 load_state_dict 載入了經過微調的模型權重，以便在測試資料上進行翻譯。  
![image](https://github.com/312513016-morrisWW/main/assets/145111464/eee990a5-4cff-49a2-9a1b-afffce71d907)  
使用 Pandas 讀取了測試資料集 test-ZH-nospace.csv，該資料集包含需要進行翻譯的中文資料。(路徑需要更改)  
![image](https://github.com/312513016-morrisWW/main/assets/145111464/a0eaf222-675a-4dad-b8c4-91550e49ebda)  
使用 torch.no_grad() 確保在翻譯時不計算梯度。透過迴圈對測試資料中的每一個中文文本進行翻譯。將每個中文文本轉換為模型可以接受的格式（PyTorch 張量），然後使用 model.generate() 進行翻譯，得到對應的英文文本。最後，使用分詞器的 decode 方法將索引轉換為文本，並將翻譯結果存儲在 predictions 這個列表中。  
![image](https://github.com/312513016-morrisWW/main/assets/145111464/5bfa7755-423d-4ded-bcae-628fffd5123a)  
將結果輸出至sub.csv
## 心得及相關討論
這次作業相對簡單，讓我深入了解了神經機器翻譯模型的微調和應用。通過使用transformer，我們能夠輕鬆地使用預訓練的模型來進行翻譯任務，並且進行進一步的微調，以提高模型在特定任務上的表現。  








