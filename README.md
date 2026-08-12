# NewYoloModel
best.pt：
已訓練完成的優化YOLO模型。

example.py：
系統測試主腳本，包含 Stage 1 車體定位、Stage 2 車損語意分割 與 Canny 紋理過濾 雙階段演算法，可用於單張圖片測試或資料夾批次推論。

damage_image/：
放置包含 刮痕 (Scratch)、凹痕 (Dent)、漆面剝落 (Flaking) 或破損 等待測車損照片的輸入目錄。

non_damage_image/：
放置 乾淨無瑕疵、正常車輛或包含太陽強光反光 的照片目錄，專門用於驗證模型的抗干擾與抗誤判能力。