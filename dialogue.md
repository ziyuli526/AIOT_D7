# 開發日誌 (Development Log)

## 基本資訊
- **日期**: 2026-04-28
- **專案名稱**: Linear Regression CRISP-DM Demo (Streamlit)
- **開發環境**: Windows, Python 3.9, Streamlit, Scikit-learn

## 任務摘要
根據用戶需求，建立一個單檔案的 Streamlit 應用程式 (`app.py`)，透過 Scikit-learn 演示線性回歸在 CRISP-DM 工作流程下的完整生命週期。

## 實作內容
1. **數據生成**:
   - 實作合成數據引擎：$n \in [100, 1000]$, $x \sim \text{Uniform}(-100, 100)$, $a \sim \text{Uniform}(-10, 10)$, $b \sim \text{Uniform}(-50, 50)$。
   - 加入常態分佈雜訊 (Noise)：均值 $\in [-10, 10]$，變異數 $\in [0, 1000]$。
2. **UI 設計**:
   - 側邊欄：實作 $n$、變異數、隨機種子、真實係數的拉桿與「生成數據」按鈕。
   - 主介面：使用 `st.tabs` 劃分 CRISP-DM 的六個階段（業務理解、數據理解、數據準備、建模、評估、部署）。
3. **機器學習流程**:
   - 使用 `train_test_split` 進行資料分割。
   - 使用 `StandardScaler` 進行特徵標準化。
   - 訓練 `LinearRegression` 模型並計算 MSE, RMSE, $R^2$。
   - 視覺化：繪製散佈圖與回歸線，並對比「真實參數」與「學習參數」。
4. **功能擴充**:
   - 提供預測輸入介面。
   - 實作模型匯出功能，支援下載 `.joblib` 檔案（包含模型與 Scaler）。

## 遭遇問題與解決方案
- **套件缺失**: 初始運行時環境缺少 `matplotlib`, `seaborn`, `scikit-learn` 等關鍵套件。
- **解決方案**: 執行 `pip install` 自動安裝所有相依套件，確保應用程式能順利載入。
- **座標轉換**: 在顯示學習到的參數時，需將標準化空間的係數轉換回原始空間，以利與真實參數對比。

## 目前狀態
- **檔案**: `app.py` 已建立。
- **運行狀態**: 應用程式已成功啟動並驗證，可於 `http://localhost:8501` 正常存取。
- **成果**: 達成專業、流暢且具備教學價值的機器學習 Demo。
