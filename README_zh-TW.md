<p align="center">
  <img src="img/banner.png" alt="OllaForge Banner" width="100%">
</p>

<h1 align="center">OllaForge 🔥</h1>

<p align="center">
  <strong>AI 驅動的 LLM 微調資料集生成與擴增工具</strong>
</p>

<p align="center">
  <a href="#-功能特色">功能特色</a> •
  <a href="#-快速開始">快速開始</a> •
  <a href="#-使用方式">使用方式</a> •
  <a href="#-資料集擴增">資料集擴增</a> •
  <a href="#-資料集格式">資料集格式</a> •
  <a href="#-效能優化">效能優化</a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-TW.md">繁體中文</a>
</p>

<p align="center">
  <a href="https://github.com/ollaforge/ollaforge/actions"><img src="https://img.shields.io/github/actions/workflow/status/ollaforge/ollaforge/ci.yml?branch=main&label=CI&logo=github" alt="CI Status"></a>
  <a href="https://pypi.org/project/ollaforge/"><img src="https://img.shields.io/pypi/v/ollaforge?color=blue&logo=pypi&logoColor=white" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/ollaforge/"><img src="https://img.shields.io/pypi/pyversions/ollaforge?logo=python&logoColor=white" alt="Python Versions"></a>
  <a href="https://github.com/ollaforge/ollaforge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ollaforge/ollaforge?color=green" alt="License"></a>
</p>

---

## 🎯 什麼是 OllaForge？

**OllaForge** 是一個高效能的 CLI 工具，利用本地 Ollama 模型來**生成**和**擴增** LLM 微調訓練資料集。具備結構化 JSON 輸出、並行批次處理和內建品質控制功能，同時兼顧品質與速度。

### 為什麼選擇 OllaForge？

- 🔒 **100% 本地與隱私** - 您的資料永遠不會離開您的電腦
- ⚡ **極速處理** - 結構化輸出的並行批次處理
- 🎨 **靈活彈性** - 生成新資料集或擴增現有資料集
- 🌐 **多語言支援** - 英文與繁體中文（台灣）並附品質控制
- 🔧 **生產就緒** - 相容 HuggingFace 與 LLaMA-Factory

---

## ✨ 功能特色

### 🆕 資料集生成
| 功能 | 說明 |
|------|------|
| 🎯 **自然語言主題** | 用自然語言描述您的資料集需求 |
| 🤖 **支援任何 Ollama 模型** | 支援 Llama 3、Mistral、Qwen、DeepSeek、Gemma 等 |
| 📊 **4 種資料集格式** | SFT、預訓練、對話（ShareGPT）、DPO |
| ⚡ **並行批次處理** | 幾分鐘內生成數百筆資料 |

### 🔄 資料集擴增
| 功能 | 說明 |
|------|------|
| 📝 **欄位修改** | 使用 AI 驅動的轉換增強現有欄位 |
| ➕ **新增欄位** | 根據現有資料新增計算欄位 |
| 👀 **預覽模式** | 在完整處理前先測試樣本 |
| 🛡️ **失敗復原** | AI 失敗時保留原始資料 |

### 📁 多格式支援
| 功能 | 說明 |
|------|------|
| 📄 **JSONL** | JSON Lines 格式（預設）- 每行一個 JSON 物件 |
| 📋 **JSON** | 單一 JSON 物件陣列 |
| 📊 **CSV** | 逗號分隔值，自動偵測標題 |
| 📑 **TSV** | Tab 分隔值，適用於結構化資料 |
| 🗃️ **Parquet** | 列式儲存格式（需要 pandas） |

### 🌐 品質與在地化
| 功能 | 說明 |
|------|------|
| 🔍 **BERT 品質控制** | 過濾大陸用語，確保台灣繁體中文品質 |
| 🌏 **多語言支援** | 英文與繁體中文（台灣）支援 |
| ✅ **結構化輸出** | JSON Schema 強制執行，0% 格式錯誤 |
| 📈 **進度追蹤** | Rich 驅動的即時進度顯示 |

---

## 🚀 快速開始

### 前置需求

- Python 3.9+
- [Ollama](https://ollama.ai/) 已安裝並執行中

### 安裝

```bash
# 從 PyPI 安裝（推薦）
pip install ollaforge

# 或從原始碼安裝
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge
pip install -e .

# 包含繁體中文品質控制支援
pip install ollaforge[qc]

# 包含多格式支援（CSV、Parquet 等）
pip install ollaforge[formats]

# 包含所有功能
pip install ollaforge[all]
```

### 您的第一個資料集

```bash
# 互動模式（推薦初學者使用）
ollaforge -i

# 生成 SFT 資料集
ollaforge generate "Python 程式設計教學" --count 100 --output python_sft.jsonl

# 繁體中文對話資料集
ollaforge generate "咖啡點餐對話" --type sft_conv --lang zh-tw --count 100
```

### 擴增現有資料集

```bash
# 處理前先預覽
ollaforge augment data.jsonl --field output --instruction "增加更多細節" --preview

# 新增欄位
ollaforge augment data.jsonl --field difficulty --new-field --instruction "評估難度：簡單/中等/困難"

# 處理 CSV 檔案
ollaforge augment data.csv --field sentiment --new-field --instruction "分析情感：正面/負面/中性"

# 格式轉換
ollaforge convert data.csv data.jsonl

# 互動式擴增精靈
ollaforge augment data.jsonl -i
```

---

## 📖 使用方式

### 生成指令

```bash
ollaforge generate <主題> [選項]
```

| 選項 | 簡寫 | 預設值 | 說明 |
|------|------|--------|------|
| `--count` | `-c` | 10 | 生成數量（1-10,000） |
| `--model` | `-m` | llama3.2 | Ollama 模型名稱 |
| `--output` | `-o` | dataset.jsonl | 輸出檔名 |
| `--type` | `-t` | sft | 格式：`sft`、`pretrain`、`sft_conv`、`dpo` |
| `--lang` | `-l` | en | 語言：`en`、`zh-tw` |
| `--concurrency` | `-j` | 5 | 並行請求數（1-20） |
| `--qc/--no-qc` | | --qc | 台灣繁體中文品質控制 |
| `--interactive` | `-i` | | 啟動精靈模式 |

### 擴增指令

```bash
ollaforge augment <輸入檔案> [選項]
```

| 選項 | 簡寫 | 預設值 | 說明 |
|------|------|--------|------|
| `--field` | `-f` | 必填 | 目標欄位 |
| `--instruction` | `-I` | 必填 | AI 擴增指令 |
| `--output` | `-o` | 自動 | 輸出檔案（預設：input_augmented.jsonl） |
| `--model` | `-m` | llama3.2 | Ollama 模型名稱 |
| `--new-field` | | false | 建立新欄位而非修改 |
| `--context` | `-c` | | 額外上下文欄位 |
| `--preview` | `-p` | | 處理前預覽 |
| `--concurrency` | `-j` | 5 | 並行請求數 |
| `--interactive` | `-i` | | 互動模式 |

---

## 🔄 資料集擴增

OllaForge 可以使用 AI 增強現有的 JSONL 資料集。

### 使用案例

- **翻譯**：將欄位翻譯成不同語言
- **豐富化**：新增難度、類別或情感等元資料
- **擴展**：將簡短回答擴展為詳細說明
- **轉換**：轉換格式或風格

### 範例

```bash
# 將 output 欄位翻譯成中文
ollaforge augment qa.jsonl -f output -I "翻譯成繁體中文（台灣用語）"

# 新增難度評級
ollaforge augment problems.jsonl -f difficulty --new-field -I "根據複雜度評級：簡單/中等/困難"

# 擴展簡短回答
ollaforge augment faq.jsonl -f answer -I "擴展此回答，加入更多細節和範例"

# 使用上下文新增類別欄位
ollaforge augment articles.jsonl -f category --new-field -c title -c content -I "分類：科技/科學/商業/其他"
```

---

## 📋 資料集格式

### SFT（Alpaca 格式）
```json
{"instruction": "解釋遞迴", "input": "", "output": "遞迴是..."}
```

### 預訓練
```json
{"text": "機器學習是人工智慧的一個子領域..."}
```

### SFT 對話（ShareGPT/ChatML）
```json
{
  "conversations": [
    {"role": "system", "content": "你是一個有幫助的助理。"},
    {"role": "user", "content": "如何反轉字串？"},
    {"role": "assistant", "content": "使用切片：`s[::-1]`"}
  ]
}
```

### DPO（偏好配對）
```json
{"prompt": "寫階乘函數", "chosen": "def factorial(n)...", "rejected": "def f(n):..."}
```

---

## ⚡ 效能優化

OllaForge 針對本地 LLM 推論進行優化：

| 優化項目 | 效益 |
|----------|------|
| **結構化 JSON 輸出** | 透過 Ollama 的 Schema 強制執行，0% 格式錯誤 |
| **小批次大小（5）** | 減少注意力衰減，提升品質 |
| **並行請求** | 最多 10 個並行批次請求 |
| **BERT 在 CPU 執行** | 保持 GPU/MPS 給 LLM 生成使用 |
| **漏斗架構** | 過量請求 → 過濾 → 保留有效資料 |

---

## 🔍 繁體中文品質控制

使用 `--lang zh-tw` 時，OllaForge 會自動過濾大陸用語：

| ❌ 過濾 | ✅ 接受 |
|---------|---------|
| 軟件 | 軟體 |
| 視頻 | 影片 |
| 程序 | 程式 |
| 網絡 | 網路 |
| 信息 | 資訊 |

```bash
# 啟用品質控制（zh-tw 預設）
ollaforge generate "對話" --lang zh-tw --qc

# 更嚴格的閾值
ollaforge generate "對話" --lang zh-tw --qc-confidence 0.95
```

---

## 🤖 推薦模型

| 模型 | 最適用途 | VRAM |
|------|----------|------|
| `llama3.2` | 通用（預設） | 8GB |
| `qwen2.5:14b` | 多語言、中文 | 16GB |
| `deepseek-r1:14b` | 推理任務 | 16GB |
| `gemma2:9b` | 高效、平衡 | 12GB |
| `mistral:7b` | 快速推論 | 8GB |

---

## 🤝 貢獻

歡迎貢獻！請參閱 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

### 快速開始

1. Fork 此儲存庫
2. 建立功能分支（`git checkout -b feature/amazing`）
3. 為您的變更撰寫測試
4. 確保所有測試通過（`make check`）
5. 提交 Pull Request

---

## 📜 授權

MIT 授權 - 詳見 [LICENSE](LICENSE)。

---

<p align="center">
  <strong>由 OllaForge 團隊用 ❤️ 製作</strong>
</p>

<p align="center">
  <a href="https://github.com/ollaforge/ollaforge/issues/new?template=bug_report.md">回報問題</a> •
  <a href="https://github.com/ollaforge/ollaforge/issues/new?template=feature_request.md">功能建議</a> •
  <a href="https://github.com/ollaforge/ollaforge/discussions">討論區</a>
</p>
