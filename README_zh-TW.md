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
  <a href="#️-網頁介面">網頁介面</a> •
  <a href="#-文件轉資料集">文件轉資料集</a> •
  <a href="#-資料集擴增">資料集擴增</a> •
  <a href="#-資料集格式">資料集格式</a> •
  <a href="#-貢獻">貢獻</a>
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

### 📄 文件轉資料集

| 功能 | 說明 |
|------|------|
| 📑 **多格式解析** | 支援 Markdown、PDF、HTML、TXT、JSON 和程式碼檔案 |
| ✂️ **智慧分塊** | 語意邊界感知的文字分割 |
| 🎯 **4 種輸出格式** | SFT、預訓練、對話、DPO |
| 📁 **批次處理** | 使用 glob 模式處理整個目錄 |
| 🔍 **品質控制** | 內建驗證與品質控制過濾 |

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

### 🖥️ 網頁介面（🚧 開發中）

| 功能 | 說明 |
|------|------|
| 🌐 **瀏覽器操作** | 無需 CLI 知識，透過瀏覽器即可使用 |
| 📊 **即時進度** | WebSocket 驅動的即時進度更新 |
| 🌍 **雙語介面** | 支援英文與繁體中文介面 |
| 🐳 **Docker 部署** | 一鍵 Docker Compose 部署 |

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
```

### 文件轉資料集

```bash
# 將 Markdown 文件轉換為 SFT 資料集
ollaforge doc2dataset README.md --type sft --output readme_dataset.jsonl

# 處理目錄中所有 Python 檔案
ollaforge doc2dataset ./src --pattern "*.py" --type pretrain

# 將 PDF 轉換為繁體中文資料集
ollaforge doc2dataset manual.pdf --lang zh-tw --qc
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

### 文件轉資料集指令

將文件（Markdown、PDF、HTML、TXT、JSON、程式碼檔案）轉換為微調資料集。

```bash
ollaforge doc2dataset <來源> [選項]
```

| 選項 | 簡寫 | 預設值 | 說明 |
|------|------|--------|------|
| `--output` | `-o` | dataset.jsonl | 輸出檔案路徑 |
| `--type` | `-t` | sft | 格式：`sft`、`pretrain`、`sft_conv`、`dpo` |
| `--model` | `-m` | llama3.2 | Ollama 模型名稱 |
| `--chunk-size` | | 2000 | 區塊大小（字元數，500-10000） |
| `--chunk-overlap` | | 200 | 區塊重疊（0-1000） |
| `--count` | `-c` | 3 | 每區塊生成筆數（1-10） |
| `--lang` | `-l` | en | 語言：`en`、`zh-tw` |
| `--pattern` | `-p` | | 目錄檔案模式（如 `*.md`） |
| `--recursive/--no-recursive` | | --recursive | 遞迴處理目錄 |
| `--qc/--no-qc` | | --qc | 啟用品質控制 |

#### 支援的檔案格式

| 格式 | 副檔名 | 說明 |
|------|--------|------|
| Markdown | `.md` | 保留標題結構 |
| PDF | `.pdf` | 提取所有頁面文字 |
| HTML | `.html`、`.htm` | 移除標籤，保留文字 |
| 純文字 | `.txt` | 直接讀取文字 |
| JSON | `.json` | 提取字串值 |
| 程式碼 | `.py`、`.js`、`.ts`、`.java`、`.go`、`.rs`、`.c`、`.cpp`、`.rb` | 語言偵測 |

---

## 🖥️ 網頁介面

> ⚠️ **注意：網頁介面目前正在積極開發中。** 部分功能可能尚未完成或不穩定。

OllaForge 提供現代化的網頁介面，讓偏好圖形化操作的使用者無需使用命令列。

### Docker 快速啟動

```bash
# 複製儲存庫
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge

# 使用 Docker Compose 啟動（需要主機上執行 Ollama）
docker-compose up -d

# 開啟網頁介面
open http://localhost
```

### 手動設定

**後端：**
```bash
pip install ollaforge[web]
python -m ollaforge.web.server
# 伺服器執行於 http://localhost:8000
```

**前端：**
```bash
cd ollaforge-web
npm install
npm run dev
# 前端執行於 http://localhost:5173
```

### 頁面說明

| 頁面 | 說明 |
|------|------|
| `/generate` | 從主題描述建立新資料集 |
| `/augment` | 上傳並增強現有資料集 |
| `/config` | 管理已儲存的設定 |

### API 文件

- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc

---

## 📄 文件轉資料集

OllaForge 可以使用 AI 驅動的內容分析，將各種文件格式轉換為微調資料集。

### 運作方式

1. **解析**：解析文件以提取文字內容
2. **分塊**：將長文件分割成可管理的區塊，尊重語意邊界
3. **生成**：AI 分析每個區塊並生成訓練資料項目
4. **驗證**：根據格式 Schema 驗證輸出，並可選擇性啟用品質控制過濾

### 使用案例

- **文件 → 問答**：將技術文件轉換為問答配對
- **程式碼 → 教學**：將程式碼檔案轉換為教學內容
- **文章 → 對話**：從文章建立對話資料集
- **手冊 → 訓練資料**：從產品手冊生成微調資料

### 範例

```bash
# 將專案 README 轉換為 SFT 資料集
ollaforge doc2dataset README.md --type sft --count 5

# 處理資料夾中所有 Markdown 文件
ollaforge doc2dataset ./docs --pattern "*.md" --type sft_conv

# 將 PDF 手冊轉換為繁體中文輸出
ollaforge doc2dataset manual.pdf --lang zh-tw --qc

# 從原始碼生成預訓練資料
ollaforge doc2dataset ./src --pattern "*.py" --type pretrain --chunk-size 1500

# 批次處理 HTML 文件
ollaforge doc2dataset ./html_docs --pattern "*.html" --type sft --output training_data.jsonl
```

### 安裝

文件解析需要額外的依賴套件：

```bash
# 安裝文件解析支援
pip install ollaforge[docs]

# 或安裝所有功能
pip install ollaforge[all]
```

---

## 🔄 資料集擴增

OllaForge 可以使用 AI 增強現有資料集。

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

| 優化項目 | 效益 |
|----------|------|
| **結構化 JSON 輸出** | 透過 Ollama 的 Schema 強制執行，0% 格式錯誤 |
| **小批次大小（5）** | 減少注意力衰減，提升品質 |
| **並行請求** | 最多 10 個並行批次請求 |
| **BERT 在 CPU 執行** | 保持 GPU/MPS 給 LLM 生成使用 |

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

## 🧪 開發

```bash
# 複製並設定
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge
pip install -e ".[dev]"

# 執行測試
make test

# 程式碼檢查與格式化
make lint
make format

# 所有檢查
make check
```

---

## 🤝 貢獻

歡迎貢獻！請參閱 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

1. Fork 此儲存庫
2. 建立功能分支（`git checkout -b feature/amazing`）
3. 為您的變更撰寫測試
4. 確保所有測試通過（`make check`）
5. 提交 Pull Request

---

## 📜 授權

MIT 授權 - 詳見 [LICENSE](LICENSE)。

---

## 🙏 致謝

- [Ollama](https://ollama.ai/) - 本地 LLM 推論
- [Rich](https://github.com/Textualize/rich) - 美觀的終端機 UI
- [Typer](https://typer.tiangolo.com/) - CLI 框架
- [Pydantic](https://pydantic.dev/) - 資料驗證

---

<p align="center">
  <strong>由 OllaForge 團隊用 ❤️ 製作</strong>
</p>

<p align="center">
  <a href="https://github.com/ollaforge/ollaforge/issues/new?template=bug_report.md">回報問題</a> •
  <a href="https://github.com/ollaforge/ollaforge/issues/new?template=feature_request.md">功能建議</a> •
  <a href="https://github.com/ollaforge/ollaforge/discussions">討論區</a>
</p>
