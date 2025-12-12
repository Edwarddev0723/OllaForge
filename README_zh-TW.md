<p align="center">
  <img src="img/banner.png" alt="OllaForge Banner" width="100%">
</p>

<h1 align="center">OllaForge 🔥</h1>

<p align="center">
  <strong>AI 驅動的 LLM 訓練資料集生成器</strong>
</p>

<p align="center">
  <a href="#功能特色">功能特色</a> •
  <a href="#快速開始">快速開始</a> •
  <a href="#使用方式">使用方式</a> •
  <a href="#資料集格式">資料集格式</a> •
  <a href="#貢獻指南">貢獻指南</a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-TW.md">繁體中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/ollama-local-orange.svg" alt="Ollama">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

---

**OllaForge** 是一款強大的命令列工具，利用本地 Ollama 模型自動生成高品質、主題特定的 LLM 訓練資料集。只需一個指令，即可生成 SFT、預訓練、對話和 DPO 資料集。

## ✨ 功能特色

| 功能 | 說明 |
|------|------|
| 🎯 **自然語言主題** | 用白話文描述你的資料集需求 |
| 🤖 **多模型支援** | 支援 Llama 3、Mistral、Qwen、DeepSeek 等模型 |
| 📊 **4 種資料集格式** | SFT、預訓練、對話 (ShareGPT)、DPO |
| 🌐 **多語言輸出** | 支援英文和繁體中文（台灣用語）|
| 🎨 **精美介面** | 基於 Rich 的互動式精靈介面 |
| ⚡ **批次處理** | 可設定並行數的高效生成 |
| ✅ **自動驗證** | 內建 JSON 驗證和錯誤恢復機制 |
| 🔄 **HuggingFace 相容** | 輸出格式相容 HuggingFace 和 LLaMA-Factory |

## 🚀 快速開始

### 前置需求

- Python 3.8+
- [Ollama](https://ollama.ai/) 已安裝並在本地運行

### 安裝步驟

```bash
# 複製專案
git clone https://github.com/yourusername/ollaforge.git
cd ollaforge

# 安裝相依套件
pip install -r requirements.txt

# 確認 Ollama 正在運行
ollama list
```

### 生成你的第一個資料集

```bash
# 互動模式（推薦新手使用）
python main.py -i

# 或直接生成
python main.py "Python 程式設計教學" --lang zh-tw --count 100 --output python_sft.jsonl
```

## 📖 使用方式

### 互動模式

啟動步驟式精靈：

```bash
python main.py -i
```

精靈會引導你完成：
1. 📝 主題描述
2. 📊 資料集類型選擇
3. 🌐 輸出語言
4. 🔢 生成數量
5. 🤖 模型選擇
6. 📄 輸出設定

### 命令列模式

```bash
python main.py <主題> [選項]
```

#### 選項說明

| 選項 | 簡寫 | 預設值 | 說明 |
|------|------|--------|------|
| `--count` | `-c` | 10 | 生成的資料筆數 |
| `--model` | `-m` | gpt-oss:20b | 使用的 Ollama 模型 |
| `--output` | `-o` | dataset.jsonl | 輸出檔案名稱 |
| `--type` | `-t` | sft | 資料集類型 (sft/pretrain/sft_conv/dpo) |
| `--lang` | `-l` | en | 輸出語言 (en/zh-tw) |
| `--concurrency` | `-j` | 5 | 並行請求數 (1-20) |
| `--interactive` | `-i` | - | 啟動互動模式 |

#### 使用範例


```bash
# 生成 SFT 訓練資料（繁體中文）
python main.py "客服對話範例" --lang zh-tw --count 500 --type sft

# 生成預訓練語料
python main.py "機器學習研究論文" --lang zh-tw --type pretrain --count 1000

# 生成多輪對話
python main.py "技術支援對話" --lang zh-tw --type sft_conv --output conversations.jsonl

# 生成 DPO 偏好配對
python main.py "程式碼審查回饋" --lang zh-tw --type dpo --count 200

# 使用特定模型
python main.py "醫療問答" --model deepseek-r1:14b --lang zh-tw --count 50
```

## 📋 資料集格式

OllaForge 生成的資料集相容 **HuggingFace** 和 **LLaMA-Factory**。

### SFT（監督式微調）

Alpaca 風格格式，用於指令微調：

```json
{
  "instruction": "解釋遞迴的概念",
  "input": "我正在學習程式設計",
  "output": "遞迴是一種程式設計技巧，函式會呼叫自己本身..."
}
```

### 預訓練

原始文字格式，用於持續預訓練：

```json
{
  "text": "機器學習是人工智慧的一個子領域，它使系統能夠從經驗中學習和改進..."
}
```

### SFT 對話（ShareGPT/ChatML）

多輪對話格式：

```json
{
  "conversations": [
    {"role": "system", "content": "你是一個有幫助的程式設計助手。"},
    {"role": "user", "content": "如何在 Python 中反轉字串？"},
    {"role": "assistant", "content": "你可以使用切片來反轉字串：`reversed_string = original[::-1]`"},
    {"role": "user", "content": "用迴圈怎麼做？"},
    {"role": "assistant", "content": "以下是使用迴圈的方法：\n```python\ndef reverse_string(s):\n    result = ''\n    for char in s:\n        result = char + result\n    return result\n```"}
  ]
}
```

### DPO（直接偏好最佳化）

用於 RLHF 訓練的偏好配對：

```json
{
  "prompt": "寫一個計算階乘的函式",
  "chosen": "以下是一個具有適當錯誤處理的高效遞迴實作：\n```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError('負數沒有階乘')\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```",
  "rejected": "def f(n): return n*f(n-1) if n else 1"
}
```

## 🌐 支援語言

| 代碼 | 語言 | 範例 |
|------|------|------|
| `en` | English | `--lang en` |
| `zh-tw` | 繁體中文（台灣） | `--lang zh-tw` |

## 🤖 推薦模型

| 模型 | 大小 | 最適用途 |
|------|------|----------|
| `gpt-oss:20b` | 20B | 通用目的（預設）|
| `deepseek-r1:14b` | 14B | 推理與複雜任務 |
| `qwen3:14b` | 14B | 多語言支援 |
| `ministral-3:14b` | 14B | 邊緣部署 |
| `gemma3:12b` | 12B | 單 GPU 效率 |

## 🏗️ 專案架構

```
ollaforge/
├── main.py              # CLI 進入點
├── ollaforge/
│   ├── client.py        # Ollama API 通訊
│   ├── processor.py     # 回應解析與驗證
│   ├── models.py        # Pydantic 資料模型
│   ├── interactive.py   # Rich 互動式介面
│   ├── progress.py      # 進度追蹤
│   └── file_manager.py  # 檔案 I/O 操作
└── tests/               # 完整測試套件
```

## 🧪 開發

```bash
# 執行測試
pytest tests/ -v

# 執行測試並產生覆蓋率報告
pytest tests/ --cov=ollaforge

# 型別檢查
mypy ollaforge/
```

## 🤝 貢獻指南

我們歡迎各種貢獻！以下是參與方式：

1. 🍴 Fork 這個專案
2. 🌿 建立功能分支 (`git checkout -b feature/amazing-feature`)
3. 💾 提交你的變更 (`git commit -m '新增超棒功能'`)
4. 📤 推送到分支 (`git push origin feature/amazing-feature`)
5. 🔃 開啟 Pull Request

### 我們需要幫助的領域

- [ ] 新增更多語言支援（日文、韓文等）
- [ ] 更多資料集格式範本
- [ ] 效能最佳化
- [ ] 文件改進
- [ ] 測試覆蓋率擴展

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案。

## 🙏 致謝

- [Ollama](https://ollama.ai/) 讓本地 LLM 變得觸手可及
- [Rich](https://github.com/Textualize/rich) 提供精美的終端機輸出
- [Typer](https://typer.tiangolo.com/) 優雅的 CLI 建立工具
- [Pydantic](https://pydantic.dev/) 資料驗證框架

---

<p align="center">
  由 OllaForge 團隊用 ❤️ 打造
</p>

<p align="center">
  <a href="https://github.com/yourusername/ollaforge/stargazers">⭐ 在 GitHub 上給我們一顆星</a>
</p>
