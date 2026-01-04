# OllaForge 專案結構

本文件描述 OllaForge 專案的目錄結構和檔案組織。

## 目錄結構

```
OllaForge/
├── .github/                    # GitHub 配置
│   ├── ISSUE_TEMPLATE/         # Issue 模板
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── workflows/              # CI/CD 工作流程
│   │   ├── ci.yml              # 持續整合
│   │   └── release.yml         # 發布流程
│   └── PULL_REQUEST_TEMPLATE.md
│
├── .kiro/                      # Kiro 規格文件 (內部設計文檔)
│   └── specs/
│       ├── dataset-augmentation/   # 資料集擴增功能設計
│       ├── document-to-dataset/    # 文件轉資料集功能設計
│       └── web-interface/          # 網頁介面架構設計
│
├── .vscode.example/            # VS Code 範例設定 (複製到 .vscode 使用)
│   ├── README.md
│   ├── settings.json
│   └── extensions.json
│
├── docs/                       # 使用者文件
│   ├── README.md               # 文件首頁
│   ├── getting-started.md      # 快速開始指南
│   └── web-interface.md        # 網頁介面文件
│
├── examples/                   # 範例檔案
│   ├── datasets/               # 範例資料集
│   ├── eng_demodataset.jsonl   # 英文範例
│   ├── zhtw_demodataset.jsonl  # 繁體中文範例
│   ├── sample_data.csv
│   ├── sample_data.json
│   ├── sample_data.jsonl
│   └── README.md
│
├── img/                        # 圖片資源
│   └── banner.png
│
├── ollaforge/                  # 核心 Python 套件
│   ├── __init__.py
│   ├── __main__.py             # CLI 入口點
│   ├── augmentor.py            # 資料集擴增邏輯
│   ├── batch_processor.py      # 批次處理
│   ├── chunk_splitter.py       # 文件分塊
│   ├── cli.py                  # 命令列介面
│   ├── client.py               # Ollama 客戶端
│   ├── doc_generator.py        # 文件轉資料集生成器
│   ├── doc_parser.py           # 文件解析器
│   ├── file_manager.py         # 檔案管理
│   ├── formats.py              # 格式轉換
│   ├── hf_loader.py            # HuggingFace 資料集載入
│   ├── interactive.py          # 互動模式
│   ├── models.py               # 資料模型
│   ├── processor.py            # 資料處理
│   ├── progress.py             # 進度顯示
│   ├── qc.py                   # 品質控制 (繁體中文)
│   │
│   └── web/                    # Web API 後端
│       ├── __init__.py
│       ├── server.py           # FastAPI 伺服器
│       ├── models.py           # API 資料模型
│       ├── routes/             # API 路由
│       │   ├── __init__.py
│       │   ├── augmentation.py # 擴增 API
│       │   ├── generation.py   # 生成 API
│       │   ├── models.py       # 模型 API
│       │   ├── tasks.py        # 任務 API
│       │   └── websocket.py    # WebSocket
│       ├── services/           # 業務邏輯
│       │   ├── __init__.py
│       │   ├── augmentation.py
│       │   ├── generation.py
│       │   └── task_manager.py
│       ├── .env.example
│       ├── README.md
│       └── SETUP.md
│
├── ollaforge-web/              # React 前端
│   ├── src/
│   │   ├── components/         # React 元件
│   │   ├── pages/              # 頁面元件
│   │   ├── layouts/            # 版面配置
│   │   ├── services/           # API 服務
│   │   ├── hooks/              # React Hooks
│   │   ├── i18n/               # 國際化
│   │   └── ...
│   ├── e2e/                    # E2E 測試
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
│
├── tests/                      # Python 測試
│   ├── test_*.py               # 單元測試與屬性測試
│   └── ...
│
├── .gitignore
├── CHANGELOG.md                # 變更日誌
├── CODE_OF_CONDUCT.md          # 行為準則
├── CONTRIBUTING.md             # 貢獻指南
├── DEPLOYMENT.md               # 部署指南
├── docker-compose.yml          # Docker 開發配置
├── docker-compose.prod.yml     # Docker 生產配置
├── Dockerfile                  # 後端 Docker 映像
├── LICENSE                     # MIT 授權
├── Makefile                    # 常用指令
├── PROJECT_STRUCTURE.md        # 本文件
├── pyproject.toml              # Python 專案配置
├── README.md                   # 英文說明
├── README_zh-TW.md             # 繁體中文說明
├── requirements.txt            # Python 依賴
└── SECURITY.md                 # 安全政策
```

## 設計規格文件 (`.kiro/specs/`)

內部設計文檔集中在 `.kiro/specs/` 目錄，每個功能模組包含：

| 檔案 | 說明 |
|------|------|
| `requirements.md` | 使用者故事與驗收標準 |
| `design.md` | 技術設計與架構 |
| `tasks.md` | 實作任務與進度追蹤 |

## 主要模組說明

### 核心模組 (`ollaforge/`)

| 模組 | 說明 |
|------|------|
| `cli.py` | 命令列介面，提供 `generate` 和 `augment` 指令 |
| `client.py` | Ollama API 客戶端封裝 |
| `processor.py` | 資料集生成處理器 |
| `augmentor.py` | 資料集擴增處理器 |
| `file_manager.py` | 檔案讀寫管理 |
| `formats.py` | 多格式轉換 (JSONL, JSON, CSV, TSV, Parquet) |
| `models.py` | Pydantic 資料模型 |
| `progress.py` | 進度條顯示 |
| `qc.py` | 繁體中文品質控制 |
| `interactive.py` | 互動式模式 |

### Web API (`ollaforge/web/`)

| 模組 | 說明 |
|------|------|
| `server.py` | FastAPI 應用程式與 CORS 配置 |
| `routes/` | REST API 路由 |
| `services/` | 業務邏輯服務 |
| `models.py` | API 請求/回應模型 |

### 前端 (`ollaforge-web/`)

| 目錄 | 說明 |
|------|------|
| `components/` | 可重用 React 元件 |
| `pages/` | 頁面元件 (Generate, Augment, Config) |
| `services/` | API 客戶端與 WebSocket |
| `i18n/` | 國際化 (英文、繁體中文) |
| `e2e/` | Playwright E2E 測試 |

## 開發指令

```bash
# 安裝依賴
pip install -e ".[dev]"
cd ollaforge-web && npm install

# 執行測試
make test              # Python 測試
make test-frontend     # 前端測試

# 啟動開發伺服器
make run-backend       # 後端 API (port 8000)
make run-frontend      # 前端 (port 5173)

# Docker
docker-compose up -d   # 啟動所有服務
docker-compose down    # 停止服務
```

## 環境變數

### 後端 (`ollaforge/web/`)
- `OLLAMA_HOST` - Ollama 服務位址 (預設: `http://localhost:11434`)
- `CORS_ORIGINS` - 允許的 CORS 來源 (預設: localhost，使用 `*` 允許全部)
- `DEBUG` - 除錯模式 (預設: `false`)
- `PORT` - 伺服器埠號 (預設: `8000`)

### 前端 (`ollaforge-web/`)
- `VITE_API_BASE_URL` - 後端 API 位址
- `VITE_WS_BASE_URL` - WebSocket 位址
