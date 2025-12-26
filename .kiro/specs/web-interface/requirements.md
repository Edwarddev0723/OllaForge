# Requirements Document

## Introduction

本文件定義 OllaForge 網頁介面的需求規格。OllaForge 目前是一個命令列工具，用於生成和擴增 LLM 訓練資料集。本專案將為其添加一個網頁介面，讓使用者能夠透過瀏覽器進行資料集生成和擴增操作，提供更友善的使用體驗。

## Glossary

- **OllaForge**: 使用本地 Ollama 模型生成和擴增 LLM 訓練資料集的工具
- **Web Interface**: 基於瀏覽器的圖形化使用者介面
- **Dataset Generation**: 根據主題描述生成新的訓練資料集
- **Dataset Augmentation**: 對現有資料集進行欄位修改或新增
- **Ollama**: 本地 LLM 推論引擎
- **Backend Server**: 提供 API 端點的後端伺服器
- **Frontend Application**: 在瀏覽器中執行的前端應用程式
- **JSONL**: JSON Lines 格式，每行一個 JSON 物件
- **SFT**: Supervised Fine-tuning，監督式微調格式
- **QC**: Quality Control，品質控制

## Requirements

### Requirement 1

**User Story:** 作為使用者，我想要透過網頁介面生成資料集，這樣我就不需要使用命令列工具。

#### Acceptance Criteria

1. WHEN a user accesses the web interface THEN the system SHALL display a dataset generation form with input fields for topic, count, model, and dataset type
2. WHEN a user submits the generation form with valid parameters THEN the system SHALL initiate dataset generation and display real-time progress
3. WHEN dataset generation completes THEN the system SHALL provide a download link for the generated dataset file
4. WHEN dataset generation fails THEN the system SHALL display clear error messages to the user
5. WHEN a user selects Traditional Chinese language THEN the system SHALL enable QC filtering options

### Requirement 2

**User Story:** 作為使用者，我想要透過網頁介面擴增現有資料集，這樣我可以更方便地修改或新增欄位。

#### Acceptance Criteria

1. WHEN a user uploads a dataset file THEN the system SHALL validate the file format and display available fields
2. WHEN a user specifies augmentation parameters THEN the system SHALL validate the target field exists in the dataset
3. WHEN a user enables preview mode THEN the system SHALL display sample augmented entries before full processing
4. WHEN augmentation completes THEN the system SHALL provide a download link for the augmented dataset
5. WHEN augmentation fails for some entries THEN the system SHALL preserve original data and report failure statistics

### Requirement 3

**User Story:** 作為使用者，我想要看到即時的處理進度，這樣我可以知道操作需要多久時間。

#### Acceptance Criteria

1. WHEN dataset generation or augmentation is in progress THEN the system SHALL display a progress bar showing completion percentage
2. WHEN processing entries THEN the system SHALL update the progress indicator in real-time
3. WHEN processing completes THEN the system SHALL display total duration and success statistics
4. WHEN an error occurs during processing THEN the system SHALL display the error without stopping the progress display
5. WHEN a user cancels an operation THEN the system SHALL stop processing and save partial results if available

### Requirement 4

**User Story:** 作為使用者，我想要在網頁介面中管理多種檔案格式，這樣我可以使用不同格式的資料集。

#### Acceptance Criteria

1. WHEN a user uploads a file THEN the system SHALL support JSONL, JSON, CSV, TSV, and Parquet formats
2. WHEN a user downloads a generated dataset THEN the system SHALL provide format selection options
3. WHEN a user converts between formats THEN the system SHALL preserve all data fields correctly
4. WHEN an unsupported file format is uploaded THEN the system SHALL display a clear error message with supported formats
5. WHEN a file is too large THEN the system SHALL display a file size limit warning

### Requirement 5

**User Story:** 作為使用者，我想要預覽資料集內容，這樣我可以在下載前確認結果是否符合預期。

#### Acceptance Criteria

1. WHEN dataset generation completes THEN the system SHALL display the first 5 entries in a formatted table
2. WHEN a user views augmentation preview THEN the system SHALL show before and after comparison for sample entries
3. WHEN a user uploads a file THEN the system SHALL display the first 3 entries and field names
4. WHEN displaying entries THEN the system SHALL format JSON data in a readable structure
5. WHEN entries contain long text THEN the system SHALL truncate display with an expand option

### Requirement 6

**User Story:** 作為使用者，我想要儲存和載入常用的設定，這樣我可以快速重複相同的操作。

#### Acceptance Criteria

1. WHEN a user completes a generation or augmentation THEN the system SHALL offer to save the configuration
2. WHEN a user saves a configuration THEN the system SHALL store it in browser local storage with a user-provided name
3. WHEN a user loads a saved configuration THEN the system SHALL populate all form fields with the saved values
4. WHEN a user views saved configurations THEN the system SHALL display a list with names and timestamps
5. WHEN a user deletes a saved configuration THEN the system SHALL remove it from local storage

### Requirement 7

**User Story:** 作為系統管理員，我想要後端 API 能夠處理並發請求，這樣多個使用者可以同時使用系統。

#### Acceptance Criteria

1. WHEN multiple users submit generation requests THEN the system SHALL process each request independently
2. WHEN a generation or augmentation is in progress THEN the system SHALL not block other API endpoints
3. WHEN system resources are limited THEN the system SHALL queue requests and process them sequentially
4. WHEN a request times out THEN the system SHALL return an appropriate error response
5. WHEN the Ollama service is unavailable THEN the system SHALL return a clear error message

### Requirement 8

**User Story:** 作為使用者，我想要網頁介面支援繁體中文和英文，這樣我可以使用我偏好的語言。

#### Acceptance Criteria

1. WHEN a user accesses the web interface THEN the system SHALL detect browser language and display appropriate UI language
2. WHEN a user switches UI language THEN the system SHALL update all interface text immediately
3. WHEN displaying error messages THEN the system SHALL show messages in the current UI language
4. WHEN a user generates Traditional Chinese datasets THEN the system SHALL display QC options in Chinese
5. WHEN a user saves preferences THEN the system SHALL remember the selected UI language

### Requirement 9

**User Story:** 作為開發者，我想要前後端分離的架構，這樣可以獨立開發和部署前端與後端。

#### Acceptance Criteria

1. WHEN the backend server starts THEN the system SHALL expose RESTful API endpoints for all operations
2. WHEN the frontend makes API requests THEN the system SHALL use JSON format for request and response bodies
3. WHEN API endpoints are accessed THEN the system SHALL implement proper CORS headers for cross-origin requests
4. WHEN authentication is required THEN the system SHALL use token-based authentication
5. WHEN API documentation is needed THEN the system SHALL provide OpenAPI/Swagger documentation

### Requirement 10

**User Story:** 作為使用者，我想要在網頁介面中查看 Ollama 模型列表，這樣我可以選擇可用的模型。

#### Acceptance Criteria

1. WHEN a user opens the model selection dropdown THEN the system SHALL fetch and display available Ollama models
2. WHEN Ollama service is running THEN the system SHALL show model names with size information
3. WHEN Ollama service is not available THEN the system SHALL display a warning and use default model
4. WHEN a user selects a model THEN the system SHALL validate the model exists before starting generation
5. WHEN model list is empty THEN the system SHALL display instructions for installing Ollama models
