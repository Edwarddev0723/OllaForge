# Implementation Plan: Document to Dataset

## Overview

本實作計畫將「文件轉資料集」功能分解為可執行的編碼任務。採用漸進式開發，從核心解析器開始，逐步建構分塊處理、資料集生成，最後整合到 CLI。每個任務都包含對應的測試。

## Tasks

- [x] 1. 建立文件解析器核心架構
  - [x] 1.1 建立 `ollaforge/doc_parser.py` 基礎結構
    - 定義 `DocumentType` 列舉、`ParsedDocument` 和 `DocumentSection` 資料類別
    - 實作 `BaseParser` 抽象基底類別
    - 實作 `DocumentParserFactory` 工廠類別
    - 定義 `UnsupportedFormatError` 例外
    - _Requirements: 1.7_
  - [x] 1.2 撰寫 Property Test: 不支援格式錯誤處理
    - **Property 2: Unsupported Format Error Handling**
    - **Validates: Requirements 1.7**

- [x] 2. 實作各格式解析器
  - [x] 2.1 實作 `MarkdownParser`
    - 解析 Markdown 檔案，保留標題結構
    - 提取區段資訊 (標題層級、內容)
    - 註冊 `.md` 副檔名
    - _Requirements: 1.1_
  - [x] 2.2 實作 `HTMLParser`
    - 使用 BeautifulSoup 移除 HTML 標籤
    - 保留文字內容和基本結構
    - 註冊 `.html`, `.htm` 副檔名
    - _Requirements: 1.3_
  - [x] 2.3 實作 `TextParser`
    - 直接讀取純文字內容
    - 處理編碼偵測
    - 註冊 `.txt` 副檔名
    - _Requirements: 1.4_
  - [x] 2.4 實作 `JSONParser`
    - 遞迴提取 JSON 中的字串值
    - 處理巢狀結構
    - 註冊 `.json` 副檔名
    - _Requirements: 1.5_
  - [x] 2.5 實作 `CodeParser`
    - 提取程式碼內容
    - 偵測程式語言
    - 註冊 `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.rb` 副檔名
    - _Requirements: 1.6_
  - [x] 2.6 撰寫 Property Test: Parser 內容提取
    - **Property 1: Parser Content Extraction**
    - **Validates: Requirements 1.1, 1.3, 1.4, 1.5, 1.6**

- [x] 3. 實作 PDF 解析器
  - [x] 3.1 實作 `PDFParser`
    - 使用 PyPDF2 或 pdfplumber 提取文字
    - 處理多頁文件
    - 註冊 `.pdf` 副檔名
    - _Requirements: 1.2_
  - [x] 3.2 撰寫 PDF 解析單元測試
    - 使用測試 fixture PDF 檔案
    - 測試多頁提取
    - _Requirements: 1.2_

- [x] 4. Checkpoint - 確保解析器測試通過
  - 執行所有解析器測試
  - 確認各格式解析正確
  - 如有問題請詢問使用者

- [x] 5. 實作分塊處理器
  - [x] 5.1 建立 `ollaforge/chunk_splitter.py`
    - 定義 `ChunkConfig` 和 `TextChunk` 資料類別
    - 定義 `SplitStrategy` 列舉
    - 實作 `ChunkSplitter` 類別基礎結構
    - _Requirements: 2.1, 2.5_
  - [x] 5.2 實作固定大小分割
    - 根據 chunk_size 分割文字
    - 實作 overlap 處理
    - _Requirements: 2.1, 2.5, 2.6_
  - [x] 5.3 實作語意邊界分割
    - 偵測段落邊界
    - 偵測 Markdown 標題邊界
    - 偵測程式碼區塊邊界
    - _Requirements: 2.2, 2.3, 2.4_
  - [x] 5.4 撰寫 Property Test: Chunk 大小合規
    - **Property 3: Chunk Size Compliance**
    - **Validates: Requirements 2.1, 2.5**
  - [x] 5.5 撰寫 Property Test: 語意邊界保留
    - **Property 4: Semantic Boundary Preservation**
    - **Validates: Requirements 2.2, 2.3, 2.4**
  - [x] 5.6 撰寫 Property Test: Chunk 重疊正確性
    - **Property 5: Chunk Overlap Correctness**
    - **Validates: Requirements 2.6**

- [x] 6. Checkpoint - 確保分塊處理器測試通過
  - 執行所有分塊測試
  - 確認分塊邏輯正確
  - 如有問題請詢問使用者

- [x] 7. 實作資料集生成器
  - [x] 7.1 建立 `ollaforge/doc_generator.py`
    - 定義 `DocGenerationConfig` 資料類別
    - 實作 `DocumentDatasetGenerator` 類別
    - 整合現有 Ollama client
    - _Requirements: 3.5_
  - [x] 7.2 實作各格式生成提示
    - SFT 格式提示模板
    - Pre-training 格式提示模板
    - Conversation 格式提示模板
    - DPO 格式提示模板
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [x] 7.3 實作輸出驗證
    - 驗證 SFT 欄位非空
    - 驗證 Conversation 角色存在
    - 驗證 DPO chosen/rejected 不同
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [x] 7.4 撰寫 Property Test: 資料項目 Schema 驗證
    - **Property 6: Dataset Entry Schema Validation**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 7.1, 7.2**
  - [x] 7.5 撰寫 Property Test: Conversation 角色要求
    - **Property 7: Conversation Role Requirements**
    - **Validates: Requirements 7.3**
  - [x] 7.6 撰寫 Property Test: DPO 回應差異
    - **Property 8: DPO Response Differentiation**
    - **Validates: Requirements 7.4**
  - [x] 7.7 撰寫 Property Test: 序列化 Round-Trip
    - **Property 9: Serialization Round-Trip**
    - **Validates: Requirements 3.6**

- [x] 8. Checkpoint - 確保生成器測試通過
  - 執行所有生成器測試
  - 確認輸出格式正確
  - 如有問題請詢問使用者

- [x] 9. 擴展資料模型
  - [x] 9.1 在 `ollaforge/models.py` 新增資料模型
    - 新增 `DocToDatasetConfig` 配置模型
    - 新增 `DocProcessingResult` 結果模型
    - 新增 `BatchProcessingResult` 批次結果模型
    - 新增驗證器
    - _Requirements: 4.2, 4.3, 4.6, 4.7, 4.8, 4.9, 5.6_

- [x] 10. 實作 CLI doc2dataset 命令
  - [x] 10.1 在 `ollaforge/cli.py` 新增 `doc2dataset` 命令
    - 定義所有命令列參數
    - 實作參數驗證回呼
    - 整合解析器、分塊器、生成器
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9_
  - [x] 10.2 實作進度顯示
    - 顯示處理進度
    - 顯示預估時間
    - _Requirements: 4.10_
  - [x] 10.3 實作錯誤處理
    - 檔案不存在錯誤
    - 權限錯誤
    - Ollama 連線錯誤
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [x] 10.4 撰寫 CLI 單元測試
    - 測試參數驗證
    - 測試錯誤訊息
    - _Requirements: 4.1, 5.1, 5.2, 5.3_

- [x] 11. 實作批次處理功能
  - [x] 11.1 實作目錄遞迴處理
    - 遞迴搜尋支援的檔案
    - 實作 glob pattern 過濾
    - _Requirements: 6.1, 6.2, 6.3_
  - [x] 11.2 實作結果合併
    - 合併多檔案結果
    - 處理單一檔案失敗
    - _Requirements: 6.5, 6.6_
  - [x] 11.3 撰寫 Property Test: Pattern 過濾正確性
    - **Property 10: Pattern Filtering Correctness**
    - **Validates: Requirements 6.3**
  - [x] 11.4 撰寫 Property Test: 多檔案結果合併
    - **Property 11: Multi-File Result Aggregation**
    - **Validates: Requirements 6.5**

- [x] 12. 實作中斷處理與部分結果儲存
  - [x] 12.1 實作 SIGINT 處理
    - 捕捉中斷信號
    - 儲存已處理的結果
    - _Requirements: 5.5_
  - [x] 12.2 撰寫中斷處理單元測試
    - 測試部分結果儲存
    - _Requirements: 5.5_

- [x] 13. 整合 QC 品質控制
  - [x] 13.1 整合現有 QC 模組
    - 在生成器中加入 QC 過濾
    - 支援 zh-tw 語言驗證
    - _Requirements: 7.5, 7.6_
  - [x] 13.2 撰寫 QC 整合測試
    - 測試 QC 過濾功能
    - _Requirements: 7.5, 7.6_

- [x] 14. 更新依賴與文件
  - [x] 14.1 更新 `pyproject.toml`
    - 新增 PyPDF2/pdfplumber 依賴
    - 新增 BeautifulSoup4 依賴
    - _Requirements: 1.2, 1.3_
  - [x] 14.2 更新 README 文件
    - 新增 doc2dataset 命令說明
    - 新增使用範例
    - _Requirements: 4.1_

- [x] 15. Final Checkpoint - 確保所有測試通過
  - 執行完整測試套件
  - 確認所有功能正常運作
  - 如有問題請詢問使用者

## Notes

- 所有任務皆為必要任務，確保完整測試覆蓋
- 每個任務都參照特定需求以確保可追溯性
- Checkpoint 任務確保漸進式驗證
- Property tests 驗證通用正確性屬性
- Unit tests 驗證特定範例和邊界情況
