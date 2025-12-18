# Requirements Document

## Introduction

本功能為 OllaForge 新增資料集擴展（Dataset Augmentation）能力。使用者可以載入現有的 JSONL 資料集，指定目標欄位，並透過 AI 模型對該欄位進行擴展、增強或轉換。這使得使用者能夠基於現有資料快速擴充資料集特徵，而非僅能從零開始生成。

## Glossary

- **Augmentation System**: 負責讀取現有資料集、處理使用者指令、呼叫 AI 模型並輸出擴展後資料的系統
- **Source Dataset**: 使用者提供的原始 JSONL 格式資料集檔案
- **Target Field**: 使用者指定要進行擴展或修改的欄位名稱
- **Augmentation Instruction**: 使用者提供給 AI 的指令，描述如何處理目標欄位
- **Augmented Entry**: 經過 AI 處理後的單筆資料記錄

## Requirements

### Requirement 1

**User Story:** As a data engineer, I want to load an existing JSONL dataset file, so that I can use it as the basis for augmentation.

#### Acceptance Criteria

1. WHEN a user provides a valid JSONL file path THEN the Augmentation System SHALL read and parse all entries from the file
2. WHEN a user provides a non-existent file path THEN the Augmentation System SHALL display a clear error message indicating the file was not found
3. WHEN a user provides a file with invalid JSONL format THEN the Augmentation System SHALL report the line number and nature of the parsing error
4. WHEN the source dataset is loaded successfully THEN the Augmentation System SHALL display the total number of entries and available field names

### Requirement 2

**User Story:** As a data engineer, I want to specify which field to augment and provide instructions, so that I can control how the AI modifies my data.

#### Acceptance Criteria

1. WHEN a user specifies a target field that exists in the dataset THEN the Augmentation System SHALL accept the field selection
2. WHEN a user specifies a target field that does not exist in any entry THEN the Augmentation System SHALL display an error listing available fields
3. WHEN a user provides an augmentation instruction THEN the Augmentation System SHALL use that instruction to guide the AI model's processing
4. WHEN a user wants to add a new field THEN the Augmentation System SHALL allow specifying a new field name and generation instruction

### Requirement 3

**User Story:** As a data engineer, I want the AI to process each entry and augment the specified field, so that I can enhance my dataset with AI-generated content.

#### Acceptance Criteria

1. WHEN processing an entry THEN the Augmentation System SHALL send the entry context and augmentation instruction to the AI model
2. WHEN the AI model returns a response THEN the Augmentation System SHALL parse the response and update the target field
3. WHEN the AI model fails to generate valid content THEN the Augmentation System SHALL retain the original entry unchanged and log the failure
4. WHEN processing entries THEN the Augmentation System SHALL serialize augmented entries to JSON and deserialize AI responses from JSON (round-trip consistency)
5. WHEN processing multiple entries THEN the Augmentation System SHALL support concurrent processing with configurable parallelism

### Requirement 4

**User Story:** As a data engineer, I want to save the augmented dataset to a new file, so that I can preserve both the original and enhanced versions.

#### Acceptance Criteria

1. WHEN augmentation completes THEN the Augmentation System SHALL write all entries to the specified output file in JSONL format
2. WHEN the output file already exists THEN the Augmentation System SHALL prompt for confirmation before overwriting
3. WHEN writing fails due to disk space or permissions THEN the Augmentation System SHALL display a clear error message
4. WHEN augmentation is interrupted THEN the Augmentation System SHALL save successfully processed entries to a partial output file

### Requirement 5

**User Story:** As a data engineer, I want to see progress and statistics during augmentation, so that I can monitor the process.

#### Acceptance Criteria

1. WHILE augmentation is in progress THEN the Augmentation System SHALL display a progress bar showing completed entries
2. WHEN augmentation completes THEN the Augmentation System SHALL display a summary including total entries, successful augmentations, and failures
3. WHEN errors occur during processing THEN the Augmentation System SHALL accumulate error counts and display them in the final summary

### Requirement 6

**User Story:** As a data engineer, I want to use the augmentation feature through CLI commands, so that I can integrate it into my workflow.

#### Acceptance Criteria

1. WHEN a user runs the augment subcommand THEN the Augmentation System SHALL accept required parameters: input file, target field, and instruction
2. WHEN a user runs the augment subcommand with --help THEN the Augmentation System SHALL display usage information and examples
3. WHEN a user omits required parameters THEN the Augmentation System SHALL display which parameters are missing
4. WHERE the user enables interactive mode THEN the Augmentation System SHALL guide the user through field selection and instruction input step by step

### Requirement 7

**User Story:** As a data engineer, I want to preview augmentation results before processing the entire dataset, so that I can verify the AI understands my instructions.

#### Acceptance Criteria

1. WHEN a user requests a preview THEN the Augmentation System SHALL process a configurable number of sample entries (default: 3)
2. WHEN displaying preview results THEN the Augmentation System SHALL show the original and augmented values side by side
3. WHEN the user confirms the preview THEN the Augmentation System SHALL proceed with full dataset processing
4. WHEN the user rejects the preview THEN the Augmentation System SHALL allow the user to modify the instruction and retry
