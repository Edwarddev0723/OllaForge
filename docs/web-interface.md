# OllaForge Web Interface User Guide

This guide covers how to use the OllaForge web interface for generating and augmenting LLM training datasets.

## Table of Contents

- [Getting Started](#getting-started)
- [Dataset Generation](#dataset-generation)
- [Dataset Augmentation](#dataset-augmentation)
- [Configuration Management](#configuration-management)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

Before using the web interface, ensure you have:

1. **Ollama installed and running**: The web interface uses Ollama for LLM inference
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **At least one model pulled**: Download a model if you haven't already
   ```bash
   ollama pull llama3.2
   ```

### Starting the Web Interface

#### Option 1: Development Mode

Start the backend and frontend separately:

```bash
# Terminal 1: Start backend
python -m ollaforge.web.server

# Terminal 2: Start frontend
cd ollaforge-web
npm run dev
```

Access the interface at `http://localhost:5173`

#### Option 2: Docker (Production)

```bash
docker-compose up
```

Access the interface at `http://localhost:80`

### Interface Overview

The web interface has three main sections:

1. **Generate**: Create new training datasets from topic descriptions
2. **Augment**: Modify or add fields to existing datasets
3. **Config**: Manage saved configurations

## Dataset Generation

### Creating a New Dataset

1. Navigate to the **Generate** page
2. Fill in the generation form:

   | Field | Description | Example |
   |-------|-------------|---------|
   | Topic | Description of the dataset content | "Customer service conversations for an e-commerce platform" |
   | Count | Number of entries to generate (1-10,000) | 100 |
   | Model | Ollama model to use | llama3.2 |
   | Dataset Type | Format of the generated data | SFT (Supervised Fine-tuning) |
   | Language | Output language | English or Traditional Chinese |

3. Click **Generate** to start

### Dataset Types

| Type | Description | Output Format |
|------|-------------|---------------|
| **SFT** | Supervised Fine-tuning | `{instruction, input, output}` |
| **Pretrain** | Pre-training data | `{text}` |
| **SFT Conv** | Conversational SFT | `{conversations: [{role, content}]}` |
| **DPO** | Direct Preference Optimization | `{prompt, chosen, rejected}` |

### Quality Control (QC)

When generating Traditional Chinese content, you can enable QC filtering:

- **QC Enabled**: Filters out low-quality or non-Traditional Chinese content
- **QC Confidence**: Threshold for filtering (0.0-1.0, default 0.9)

Higher confidence values result in stricter filtering.

### Monitoring Progress

During generation:
- A progress bar shows completion percentage
- Real-time updates display the number of entries generated
- Estimated time remaining is calculated based on current speed

### Downloading Results

After generation completes:

1. Preview the first 5 entries in the results table
2. Select your preferred download format:
   - **JSONL** (default): One JSON object per line
   - **JSON**: Standard JSON array
   - **CSV**: Comma-separated values
   - **TSV**: Tab-separated values
   - **Parquet**: Apache Parquet binary format

3. Click **Download** to save the file

## Dataset Augmentation

### Uploading a Dataset

1. Navigate to the **Augment** page
2. Upload your dataset file by:
   - Dragging and dropping onto the upload area
   - Clicking to browse and select a file

Supported formats: JSONL, JSON, CSV, TSV, Parquet

After upload, you'll see:
- Total number of entries
- List of available fields
- Preview of the first 3 entries

### Configuring Augmentation

Fill in the augmentation form:

| Field | Description | Example |
|-------|-------------|---------|
| Target Field | Field to modify or create | "output" |
| Instruction | What to do with the field | "Make the response more formal and professional" |
| Model | Ollama model to use | llama3.2 |
| Create New Field | Create a new field instead of modifying | ☐ |
| Context Fields | Additional fields to provide as context | instruction, input |
| Concurrency | Number of parallel requests (1-20) | 5 |

### Previewing Changes

Before processing the entire dataset:

1. Click **Preview** to see sample augmentations
2. Review the before/after comparison for 3 entries
3. Adjust your instruction if needed
4. Click **Start Augmentation** when satisfied

### Handling Partial Failures

If some entries fail during augmentation:
- Original data is preserved for failed entries
- Success/failure statistics are displayed
- You can download the partial results

## Configuration Management

### Saving Configurations

After completing a generation or augmentation:

1. A prompt appears asking if you want to save the configuration
2. Enter a descriptive name (e.g., "Customer Service SFT - English")
3. Click **Save**

Configurations are stored in your browser's local storage.

### Loading Configurations

1. Navigate to the **Config** page
2. Browse your saved configurations
3. Click **Load** to populate the form with saved values
4. Modify if needed and run the operation

### Managing Configurations

On the Config page, you can:
- View all saved configurations with timestamps
- Load a configuration to use it
- Delete configurations you no longer need

## Troubleshooting

### Common Issues

#### "Unable to connect to Ollama service"

**Cause**: Ollama is not running or not accessible

**Solution**:
1. Start Ollama: `ollama serve`
2. Verify it's running: `curl http://localhost:11434/api/tags`
3. Check if a firewall is blocking the connection

#### "Model not found"

**Cause**: The selected model is not installed

**Solution**:
1. List available models: `ollama list`
2. Pull the required model: `ollama pull <model_name>`

#### "File format not supported"

**Cause**: Uploaded file has an unsupported extension

**Solution**: Convert your file to one of the supported formats:
- JSONL, JSON, CSV, TSV, or Parquet

#### Generation is slow

**Possible causes and solutions**:
1. **Large model**: Use a smaller model (e.g., llama3.2 instead of llama3.1:70b)
2. **High count**: Generate in smaller batches
3. **System resources**: Close other applications using GPU/CPU

#### WebSocket connection failed

**Cause**: Real-time progress updates are not working

**Solution**:
1. Refresh the page
2. Check if the backend is running
3. Verify no proxy is blocking WebSocket connections

### Getting Help

- **API Documentation**: Access `/docs` on the backend server for Swagger UI
- **GitHub Issues**: Report bugs or request features
- **Logs**: Check browser console and backend logs for detailed error messages

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + Enter` | Submit form |
| `Escape` | Cancel current operation |

## Language Support

The interface supports:
- **English** (default)
- **繁體中文** (Traditional Chinese)

To change language:
1. Click the language selector in the header
2. Select your preferred language
3. The interface updates immediately

Your language preference is saved and remembered across sessions.
