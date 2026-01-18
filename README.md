# DS-STAR: A Data Science Agentic Framework

DS-STAR (Data Science - Structured Thought and Action) is a Python-based agentic framework for automating data science tasks. It features a multi-agent system supporting OpenAI, Google Gemini, and Ollama models with an interactive Gradio web interface.

This project is an implementation of the paper from Google Research: [DS-STAR: A State-of-the-Art Versatile Data Science Agent](https://research.google/blog/ds-star-a-state-of-the-art-versatile-data-science-agent/). [Paper](https://arxiv.org/pdf/2509.21825)

## Live Demo

Access the web interface at: **http://127.0.0.1:7861** (after starting the application)

## Features

- **Gradio Web Interface**: User-friendly web UI with real-time execution logs, file upload support (CSV, Excel, JSON, Parquet, TSV), and data preview
- **Multi-Model Support**: Seamlessly switch between OpenAI GPT, Google Gemini, and Ollama models
- **Agentic Workflow**: 7 specialized AI agents (Analyzer, Planner, Coder, Verifier, Router, Debugger, Finalizer, Plotter) collaborate to solve data science problems
- **Real-time Streaming**: Live execution logs and progress tracking in the web interface
- **Reproducibility**: Every step is saved with prompts, code, results, and metadata for complete auditability
- **Interactive & Resume-able**: Runs can be paused and resumed from any step
- **Auto-debugging**: Intelligent error detection and automatic code fixing
- **Visualization Generation**: Automatic creation of relevant charts and plots
- **Configuration-driven**: Flexible settings through `config.yaml` and `prompt.yaml`

## How it Works

The DS-STAR pipeline is composed of several phases and agents:

1.  **Analysis**: The `Analyzer` agent inspects the initial data files and generates summaries.
2.  **Iterative Planning & Execution**:
    *   The `Planner` creates an initial plan to address the user's query.
    *   The `Coder` generates Python code to execute the current step of the plan.
    *   The code is executed, and the result is captured.
    *   An automatic `Debugger` agent attempts to fix any code that fails.
    *   The `Verifier` checks if the result sufficiently answers the query.
    *   The `Router` decides what to do next: either finalize the plan or add a new step for refinement.
    *   This loop continues until the plan is deemed sufficient or the maximum number of refinement rounds is reached.
3.  **Finalization**: The `Finalyzer` agent takes the final code and results and formats them into a clean, specified output format (e.g., JSON).

All artifacts for each run are stored in the `runs/` directory, organized by `run_id`.

## Project Structure

```
DS-Star/
├─── app.py                  # Gradio web interface with live execution logs
├─── dsstar.py               # Main pipeline with 7 specialized agents
├─── provider.py             # LLM provider integrations (OpenAI, Gemini, Ollama)
├─── config.yaml             # Main configuration file
├─── prompt.yaml             # Prompts for the different AI agents
├─── pyproject.toml          # Project metadata and dependencies
├─── uv.lock                 # Locked dependency versions
├─── .python-version         # Python version specification
├─── .env                    # API keys (create this file)
├─── data/                   # Directory for your data files
│    └─── customer_churn_data.csv
└─── runs/                   # Experiment runs and artifacts
     └─── {run_id}/
          ├─── pipeline_state.json
          ├─── logs/
          ├─── steps/
          ├─── final_output/
          └─── exec_env/
```

## Getting Started

### Prerequisites

- Python 3.11+
- API keys for your chosen provider (OpenAI, Gemini, or Ollama)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Installation

#### Using uv (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd DS-Star
    ```

2.  **Install uv (if not already installed):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Install dependencies with uv:**
    ```bash
    uv sync
    ```

### Configuration

1.  **Set your API Keys:**
    Create a `.env` file in the project root:
    ```bash
    # .env file
    OPENAI_API_KEY=your-openai-api-key
    GEMINI_API_KEY=your-gemini-api-key
    OLLAMA_API_KEY=your-ollama-api-key  # Optional
    ```

2.  **Customize `config.yaml`:**
    Configure the model and behavior settings:

    ```yaml
    # config.yaml
    model_name: 'gemini-2.0-flash-exp'  # or 'gpt-4', 'ollama/llama3'
    max_refinement_rounds: 5
    interactive: false
    preserve_artifacts: true
    ```

## Usage

### Web Interface (Recommended)

1. **Start the Gradio web application:**
   ```bash
   # Windows
   .venv\Scripts\python.exe app.py
   
   # Linux/Mac
   uv run python app.py
   ```

2. **Access the interface:**
   Open your browser to **http://127.0.0.1:7861**

3. **Upload and Analyze:**
   - Upload your data file (CSV, Excel, JSON, Parquet, TSV)
   - Enter your analysis query
   - Select your preferred model
   - Click "Start Analysis" and watch live execution logs

### Command Line Interface

Place your data files in the `data/` directory.

### Starting a New Run

To start a new analysis, you need to provide the data files and a query.

Using uv:
```bash
uv run python dsstar.py --data-files file1.xlsx file2.xlsx --query "What is the total sales for each department?"
```

### Resuming a Run

If a run was interrupted, you can resume it using its `run_id`.

```bash
uv run python dsstar.py --resume <run_id>
```

### Editing Code During a Run

You can manually edit the last generated piece of code and re-run it. This is useful for manual debugging or tweaking the agent's logic.

```bash
uv run python dsstar.py --edit-last --resume <run_id>
```
This will open the last code file in your default text editor (`nano`, `vim`, etc.). After you save and close the editor, the script will re-execute the modified code.

### Interactive Mode

To review each step before proceeding, use the interactive flag.

```bash
uv run python dsstar.py --interactive --data-files ... --query "..."
```

## UV Package Manager

This project uses `uv` for fast and reliable dependency management. Here are some useful commands:

### Common UV Commands

- **Install dependencies**: `uv sync`
- **Add a new dependency**: `uv add package-name`
- **Remove a dependency**: `uv remove package-name`
- **Update dependencies**: `uv sync --upgrade`
- **Run a command in the virtual environment**: `uv run python script.py`
- **Show installed packages**: `uv pip list`

### Benefits of UV

- **Speed**: uv is 10-100x faster than pip
- **Reliability**: Consistent dependency resolution with lock files
- **No virtual environment activation needed**: Use `uv run` to execute commands directly
- **Better dependency resolution**: Automatically resolves complex dependency conflicts

## Configuration

The following options are available in `config.yaml` and can be overridden by CLI arguments:

- `run_id` (string): The ID of a run to resume.
- `max_refinement_rounds` (int): The maximum number of times the agent will try to refine its plan.
- `api_key` (string): Your Gemini API key.
- `model_name` (string): The Gemini model to use (e.g., `gemini-1.5-flash`).
- `interactive` (bool): If true, waits for user input before executing each step.
- `auto_debug` (bool): If true, the `Debugger` agent will automatically try to fix failing code.
- `execution_timeout` (int): Timeout in seconds for code execution.
- `execution_timeout` (int): Timeout in seconds for code execution.
- `preserve_artifacts` (bool): If true, all step artifacts are saved to the `runs` directory.
- `agent_models` (dict): A dictionary mapping agent names (e.g., `PLANNER`, `CODER`) to specific model names. If not specified, `model_name` is used.

## Providers

DS-STAR supports multiple AI model providers. Each provider requires specific environment variables to be configured:

### Google Gemini

**Provider Identifier**: Default provider (no prefix required)

**Environment Variable**:
```bash
export GEMINI_API_KEY='your-gemini-api-key'
```

**Model Examples**: `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash`

### OpenAI

**Provider Identifier**: Models prefixed with `gpt` or `o1`

**Environment Variable**:
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

**Model Examples**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`, `o1-preview`

### Ollama

**Provider Identifier**: Models prefixed with `ollama/`

**Environment Variables**:
```bash
export OLLAMA_API_KEY='your-ollama-api-key'  # Optional
export OLLAMA_HOST='http://localhost:11434'  # Optional, defaults to http://localhost:11434
```

**Model Examples**: `ollama/llama3`, `ollama/qwen3-coder`, `ollama/mistral`

## Technical Stack

- **Frontend**: Gradio 5.22.0+ for interactive web interface
- **Data Processing**: Pandas 2.3.3+, support for multiple file formats
- **AI Providers**: OpenAI 2.8.1+, Google Generative AI 0.8.0+, Ollama 0.6.1+
- **Visualization**: Plotly, Matplotlib
- **Configuration**: Pydantic, PyYAML
- **Environment**: Python 3.11+ with UV package manager


```
