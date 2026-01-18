import gradio as gr
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
import threading
import queue
import pandas as pd
import sys
import io
import logging
from dsstar import DS_STAR_Agent, DSConfig

# Global variables for streaming output
output_queue = queue.Queue()
current_run = None

class LogCapture(io.StringIO):
    """Capture logs and store them for streaming."""
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def write(self, message):
        if message.strip():
            self.logs.append(message)
        super().write(message)
    
    def get_logs(self):
        return ''.join(self.logs)

def list_data_files():
    """List all data files in the data directory."""
    data_dir = Path("data")
    if data_dir.exists():
        # Support multiple file formats
        extensions = ['*.csv', '*.xlsx', '*.xls', '*.json', '*.parquet', '*.txt', '*.tsv']
        files = []
        for ext in extensions:
            files.extend([f.name for f in data_dir.glob(ext)])
        return sorted(files)
    return []

def upload_data_file(file):
    """Upload a data file to the data directory."""
    if file is None:
        return "No file uploaded.", None, []
    
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Get the filename
        filename = Path(file.name).name
        dest_path = data_dir / filename
        
        # Validate file extension
        valid_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt', '.tsv']
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            return f"❌ Unsupported file format. Supported: {', '.join(valid_extensions)}", list_data_files(), []
        
        # Copy the file
        import shutil
        shutil.copy(file.name, dest_path)
        
        return f"✅ File uploaded successfully: {filename}", list_data_files(), [filename]
    except Exception as e:
        return f"❌ Error uploading file: {str(e)}", list_data_files(), []

def preview_data(selected_files):
    """Preview the selected data files."""
    if not selected_files:
        return None, "No files selected."
    
    try:
        data_dir = Path("data")
        preview_text = ""
        dataframes = []
        
        for filename in selected_files:
            filepath = data_dir / filename
            if filepath.exists():
                # Read file based on extension
                file_ext = filepath.suffix.lower()
                df = None
                
                try:
                    if file_ext == '.csv':
                        df = pd.read_csv(filepath)
                    elif file_ext in ['.xlsx', '.xls']:
                        df = pd.read_excel(filepath)
                    elif file_ext == '.json':
                        df = pd.read_json(filepath)
                    elif file_ext == '.parquet':
                        df = pd.read_parquet(filepath)
                    elif file_ext in ['.txt', '.tsv']:
                        df = pd.read_csv(filepath, sep='\t')
                    else:
                        preview_text += f"\n⚠️ Unsupported format: {filename}\n"
                        continue
                    
                    dataframes.append(df)
                    
                    preview_text += f"\n{'='*60}\n"
                    preview_text += f"📄 File: {filename} ({file_ext[1:].upper()})\n"
                    preview_text += f"{'='*60}\n"
                    preview_text += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                    preview_text += f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
                    preview_text += f"Columns:\n"
                    for i, col in enumerate(df.columns, 1):
                        dtype = df[col].dtype
                        null_count = df[col].isnull().sum()
                        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
                        preview_text += f"  {i}. {col} ({dtype}) - {null_count} nulls ({null_pct:.1f}%)\n"
                    preview_text += f"\n"
                    
                except Exception as read_error:
                    preview_text += f"\n❌ Error reading {filename}: {str(read_error)}\n"
        
        # Return the first dataframe for display
        if dataframes:
            return dataframes[0].head(100), preview_text
        return None, preview_text
        
    except Exception as e:
        return None, f"❌ Error previewing files: {str(e)}"

def list_previous_runs():
    """List all previous run directories."""
    runs_dir = Path("runs")
    if runs_dir.exists():
        runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
        return runs
    return []

def load_config():
    """Load default configuration from config.yaml."""
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f) or {}
        return config
    except FileNotFoundError:
        return {}

def json_to_english(result):
    """Convert JSON result to readable English sentences."""
    if not result:
        return "No results available."
    
    try:
        # If result is a string, try to parse it as JSON
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                return result
        
        if not isinstance(result, dict):
            return str(result)
        
        # Convert to English sentences
        sentences = []
        
        # Handle common result structures
        if 'summary' in result:
            sentences.append(f"📋 Summary: {result['summary']}")
        
        if 'insights' in result:
            insights = result['insights']
            if isinstance(insights, list):
                sentences.append("\n🔍 Key Insights:")
                for i, insight in enumerate(insights, 1):
                    sentences.append(f"  {i}. {insight}")
            else:
                sentences.append(f"🔍 Insights: {insights}")
        
        if 'recommendations' in result:
            recs = result['recommendations']
            if isinstance(recs, list):
                sentences.append("\n💡 Recommendations:")
                for i, rec in enumerate(recs, 1):
                    sentences.append(f"  {i}. {rec}")
            else:
                sentences.append(f"💡 Recommendations: {recs}")
        
        if 'metrics' in result or 'performance' in result:
            metrics = result.get('metrics') or result.get('performance')
            sentences.append("\n📊 Performance Metrics:")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    sentences.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
        if 'model' in result:
            sentences.append(f"\n🤖 Model: {result['model']}")
        
        if 'accuracy' in result:
            sentences.append(f"  • Accuracy: {result['accuracy']}")
        
        if 'predictions' in result:
            preds = result['predictions']
            if isinstance(preds, (list, dict)):
                sentences.append(f"\n🎯 Predictions: {len(preds) if isinstance(preds, list) else 'Available'}")
        
        # If no specific fields found, convert all key-value pairs
        if not sentences:
            sentences.append("📄 Analysis Results:")
            for key, value in result.items():
                if not isinstance(value, (dict, list)):
                    sentences.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(sentences) if sentences else str(result)
        
    except Exception as e:
        return f"Error converting result: {str(e)}\n\nRaw output:\n{str(result)}"


def run_analysis(query, selected_files, model_name, max_rounds, interactive, resume_run, progress=gr.Progress()):
    """Run the DS-STAR analysis pipeline with real-time logging."""
    if not query and not resume_run:
        yield "❌ Error: Please provide an analysis query.", None, None, None
        return
    
    if not selected_files and not resume_run:
        yield "❌ Error: Please select at least one data file.", None, None, None
        return
    
    try:
        # Prepare configuration
        config_params = {
            'interactive': interactive,
            'max_refinement_rounds': max_rounds,
            'model_name': model_name,
            'preserve_artifacts': True
        }
        
        if resume_run:
            config_params['run_id'] = resume_run
        
        # Filter out None values
        config_params = {k: v for k, v in config_params.items() if v is not None}
        
        config = DSConfig(**config_params)
        
        # Initial status
        output_text = f"🚀 Starting DS-STAR Analysis...\n"
        output_text += f"📊 Model: {model_name}\n"
        output_text += f"🔄 Max Refinement Rounds: {max_rounds}\n"
        output_text += f"📁 Run ID: {config.run_id}\n\n"
        
        if resume_run:
            output_text += f"♻️ Resuming from previous run: {resume_run}\n\n"
        else:
            output_text += f"📂 Data Files: {', '.join(selected_files)}\n"
            output_text += f"❓ Query: {query}\n\n"
        
        progress(0, desc="Initializing...")
        output_text += "⏳ Initializing pipeline...\n\n"
        
        # Yield initial status
        yield output_text, None, config.run_id, ""
        
        # Create agent
        agent = DS_STAR_Agent(config)
        
        output_text += "✅ Agent initialized\n\n"
        yield output_text, None, config.run_id, ""
        
        # Prepare log file path
        log_file = Path(config.runs_dir) / config.run_id / "logs" / "pipeline.log"
        
        # Start the pipeline in a thread
        import threading
        result_container = {}
        
        def run_pipeline_thread():
            try:
                if resume_run:
                    result_container['result'] = {"status": "resumed", "run_id": resume_run}
                else:
                    result_container['result'] = agent.run_pipeline(query, selected_files)
                result_container['done'] = True
            except Exception as e:
                result_container['error'] = str(e)
                result_container['done'] = True
        
        pipeline_thread = threading.Thread(target=run_pipeline_thread)
        pipeline_thread.start()
        
        # Stream logs while pipeline runs
        import time
        last_position = 0
        log_header_added = False
        
        while not result_container.get('done', False):
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        f.seek(last_position)
                        new_content = f.read()
                        if new_content:
                            if not log_header_added:
                                output_text += "\n" + "="*60 + "\n"
                                output_text += "📋 LIVE EXECUTION LOG:\n"
                                output_text += "="*60 + "\n\n"
                                log_header_added = True
                            output_text += new_content
                            last_position = f.tell()
                            yield output_text, None, config.run_id, ""
                except Exception as e:
                    pass
            time.sleep(0.5)  # Check for updates every 0.5 seconds
        
        # Wait for thread to complete
        pipeline_thread.join()
        
        # Read any remaining logs
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    f.seek(last_position)
                    remaining_content = f.read()
                    if remaining_content:
                        output_text += remaining_content
            except Exception as e:
                pass
        
        # Check for errors
        if 'error' in result_container:
            output_text += f"\n\n❌ Error during execution: {result_container['error']}\n"
            yield output_text, None, config.run_id, ""
            return
        
        result = result_container.get('result', {})
        
        if log_header_added:
            output_text += "\n" + "="*60 + "\n\n"
        
        # Format results
        output_text += "="*60 + "\n"
        output_text += "✅ ANALYSIS COMPLETED!\n"
        output_text += "="*60 + "\n\n"
        output_text += f"📋 Run ID: {result['run_id']}\n"
        output_text += f"📄 Output File: {result.get('output_file', 'N/A')}\n\n"
        
        # Convert final result to English
        final_result = result.get('final_result', {})
        english_result = json_to_english(final_result)
        
        output_text += "🎯 FINAL RESULT:\n"
        output_text += "-"*60 + "\n"
        output_text += english_result
        output_text += "\n" + "="*60 + "\n"
        
        # Load the output file if available
        output_file = result.get('output_file')
        output_content = None
        if output_file and Path(output_file).exists():
            try:
                with open(output_file, 'r') as f:
                    output_content = f.read()
            except Exception as e:
                output_content = f"Error reading output file: {str(e)}"
        
        yield output_text, output_content, result['run_id'], english_result
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}\n\n"
        error_msg += "Please check your configuration and try again."
        yield error_msg, None, None, None

def view_run_details(run_id):
    """View details of a previous run."""
    if not run_id:
        return "Please select a run to view.", None
    
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        return f"Run directory not found: {run_id}", None
    
    # Load pipeline state
    state_file = run_dir / "pipeline_state.json"
    output_text = f"📊 Run Details: {run_id}\n"
    output_text += "="*60 + "\n\n"
    
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            output_text += f"📈 Current Step: {state.get('current_step', 'N/A')}\n"
            output_text += f"✓ Completed Steps: {len(state.get('completed_steps', []))}\n"
            output_text += f"📋 Plan Steps: {len(state.get('plan', []))}\n\n"
            
            if state.get('plan'):
                output_text += "📝 Execution Plan:\n"
                for i, step in enumerate(state['plan'], 1):
                    output_text += f"  {i}. {step}\n"
                output_text += "\n"
            
            if state.get('completed_steps'):
                output_text += "✅ Completed Steps:\n"
                for step in state['completed_steps']:
                    output_text += f"  • {step}\n"
                output_text += "\n"
        except Exception as e:
            output_text += f"Error loading state: {str(e)}\n\n"
    
    # Check for final output
    final_output_dir = run_dir / "final_output"
    output_content = None
    
    if final_output_dir.exists():
        result_file = final_output_dir / "result.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    output_content = json.dumps(json.load(f), indent=2)
            except Exception as e:
                output_content = f"Error reading result: {str(e)}"
    
    return output_text, output_content

# Load default config
default_config = load_config()

# Create Gradio interface
with gr.Blocks(title="DS-STAR Data Science Agent") as app:
    gr.Markdown(
        """
        # 🌟 DS-STAR: Data Science Self-Testing Agent with Refinement
        
        An AI-powered data science pipeline that analyzes data, generates insights, and produces reproducible results.
        """
    )
    
    with gr.Tabs():
        # Main Analysis Tab
        with gr.Tab("🚀 New Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Analysis Configuration")
                    
                    query_input = gr.Textbox(
                        label="Analysis Query",
                        placeholder="e.g., Predict customer churn and identify key factors",
                        lines=3,
                        value=default_config.get('query', '')
                    )
                    
                    gr.Markdown("### 📂 Data Files")
                    
                    with gr.Row():
                        upload_file = gr.File(
                            label="Upload Data File",
                            file_types=[".csv", ".xlsx", ".xls", ".json", ".parquet", ".txt", ".tsv"],
                            type="filepath"
                        )
                        upload_btn = gr.Button("📤 Upload", size="sm")
                    
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        interactive=False,
                        show_label=False,
                        visible=True
                    )
                    
                    data_files_dropdown = gr.Dropdown(
                        choices=list_data_files(),
                        label="Select Data Files",
                        multiselect=True,
                        value=default_config.get('data_files', [])
                    )
                    
                    with gr.Row():
                        refresh_files_btn = gr.Button("🔄 Refresh File List", size="sm")
                        preview_btn = gr.Button("👁️ Preview Data", size="sm", variant="secondary")
                    
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "gpt-4-turbo-preview",
                                "gpt-4",
                                "gpt-3.5-turbo",
                                "gemini-pro",
                                "ollama/llama2"
                            ],
                            label="Model",
                            value=default_config.get('model_name', 'gpt-4-turbo-preview')
                        )
                        
                        max_rounds_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=default_config.get('max_refinement_rounds', 3),
                            label="Max Refinement Rounds"
                        )
                    
                    interactive_checkbox = gr.Checkbox(
                        label="Interactive Mode (pause between steps)",
                        value=default_config.get('interactive', False)
                    )
                    
                    run_btn = gr.Button("▶️ Run Analysis", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Output")
                    
                    # Data Preview Section
                    with gr.Accordion("📊 Data Preview", open=False):
                        preview_info = gr.Textbox(
                            label="Data Information",
                            lines=10,
                            interactive=False
                        )
                        preview_df = gr.Dataframe(
                            label="Sample Data (first 100 rows)",
                            interactive=False,
                            wrap=True
                        )
                    
                    output_text = gr.Textbox(
                        label="Execution Log",
                        lines=15,
                        max_lines=25,
                        interactive=False
                    )
                    
                    result_summary = gr.Textbox(
                        label="📝 Result Summary (Plain English)",
                        lines=10,
                        interactive=False
                    )
                    
                    run_id_output = gr.Textbox(
                        label="Run ID",
                        interactive=False
                    )
            
            with gr.Row():
                output_file_content = gr.Code(
                    label="📄 Generated Output File (JSON)",
                    language="json",
                    lines=15,
                    interactive=False
                )
        
        # Resume Tab
        with gr.Tab("♻️ Resume Run"):
            gr.Markdown("### Resume a Previous Run")
            
            with gr.Row():
                with gr.Column(scale=1):
                    resume_run_dropdown = gr.Dropdown(
                        choices=list_previous_runs(),
                        label="Select Run to Resume",
                        interactive=True
                    )
                    
                    refresh_runs_btn = gr.Button("🔄 Refresh Run List", size="sm")
                    
                    resume_btn = gr.Button("▶️ Resume Run", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    resume_output_text = gr.Textbox(
                        label="Execution Log",
                        lines=20,
                        max_lines=30,
                        interactive=False
                    )
        
        # View Results Tab
        with gr.Tab("📂 View Results"):
            gr.Markdown("### View Previous Run Results")
            
            with gr.Row():
                with gr.Column(scale=1):
                    view_run_dropdown = gr.Dropdown(
                        choices=list_previous_runs(),
                        label="Select Run to View",
                        interactive=True
                    )
                    
                    refresh_view_runs_btn = gr.Button("🔄 Refresh Run List", size="sm")
                    
                    view_btn = gr.Button("👁️ View Details", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    view_output_text = gr.Textbox(
                        label="Run Details",
                        lines=15,
                        interactive=False
                    )
            
            with gr.Row():
                view_output_content = gr.Code(
                    label="📄 Final Output",
                    language="json",
                    lines=15,
                    interactive=False
                )
    
    # Event handlers
    upload_btn.click(
        fn=upload_data_file,
        inputs=[upload_file],
        outputs=[upload_status, data_files_dropdown, data_files_dropdown]
    )
    
    preview_btn.click(
        fn=preview_data,
        inputs=[data_files_dropdown],
        outputs=[preview_df, preview_info]
    )
    
    refresh_files_btn.click(
        fn=lambda: gr.Dropdown(choices=list_data_files()),
        outputs=data_files_dropdown
    )
    
    refresh_runs_btn.click(
        fn=lambda: gr.Dropdown(choices=list_previous_runs()),
        outputs=resume_run_dropdown
    )
    
    refresh_view_runs_btn.click(
        fn=lambda: gr.Dropdown(choices=list_previous_runs()),
        outputs=view_run_dropdown
    )
    
    run_btn.click(
        fn=run_analysis,
        inputs=[
            query_input,
            data_files_dropdown,
            model_dropdown,
            max_rounds_slider,
            interactive_checkbox,
            gr.State(None)  # No resume for new runs
        ],
        outputs=[output_text, output_file_content, run_id_output, result_summary]
    )
    
    resume_btn.click(
        fn=run_analysis,
        inputs=[
            gr.State(""),  # No query needed for resume
            gr.State([]),  # No files needed for resume
            model_dropdown,
            max_rounds_slider,
            interactive_checkbox,
            resume_run_dropdown
        ],
        outputs=[resume_output_text, gr.State(), gr.State()]
    )
    
    view_btn.click(
        fn=view_run_details,
        inputs=[view_run_dropdown],
        outputs=[view_output_text, view_output_content]
    )

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Starting DS-STAR Gradio App...")
        print("=" * 60)
        print("\nPlease wait while loading dependencies...")
        print("This may take 30-60 seconds on first run.\n")
        
        app.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            show_error=True,
            inbrowser=False,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\n\nApp stopped by user.")
    except Exception as e:
        print(f"\n\nError starting app: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")

