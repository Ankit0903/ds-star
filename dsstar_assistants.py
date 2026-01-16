import os
import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys
import yaml
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

with open("prompt.yaml", "r") as f:
    PROMPT_TEMPLATES = yaml.safe_load(f)

@dataclass
class DSConfig:
    """Centralized configuration for the entire pipeline."""
    run_id: str = None
    max_refinement_rounds: int = 5
    model_name: str = "gpt-4-turbo-preview"  # Assistants API compatible model
    interactive: bool = False
    preserve_artifacts: bool = True
    runs_dir: str = "runs"
    data_dir: str = "data"
    
    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================================
# ARTIFACT STORAGE SYSTEM
# =============================================================================

class ArtifactStorage:
    """Persistently stores every step of the pipeline for reproducibility."""
    
    def __init__(self, config: DSConfig):
        self.config = config
        self.run_dir = Path(config.runs_dir) / config.run_id
        self._setup_directories()
        
    def _setup_directories(self):
        """Create directory structure for this run."""
        dirs = [
            self.run_dir,
            self.run_dir / "steps",
            self.run_dir / "logs",
            self.run_dir / "final_output"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            
    def save_step(self, step_id: str, step_type: str, prompt: str, 
                  result: str, metadata: Dict[str, Any]):
        """Save all artifacts for a single pipeline step."""
        step_dir = self.run_dir / "steps" / step_id
        step_dir.mkdir(exist_ok=True)
        
        (step_dir / "prompt.md").write_text(prompt, encoding='utf-8')
        (step_dir / "result.txt").write_text(result, encoding='utf-8')
        
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "step_id": step_id
        })
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    def get_current_state(self) -> Dict[str, Any]:
        """Load the pipeline state."""
        state_file = self.run_dir / "pipeline_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())
        return {
            "current_step": 0, 
            "completed_steps": [], 
            "thread_id": None,
            "assistant_ids": {},
            "file_ids": []
        }
    
    def save_state(self, state: Dict[str, Any]):
        """Save the pipeline state."""
        state_file = self.run_dir / "pipeline_state.json"
        state_file.write_text(json.dumps(state, indent=2))

# =============================================================================
# OPENAI ASSISTANTS API WRAPPER
# =============================================================================

class AssistantAgent:
    """Wrapper for OpenAI Assistant with specific role."""
    
    def __init__(self, client: OpenAI, name: str, instructions: str, 
                 model: str = "gpt-4-turbo-preview", tools: List[Dict] = None):
        self.client = client
        self.name = name
        self.tools = tools or []
        
        # Create or retrieve assistant
        self.assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model,
            tools=self.tools
        )
        
    def run(self, thread_id: str, message: str, file_ids: List[str] = None) -> str:
        """Run the assistant on a thread with a message."""
        # Add message to thread
        msg_params = {"role": "user", "content": message}
        if file_ids:
            msg_params["attachments"] = [
                {"file_id": fid, "tools": [{"type": "code_interpreter"}]} 
                for fid in file_ids
            ]
        
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            **msg_params
        )
        
        # Run the assistant with retry logic
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                run = self.client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=self.assistant.id
                )
                
                # Wait for completion
                while run.status in ["queued", "in_progress"]:
                    time.sleep(1)
                    run = self.client.beta.threads.runs.retrieve(
                        thread_id=thread_id,
                        run_id=run.id
                    )
                
                if run.status == "failed":
                    error_msg = str(run.last_error)
                    # Check if it's a rate limit error
                    if "rate_limit_exceeded" in error_msg:
                        # Extract wait time from error message or use exponential backoff
                        wait_time = retry_delay * (2 ** attempt)
                        if attempt < max_retries - 1:
                            print(f"Rate limit exceeded. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                            time.sleep(wait_time)
                            continue
                    raise Exception(f"Assistant run failed: {run.last_error}")
                
                # Success - break out of retry loop
                break
                
            except Exception as e:
                error_msg = str(e)
                if "rate_limit_exceeded" in error_msg or "Rate limit" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Rate limit error. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                raise
        
        # Get the response
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=1
        )
        
        # Extract text content from the message (handle both text and image blocks)
        response_text = ""
        for content_block in messages.data[0].content:
            if hasattr(content_block, 'text'):
                response_text += content_block.text.value
            elif hasattr(content_block, 'image_file'):
                # Image generated, add a note
                response_text += f"\n[Image generated: {content_block.image_file.file_id}]\n"
        
        return response_text if response_text else "Code executed successfully."

# =============================================================================
# DS-STAR AGENT WITH ASSISTANTS API
# =============================================================================

class DS_STAR_Agent:
    """DS-STAR agent using OpenAI Assistants API."""
    
    def __init__(self, config: DSConfig):
        self.config = config
        self.storage = ArtifactStorage(config)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Setup logging
        import logging
        log_file = self.storage.run_dir / "logs" / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load or create thread
        state = self.storage.get_current_state()
        if state["thread_id"]:
            self.thread_id = state["thread_id"]
            self.logger.info(f"Resuming thread: {self.thread_id}")
        else:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
            state["thread_id"] = self.thread_id
            self.storage.save_state(state)
            self.logger.info(f"Created new thread: {self.thread_id}")
        
        # Create specialized assistants
        self._create_assistants()
        
        # Uploaded files
        self.file_ids = state.get("file_ids", [])
    
    def _create_assistants(self):
        """Create specialized assistants for each role."""
        self.assistants = {}
        
        # Generic assistants with minimal instructions - rely on runtime prompts
        self.assistants["ANALYZER"] = AssistantAgent(
            self.client,
            name="Data Analyzer",
            instructions="You are an expert data analyst. Follow the instructions provided in each request.",
            model=self.config.model_name,
            tools=[{"type": "code_interpreter"}]
        )
        
        self.assistants["PLANNER"] = AssistantAgent(
            self.client,
            name="Analysis Planner",
            instructions="You are an expert at planning data analysis tasks. Follow the instructions provided in each request.",
            model=self.config.model_name
        )
        
        self.assistants["CODER"] = AssistantAgent(
            self.client,
            name="Code Generator",
            instructions="You are an expert Python programmer for data analysis. Follow the instructions provided in each request.",
            model=self.config.model_name,
            tools=[{"type": "code_interpreter"}]
        )
        
        self.assistants["VERIFIER"] = AssistantAgent(
            self.client,
            name="Solution Verifier",
            instructions="You verify analysis results. Follow the instructions provided in each request.",
            model=self.config.model_name
        )
        
        self.assistants["FINALYZER"] = AssistantAgent(
            self.client,
            name="Output Finalizer",
            instructions="You format analysis results. Follow the instructions provided in each request.",
            model=self.config.model_name
        )
        
        self.logger.info("Created specialized assistants")
    
    def upload_files(self, data_files: List[str]) -> List[str]:
        """Upload data files to OpenAI."""
        file_ids = []
        for filepath in data_files:
            full_path = Path(self.config.data_dir) / filepath
            if not full_path.exists():
                full_path = Path(filepath)
            
            if not full_path.exists():
                self.logger.error(f"File not found: {filepath}")
                continue
            
            self.logger.info(f"Uploading {filepath}...")
            with open(full_path, "rb") as f:
                file = self.client.files.create(file=f, purpose="assistants")
                file_ids.append(file.id)
                self.logger.info(f"Uploaded: {file.id}")
        
        return file_ids
    
    def analyze_data(self, file_ids: List[str]) -> str:
        """Analyze uploaded data files."""
        # Use a generic prompt since we don't have specific filenames for uploaded files
        prompt = """Analyze the uploaded data files. For each file:
        - Describe the structure and format
        - List all column names and data types
        - Provide summary statistics
        - Note any important patterns or characteristics
        Print all essential information."""
        
        result = self.assistants["ANALYZER"].run(
            self.thread_id, 
            prompt,
            file_ids=file_ids
        )
        
        state = self.storage.get_current_state()
        step_id = f"{len(state['completed_steps']):03d}_analyzer"
        self.storage.save_step(
            step_id=step_id,
            step_type="analyzer",
            prompt=prompt,
            result=result,
            metadata={"file_ids": file_ids}
        )
        
        state["completed_steps"].append(step_id)
        self.storage.save_state(state)
        
        return result
    
    def plan_analysis(self, query: str, data_desc: str, current_plan: List[str] = None) -> str:
        """Generate next analysis step."""
        if not current_plan:
            # Use planner_init template
            prompt = PROMPT_TEMPLATES.get("planner_init", """
            Question: {question}
            Data available:
            {summaries}
            
            Suggest your very first step to answer the question above.
            """).format(question=query, summaries=data_desc)
        else:
            # Use planner_next template
            plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(current_plan))
            prompt = PROMPT_TEMPLATES.get("planner_next", """
            Question: {question}
            Data available:
            {summaries}
            Current plan:
            {plan}
            
            Suggest the next step.
            """).format(
                question=query,
                summaries=data_desc,
                plan=plan_str,
                current_step=current_plan[-1] if current_plan else "",
                result=""
            )
        
        result = self.assistants["PLANNER"].run(self.thread_id, prompt)
        
        state = self.storage.get_current_state()
        step_id = f"{len(state['completed_steps']):03d}_planner"
        self.storage.save_step(
            step_id=step_id,
            step_type="planner",
            prompt=prompt,
            result=result,
            metadata={"plan_length": len(current_plan) if current_plan else 0}
        )
        
        state["completed_steps"].append(step_id)
        self.storage.save_state(state)
        
        return result.strip()
    
    def generate_and_execute_code(self, plan: List[str], data_desc: str, file_ids: List[str]) -> str:
        """Generate code for the plan and execute it."""
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        
        # Use coder_init template
        prompt = PROMPT_TEMPLATES.get("coder_init", """
        Data files:
        {summaries}
        
        Plan:
        {plan}
        
        Implement the plan with the uploaded data files.
        Write and execute Python code. Print the results clearly.
        """).format(
            summaries=data_desc,
            plan=plan_str
        )
        
        result = self.assistants["CODER"].run(
            self.thread_id,
            prompt,
            file_ids=file_ids
        )
        
        state = self.storage.get_current_state()
        step_id = f"{len(state['completed_steps']):03d}_coder"
        self.storage.save_step(
            step_id=step_id,
            step_type="coder",
            prompt=prompt,
            result=result,
            metadata={"plan_length": len(plan)}
        )
        
        state["completed_steps"].append(step_id)
        self.storage.save_state(state)
        
        return result
    
    def verify_solution(self, query: str, current_result: str) -> bool:
        """Check if current solution answers the question."""
        # Use a simplified verifier prompt
        prompt = f"""Question: {query}

Current results:
{current_result}

Does this fully answer the question? Reply ONLY with 'Yes' or 'No'."""
        
        result = self.assistants["VERIFIER"].run(self.thread_id, prompt)
        
        state = self.storage.get_current_state()
        step_id = f"{len(state['completed_steps']):03d}_verifier"
        self.storage.save_step(
            step_id=step_id,
            step_type="verifier",
            prompt=prompt,
            result=result,
            metadata={}
        )
        
        state["completed_steps"].append(step_id)
        self.storage.save_state(state)
        
        return "yes" in result.lower()
    
    def finalize_output(self, query: str, result: str) -> str:
        """Format final output as JSON."""
        prompt = f"""Question: {query}

Analysis results:
{result}

Format the final answer as valid JSON with the key 'final_answer'. 
Extract the actual result values from the analysis above.
Return ONLY the JSON object, no code, no markdown blocks, just the raw JSON.
Example: {{"final_answer": "the actual answer here"}}"""
        
        
        final_result = self.assistants["FINALYZER"].run(self.thread_id, prompt)
        
        state = self.storage.get_current_state()
        step_id = f"{len(state['completed_steps']):03d}_finalyzer"
        self.storage.save_step(
            step_id=step_id,
            step_type="finalyzer",
            prompt=prompt,
            result=final_result,
            metadata={}
        )
        
        state["completed_steps"].append(step_id)
        self.storage.save_state(state)
        
        return final_result
    
    def run_pipeline(self, query: str, data_files: List[str]) -> Dict[str, Any]:
        """Main pipeline using Assistants API."""
        self.logger.info(f"Starting pipeline: {self.config.run_id}")
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Data files: {data_files}")
        
        state = self.storage.get_current_state()
        
        # PHASE 1: Upload and analyze data files
        if not self.file_ids:
            self.logger.info("=== PHASE 1: UPLOADING & ANALYZING DATA ===")
            self.file_ids = self.upload_files(data_files)
            state["file_ids"] = self.file_ids
            self.storage.save_state(state)
            
            data_desc = self.analyze_data(self.file_ids)
        else:
            self.logger.info("Using previously uploaded files")
            # Get data description from previous run
            data_desc = "Previously analyzed data files"
        
        # PHASE 2: Iterative planning and execution
        self.logger.info("=== PHASE 2: ITERATIVE PLANNING & EXECUTION ===")
        plan = []
        current_result = ""
        
        for round_idx in range(self.config.max_refinement_rounds):
            self.logger.info(f"--- Round {round_idx + 1} ---")
            
            # Plan next step
            next_step = self.plan_analysis(query, data_desc, plan)
            plan.append(next_step)
            
            # Execute plan
            current_result = self.generate_and_execute_code(plan, data_desc, self.file_ids)
            
            # Verify if complete
            if self.verify_solution(query, current_result):
                self.logger.info("Solution verified as complete!")
                break
            
            self.logger.info("Solution incomplete, planning next step...")
        
        # PHASE 3: Finalization
        self.logger.info("=== PHASE 3: FINALIZING ===")
        final_result = self.finalize_output(query, current_result)
        
        # Save final output
        output_file = self.storage.run_dir / "final_output" / "result.json"
        output_file.write_text(final_result)
        
        self.logger.info("Pipeline completed successfully!")
        
        return {
            "run_id": self.config.run_id,
            "final_result": final_result,
            "output_file": str(output_file),
            "thread_id": self.thread_id
        }
    
    def cleanup(self):
        """Clean up uploaded files and assistants."""
        self.logger.info("Cleaning up resources...")
        
        # Delete uploaded files
        for file_id in self.file_ids:
            try:
                self.client.files.delete(file_id)
                self.logger.info(f"Deleted file: {file_id}")
            except Exception as e:
                self.logger.error(f"Failed to delete file {file_id}: {e}")
        
        # Delete assistants
        for name, assistant in self.assistants.items():
            try:
                self.client.beta.assistants.delete(assistant.assistant.id)
                self.logger.info(f"Deleted assistant: {name}")
            except Exception as e:
                self.logger.error(f"Failed to delete assistant {name}: {e}")

# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI interface for Assistants API version."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DS-STAR with OpenAI Assistants API")
    parser.add_argument("--resume", type=str, help="Resume from run ID")
    parser.add_argument("--data-files", nargs="+", help="Data files to analyze")
    parser.add_argument("--query", type=str, help="Analysis query")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max refinement rounds")
    parser.add_argument("--cleanup", action="store_true", help="Clean up resources after run")
    parser.add_argument("--config", type=str, help="Path to config file", default="config.yaml")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r') as f:
            config_defaults = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config_defaults = {}
    
    config_params = {
        'run_id': args.resume,
        'max_refinement_rounds': args.max_rounds or config_defaults.get('max_refinement_rounds', 5),
        'model_name': config_defaults.get('model_name', 'gpt-4-turbo-preview')
    }
    
    config_params = {k: v for k, v in config_params.items() if v is not None}
    config = DSConfig(**config_params)
    
    # Get query and data files
    query = args.query or config_defaults.get('query')
    data_files = args.data_files or config_defaults.get('data_files')
    
    if not (data_files and query):
        parser.error("--data-files and --query are required.")
    
    # Expand wildcards
    import glob
    from pathlib import Path
    expanded_files = []
    for pattern in data_files:
        if '*' in pattern or '?' in pattern:
            data_dir = Path(config.data_dir)
            matches = list(data_dir.glob(pattern))
            if matches:
                expanded_files.extend([f.name for f in matches if f.is_file()])
        else:
            expanded_files.append(pattern)
    
    if not expanded_files:
        parser.error("No data files found.")
    
    data_files = expanded_files
    print(f"Processing files: {', '.join(data_files)}")
    
    # Run pipeline
    agent = DS_STAR_Agent(config)
    
    try:
        result = agent.run_pipeline(query, data_files)
        print(f"\n{'='*60}")
        print(f"RUN COMPLETED: {result['run_id']}")
        print(f"THREAD ID: {result['thread_id']}")
        print(f"OUTPUT: {result['output_file']}")
        print(f"FINAL RESULT:\n{result['final_result']}")
        print(f"{'='*60}")
    finally:
        if args.cleanup:
            agent.cleanup()

if __name__ == "__main__":
    main()
