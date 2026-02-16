#!/usr/bin/env python3
"""
JustGenerate: A tool to validate and run language model generations.

This script:
1. Validates that model configs and task configs exist
2. Updates run configs with specified tasks
3. Runs lm-eval for each model sequentially to generate outputs
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import glob

try:
    from ruamel.yaml import YAML
    HAS_RUAMEL = True
except ImportError:
    HAS_RUAMEL = False
    print("Warning: ruamel.yaml not found. Install with: pip install ruamel.yaml")
    print("Using basic yaml library (will not preserve comments/formatting)")


def load_yaml(filepath: str) -> dict:
    """Load and parse a YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def check_models_exist(models: List[Dict], run_configs_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Check which models have corresponding run config files.
    
    Args:
        models: List of model dictionaries from config
        run_configs_dir: Path to run_configs directory
        
    Returns:
        Tuple of (found_models, missing_models)
    """
    found = []
    missing = []
    
    for model in models:
        model_name = model['name']
        config_file = run_configs_dir / f"{model_name}.yaml"
        
        if config_file.exists():
            found.append(model_name)
        else:
            missing.append(model_name)
    
    return found, missing


def check_tasks_exist(tasks: List[Dict], task_configs_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Check which tasks have corresponding task config files.
    
    Args:
        tasks: List of task dictionaries from config
        task_configs_dir: Path to task_configs directory
        
    Returns:
        Tuple of (found_tasks, missing_tasks)
    """
    found = []
    missing = []
    
    for task in tasks:
        task_name = task['name']
        config_file = task_configs_dir / f"{task_name}.yaml"
        
        if config_file.exists():
            found.append(task_name)
        else:
            missing.append(task_name)
    
    return found, missing


def check_existing_outputs(model_name: str, task_name: str, outputs_dir: Path) -> bool:
    """
    Check if outputs already exist for a model-task combination.
    
    Args:
        model_name: Name of the model
        task_name: Name of the task
        outputs_dir: Path to the outputs directory
        
    Returns:
        True if outputs exist, False otherwise
    """
    # Check if outputs directory exists
    if not outputs_dir.exists():
        return False
    
    # Try exact match first
    model_output_dir = outputs_dir / model_name
    if model_output_dir.exists():
        pattern = str(model_output_dir / f"samples_{task_name}_*.jsonl")
        matches = glob.glob(pattern)
        if len(matches) > 0:
            return True
    
    # Try case-insensitive match by checking all subdirectories
    for subdir in outputs_dir.iterdir():
        if subdir.is_dir() and subdir.name.lower() == model_name.lower():
            pattern = str(subdir / f"samples_{task_name}_*.jsonl")
            matches = glob.glob(pattern)
            if len(matches) > 0:
                return True
    
    return False


def filter_tasks_for_model(model_name: str, tasks: List[str], outputs_dir: Path, force: bool) -> Tuple[List[str], List[str]]:
    """
    Filter tasks for a model based on existing outputs and force flag.
    
    Args:
        model_name: Name of the model
        tasks: List of task names
        outputs_dir: Path to the outputs directory
        force: Whether to force regeneration
        
    Returns:
        Tuple of (tasks_to_run, tasks_to_skip)
    """
    if force:
        return tasks, []
    
    tasks_to_run = []
    tasks_to_skip = []
    
    for task in tasks:
        if check_existing_outputs(model_name, task, outputs_dir):
            tasks_to_skip.append(task)
        else:
            tasks_to_run.append(task)
    
    return tasks_to_run, tasks_to_skip


def update_run_config_tasks(run_config_path: Path, tasks: List[str]) -> None:
    """
    Update the tasks list in a run config file.
    Preserves comments and formatting using ruamel.yaml.
    
    Args:
        run_config_path: Path to the run config file
        tasks: List of task names to set
    """
    if HAS_RUAMEL:
        # Use ruamel.yaml to preserve comments and formatting
        yaml_handler = YAML()
        yaml_handler.preserve_quotes = True
        yaml_handler.default_flow_style = False
        yaml_handler.width = 4096  # Prevent line wrapping
        yaml_handler.explicit_start = False
        yaml_handler.indent(mapping=2, sequence=2, offset=0)
        
        with open(run_config_path, 'r') as f:
            config = yaml_handler.load(f)
        
        # Update only the tasks field
        config['tasks'] = tasks
        
        with open(run_config_path, 'w') as f:
            yaml_handler.dump(config, f)
    else:
        # Fallback to basic yaml (will not preserve comments)
        config = load_yaml(str(run_config_path))
        config['tasks'] = tasks
        with open(str(run_config_path), 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  Updated {run_config_path.name} with tasks: {tasks}")


def run_lm_eval(run_config_path: Path) -> bool:
    """
    Run lm-eval with the specified run config file to generate outputs.
    
    Args:
        run_config_path: Path to the run config file
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running generation for: {run_config_path.name}")
    print(f"{'='*80}\n")
    
    # Change to the generation directory to run lm-eval
    original_dir = os.getcwd()
    try:
        os.chdir(run_config_path.parent)
        
        cmd = ['lm-eval', 'run', '--config', str(run_config_path.name)]
        
        # Run the command and stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output line by line
        for line in process.stdout:
            print(line, end='')
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            print(f"\n✓ Successfully completed generation for {run_config_path.name}")
            return True
        else:
            print(f"\n✗ Generation failed for {run_config_path.name} with return code {return_code}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error running generation: {e}")
        return False
    finally:
        os.chdir(original_dir)


def main():
    """Main execution function."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python just_generate.py <config_file>")
        print("Example: python just_generate.py just_eval.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Validate config file exists
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found")
        sys.exit(1)
    
    # Load the configuration
    print(f"Loading configuration from: {config_file}")
    config = load_yaml(config_file)
    
    # Setup paths
    base_dir = Path(os.path.dirname(os.path.abspath(config_file)))
    generation_dir = base_dir / 'generation'
    run_configs_dir = generation_dir / 'run_configs'
    task_configs_dir = generation_dir / 'task_configs'
    outputs_dir = generation_dir / 'outputs'
    
    # Get models, tasks, and force flag from config
    models = config.get('models', [])
    tasks = config.get('tasks', [])
    force = config.get('force', False)
    
    print(f"\nFound {len(models)} model(s) and {len(tasks)} task(s) in config")
    print(f"Force regeneration: {force}")
    
    # Check which models exist
    print("\n" + "="*80)
    print("CHECKING MODELS")
    print("="*80)
    found_models, missing_models = check_models_exist(models, run_configs_dir)
    
    if found_models:
        print(f"✓ Found models ({len(found_models)}):")
        for model in found_models:
            print(f"  - {model}")
    
    if missing_models:
        print(f"\n✗ Missing models ({len(missing_models)}):")
        for model in missing_models:
            print(f"  - {model}")
    
    # Check which tasks exist
    print("\n" + "="*80)
    print("CHECKING TASKS")
    print("="*80)
    found_tasks, missing_tasks = check_tasks_exist(tasks, task_configs_dir)
    
    if found_tasks:
        print(f"✓ Found tasks ({len(found_tasks)}):")
        for task in found_tasks:
            print(f"  - {task}")
    
    if missing_tasks:
        print(f"\n✗ Missing tasks ({len(missing_tasks)}):")
        for task in missing_tasks:
            print(f"  - {task}")
    
    # Exit if no models found
    if not found_models:
        print("\n" + "="*80)
        print("ERROR: No models to generate outputs")
        print("="*80)
        sys.exit(1)
    
    # Report what will be used
    if missing_models or missing_tasks:
        print("\n" + "="*80)
        print("WARNING: Some models or tasks are missing")
        print("Proceeding with found models and tasks only")
        print("="*80)
    
    # Check for existing outputs and filter tasks
    if found_tasks:
        print("\n" + "="*80)
        print("CHECKING EXISTING OUTPUTS")
        print("="*80)
        
        models_to_run = {}  # model_name -> list of tasks to run
        
        for model_name in found_models:
            tasks_to_run, tasks_to_skip = filter_tasks_for_model(
                model_name, found_tasks, outputs_dir, force
            )
            
            if tasks_to_skip:
                print(f"\n{model_name}:")
                print(f"  ⊙ Skipping existing tasks ({len(tasks_to_skip)}): {tasks_to_skip}")
            
            if tasks_to_run:
                if not tasks_to_skip:
                    print(f"\n{model_name}:")
                print(f"  → Will run tasks ({len(tasks_to_run)}): {tasks_to_run}")
                models_to_run[model_name] = tasks_to_run
            elif not tasks_to_skip:
                print(f"\n{model_name}:")
                print(f"  ✓ No tasks to run")
        
        # Update run configs with tasks that need to be run
        if models_to_run:
            print("\n" + "="*80)
            print("UPDATING RUN CONFIGS WITH TASKS")
            print("="*80)
            for model_name, tasks_to_run in models_to_run.items():
                run_config_path = run_configs_dir / f"{model_name}.yaml"
                update_run_config_tasks(run_config_path, tasks_to_run)
        else:
            print("\n" + "="*80)
            print("NO TASKS TO RUN (all outputs exist and force=false)")
            print("="*80)
            sys.exit(0)
    else:
        print("\n" + "="*80)
        print("SKIPPING TASK UPDATE (no tasks specified)")
        print("="*80)
        models_to_run = {model: [] for model in found_models}
    
    # Run generations for each model that has tasks
    if models_to_run:
        print("\n" + "="*80)
        print("RUNNING GENERATIONS")
        print("="*80)
        
        success_count = 0
        failed_count = 0
        
        for model_name in models_to_run.keys():
            run_config_path = run_configs_dir / f"{model_name}.yaml"
            
            if run_lm_eval(run_config_path):
                success_count += 1
            else:
                failed_count += 1
    else:
        success_count = 0
        failed_count = 0
    
    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    if models_to_run:
        print(f"Total models: {len(models_to_run)}")
    else:
        print(f"Total models: 0")
    print(f"✓ Successful: {success_count}")
    if failed_count > 0:
        print(f"✗ Failed: {failed_count}")
    
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
