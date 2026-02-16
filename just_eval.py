import yaml
import argparse
import sys
import glob
from pathlib import Path
from evaluation import just_metrics


def load_config(config_path: str):
    """Load the main configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_task_config(task_name: str):
    """Load a task configuration file."""
    task_config_path = f"generation/task_configs/{task_name}.yaml"
    try:
        with open(task_config_path, 'r', encoding='utf-8') as f:
            task_config = yaml.safe_load(f)
        return task_config
    except FileNotFoundError:
        print(f"Warning: Task config not found for {task_name} at {task_config_path}")
        return None


def check_existing_outputs(model_name: str, task_name: str, outputs_dir: Path) -> bool:
    """
    Check if outputs already exist for a model-task combination.
    Uses case-insensitive matching for model folder names.
    
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
    
    # Convert model name (replace '/' with '__' as done in utils.py)
    model_folder_name = model_name.replace('/', '__')
    model_folder_lower = model_folder_name.lower()
    
    # Try exact match first
    model_output_dir = outputs_dir / model_folder_name
    if model_output_dir.exists():
        pattern = str(model_output_dir / f"samples_{task_name}_*.jsonl")
        matches = glob.glob(pattern)
        if len(matches) > 0:
            return True
    
    # Try case-insensitive match by checking all subdirectories (lowercase comparison)
    # Use the actual folder name (subdir) for file operations, not the lowercase version
    for subdir in outputs_dir.iterdir():
        if subdir.is_dir() and subdir.name.lower() == model_folder_lower:
            pattern = str(subdir / f"samples_{task_name}_*.jsonl")
            matches = glob.glob(pattern)
            if len(matches) > 0:
                return True
    
    return False


def run_evaluation(config_path: str):
    """Run evaluation for all model-task pairs defined in config."""
    config = load_config(config_path)
    
    models = config.get('models', [])
    tasks = config.get('tasks', [])
    
    if not models:
        print("No models found in config file.")
        return
    
    if not tasks:
        print("No tasks found in config file.")
        return
    
    # Setup paths
    outputs_dir = Path("generation/outputs")
    
    # Iterate through all model-task pairs
    for model in models:
        model_name = model.get('name')
        if not model_name:
            print("Warning: Model entry missing 'name' field, skipping.")
            continue
            
        for task in tasks:
            task_name = task.get('name')
            if not task_name:
                print("Warning: Task entry missing 'name' field, skipping.")
                continue
            
            # Check if outputs exist for this model-task pair
            if not check_existing_outputs(model_name, task_name, outputs_dir):
                print(f"Skipped: {model_name} on {task_name} (outputs not found)")
                continue
            
            # Load task configuration
            task_config = load_task_config(task_name)
            if task_config is None:
                continue
            
            # Get just_metrics from metadata from task config
            just_metrics_list = task_config.get('metadata', {}).get('just_metrics', [])
            if not just_metrics_list:
                continue
            
            # Print headline only when we have outputs and metrics to run
            print(f"\n{'='*80}")
            print(f"Evaluating model: {model_name} on task: {task_name}")
            print(f"{'='*80}\n")
            
            # Process each metric
            for metric_def in just_metrics_list:
                metric_name = metric_def.get('metric')
                if not metric_name:
                    print("Warning: Metric definition missing 'metric' field, skipping.")
                    continue
                
                is_custom = metric_def.get('custom', False)
                
                # Prepare parameters (exclude 'metric' and 'custom' keys)
                params = {k: v for k, v in metric_def.items() if k not in ['metric', 'custom']}
                params['model_id'] = model_name
                params['task'] = task_name
                
                try:
                    if is_custom:
                        # Call custom function from just_metrics module
                        print(f"Running custom metric: {metric_name}")
                        metric_func = getattr(just_metrics, metric_name)
                        results, metadata = metric_func(**params)
                    else:
                        # Call evaluate_hf_metric
                        print(f"Running HuggingFace metric: {metric_name}")
                        params['metric_name'] = metric_name
                        results, metadata = just_metrics.evaluate_hf_metric(**params)
                    
                    print(f"✓ Metric {metric_name} completed successfully\n")
                    
                except AttributeError as e:
                    print(f"✗ Error: Metric function '{metric_name}' not found in just_metrics module")
                    print(f"  {str(e)}\n")
                except Exception as e:
                    print(f"✗ Error running metric {metric_name}: {str(e)}\n")
                    import traceback
                    traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Run evaluations based on configuration file')
    parser.add_argument(
        '--config',
        type=str,
        default='just_config.yaml',
        help='Path to configuration file (default: just_config.yaml)'
    )
    
    args = parser.parse_args()
    
    run_evaluation(args.config)


if __name__ == "__main__":
    main()
