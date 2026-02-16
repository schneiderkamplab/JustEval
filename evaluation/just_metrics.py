import json
import os
import evaluate
from evaluation.utils import load_generations

def log_results(results, metadata):
    # rename metadata keys dataset_path to dataset
    metadata['dataset'] = metadata.pop('dataset_path')
    metadata['model_id'] = metadata.pop('pretrained')

    print("Metric results:", results)
    for key, value in metadata.items():
        print(f"\t{key}: {value}")
    
    # save to a json file merging results by model and task
    model_id = metadata['model_id']
    task = metadata['dataset']
    timestamp = metadata['timestamp']
    
    output_dir = "metric_results"
    output_file = "results.json"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    # Load existing results or initialize empty list
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    # Find existing entry for this model-task pair
    existing_entry = None
    for entry in all_results:
        if entry.get('model_id') == model_id and entry.get('dataset') == task:
            existing_entry = entry
            break
    
    if existing_entry:
        # Merge new metrics into existing entry
        existing_entry['metrics'].update(results)
    else:
        # Create new entry with metadata and metrics
        new_entry = {
            'model_id': model_id,
            'task': metadata['task'],
            'dataset': task,
            'timestamp': timestamp,
            'metrics': results,
        }
        all_results.append(new_entry)
    
    # Save back to file
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

def gleu_reimp(
        model_id: str,
        task: str,
        generation_path: str = None,
        sources_key: str = 'corrupted', 
        predictions_key: str = 'resps', 
        references_key: str = 'original'
    ):
    generations, metadata = load_generations(model_id=model_id, task=task, gen_path=generation_path)

    # if a reference is a single string and not a list of strings wrap it in a list to ensure it matches the expected input format for the metric
    for gen in generations:
        if isinstance(gen[references_key], str):
            gen[references_key] = [gen[references_key]]

    sources = [gen[sources_key] for gen in generations]
    predictions = [gen[predictions_key] for gen in generations]
    references = [gen[references_key] for gen in generations]

    gleu = evaluate.load("evaluation/gleu_reimp")
    results = gleu.compute(sources=sources, predictions=predictions, references=references)

    log_results(results, metadata)
    return results, metadata

def evaluate_hf_metric(
        metric_name: str, 
        model_id: str,
        task: str,
        references_key: str,
        predictions_key: str = 'resps',
        generation_path: str = None,
    ):
    generations, metadata = load_generations(model_id=model_id, task=task, gen_path=generation_path)

    predictions = [gen[predictions_key] for gen in generations]
    references = [gen[references_key] for gen in generations]

    metric = evaluate.load(metric_name)
    results = metric.compute(predictions=predictions, references=references)
    
    # convert any numpy types in results to native Python types for better printing
    for key, value in results.items():
        if isinstance(value, (int, float)):
            results[key] = value.item()

    log_results(results, metadata)
    return results, metadata