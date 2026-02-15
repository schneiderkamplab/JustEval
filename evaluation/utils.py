import json
import os

def load_generations(model_id: str, task: str, gen_path: str = None, single_resp: bool = True):
    results = []
    outputs_folder = "generation/outputs"
    metadata = {}

    if gen_path is None:
        partial_generation_path = f"samples_{task}_"

        outputs_folder = f"{outputs_folder}/{model_id.replace('/', '__')}"

        most_recent_file = None
        most_recent_time = None

        # From the outputs folder, find the file that starts with the partial_generation_path and ends with .jsonl
        gen_path = None
        for file in os.listdir(outputs_folder):
            if file.startswith(partial_generation_path) and file.endswith(".jsonl"):
                gen_path = os.path.join(outputs_folder, file)
                timestamp = file.split(partial_generation_path)[-1].split('.jsonl')[0]
                # If this file is more recent than the most recent file we've seen so far, update the most recent file and time
                if most_recent_time is None or timestamp > most_recent_time:
                    most_recent_file = gen_path
                    most_recent_time = timestamp

        gen_path = most_recent_file
        metadata_path = os.path.join(outputs_folder, f"results_{most_recent_time}.json")

        if not os.path.exists(gen_path):
            raise FileNotFoundError(f"No generation file found in {gen_path}.")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata file found in {metadata_path}.")

        print(f"Loading generations from {gen_path}")
        print(f"Loading metadata from {metadata_path}")

        # Load metadata and extract task, dataset_path, and pretrained
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_json = json.load(f)
            # Get the task config
            if task in metadata_json.get('configs', {}):
                task_config = metadata_json['configs'][task]
                metadata = {
                    'task': task_config.get('task'),
                    'dataset_path': task_config.get('dataset_path'),
                    'pretrained': task_config.get('metadata', {}).get('pretrained')
                }


    with open(gen_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            
            # Extract all fields from doc object and add resps
            if single_resp:
                entry = {**data['doc'], 'resps': data['resps'][0][0]}
            else:
                entry = {**data['doc'], 'resps': data['resps']}
            
            results.append(entry)
    
    return results, metadata