import evaluate
from utils import load_generations

def evaluate_gleu(
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

    gleu = evaluate.load("evaluation/gleu_reimp")  # or "./" dir containing gleu_reimp.py
    results = gleu.compute(sources=sources, predictions=predictions, references=references)

    print("GLEU results:", results)
    for key, value in metadata.items():
        print(f"\t{key}: {value}")
    return results, metadata

def evaluate_hf_metric(
        metric_name: str, 
        model_id: str,
        task: str,
        references_key: str,
        generation_path: str = None,
        predictions_key: str = 'resps'
    ):
    generations, metadata = load_generations(model_id=model_id, task=task, gen_path=generation_path)

    predictions = [gen[predictions_key] for gen in generations]
    references = [gen[references_key] for gen in generations]

    metric = evaluate.load(metric_name)
    results = metric.compute(predictions=predictions, references=references)

    print(f"{metric_name} results:", results)
    for key, value in metadata.items():
        print(f"\t{key}: {value}")
    return results, metadata