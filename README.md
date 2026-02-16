# JustEval

<p align="center">
  <img src="assets/simple_logo.png" alt="Description" width="300"/>
</p>

Simple tool for evaluating (for now) generation capabilities of LLMs in two distinct steps: generation and evaluation.

## Motivation

The motivation behind JustEval is to have a simple, flexible and customizable tool for evaluating LLMs on various tasks and datasets. 

Our main goal is to separate generation from evaluation to allow evaluations to be run without having to generate outputs every time, making it faster to re-evaluate models in cases where the task (e.g. data or prompt) don't change but metrics or other aspects of the evaluation do. We also wanted to have easy customization of evaluation metrics, while relying also on Hugging Face metrics. 

JustEval is built on top of the lm-evaluation-harness library, which provides a lot of flexibility in defining tasks and generation settings through YAML configuration files (however it does not separates generation from evaluation). 

JustEval extends this by adding its own evaluation system that can compute any Hugging Face metric or easily craftable custom evaluation function implemented in `evaluation/just_metrics.py` on the generated outputs.

## Generation

The generation relies on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) backend. The setup is fully based on YAML configuration files. More details on how to define configuration files and run generations are below.

## Evaluation

The goal of JustEval is to separate generation from evaluation and keep evaluation easy, flexible and customizable. JustEval can compute any Hugging Face metric or custom evaluation function implemented in `evaluation/just_metrics.py` on the generated outputs. More details on how to define evaluation metrics and run evaluations are below.

## Installation

You just need to install lm-eval with the models backends you plan to use among: hf, vllm and api. The following will install all, you can pick and choose the ones you need.

```bash
pip install -r requirements.txt
```




<!-- ## Run

Once you defined your dataset and run configuration files (as described in the next section), you can run the evaluation with the following command:

```bash
lm-eval run --config <run_config_file>
``` -->

## Configuration files

NOTE: part of the following documentation is based on lm-evaluation-harness library. For more info on listed or additional fields check their [docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs). In particular:
- For task configuration file check [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) and [here (advanced)](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
- For run configuration file check [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/config_files.md)

Example configuration files can be found in `generation/task_configs` and `generation/run_configs` directories.

### Task configuration file

The task configuration file defines how to load and format your dataset for generation using the lm-evaluation-harness backend. Here you can also set custom generation parameters. They should be placed in `generation/task_configs` directory. Here's a breakdown of all available fields:

#### Basic Task Configuration

- `task`: The task type identifier. There are no predefined values, this is used by the run configuration files to find the task types requested for evaluation

- `dataset_path`: A Hugging Face dataset identifier (e.g., `giannor/dala`)

- `dataset_name`: The dataset name. Set to `null` if not needed.

- `output_type`: Selects the type of model output for the given task. Defaults to "generate_until". Options are generate_until, loglikelihood, loglikelihood_rolling, and multiple_choice.
  - `generate_until`: Generate tokens until stopping criteria
  - `loglikelihood`: Compute log-likelihood of given completions
  - `multiple_choice`: Select from multiple options

  **NOTE: The current evaluation pipeline in JustEval is focused on generation only, hence we expect outputs from the `generate_until` output type.**

#### Data Splits

- `training_split`: Name of the training split
- `validation_split`: Name of the validation split
- `test_split`: Name of the test split

  **NOTE: lm-evaluation-harness expects test split to be defined.**


#### Prompt Engineering

- `doc_to_text`: Template for converting dataset examples to input prompts. Supports:
  - Multi-line prompts using YAML block scalar notation (`|`)
  - Variable interpolation using double curly braces: `{{column_name}}`
  - The variables must match column names in your dataset

- `doc_to_target`: Defines the expected output/target:
  - Can reference a dataset column: `{{label}}`
  - Can be a dummy value if not used for evaluation: `0`


#### Generation Settings

- `generation_kwargs`: Parameters controlling text generation, e.g.:
  - `max_gen_toks`: Maximum number of tokens to generate
  - `temperature`: Sampling temperature
  - `do_sample`: Whether to use sampling
  - `top_p`: Nucleus sampling parameter
  - `top_k`: Top-k sampling parameter


#### Evaluation Metrics

In order to separate generation from evaluation JustEval uses another metric system than lm-evaluation-harness, hence this part of the configuration file can be ignored unless you want to compute metrics with lm-evaluation-harness too. In our configs we skip lm-evaluation-harness evaluations by using the following:

``` yaml
metric_list:
  - metric: bypass
```

Our metric system is described in the **Metadata** section of the task configuration file, more on that in the respective section below.

#### Metadata

- `metadata`: Additional information about the configuration:
  - `version`: Version number of the configuration
  - `description`: Human-readable description of the task
  - `just_metrics`: List of metrics to compute with JustEval after generation.

Inside `just_metrics` you can specify the metrics to compute with JustEval after generation. This is what a JustEval metric looks like in the configuration file:

``` yaml
just_metrics:
  - metric: metric_name # HF metric or custom function
    custom: false
    references_key: dataset_column1
    predictions_key: generated_output_key # Default is resps according to lm-evaluation-harness generations
    param1: value1
    param2: value2
```

- If `custom` is `true`, the metric name is expected to be a function with the same name in `evaluation/just_metrics.py` otherwise (`false` or not set) it is expected to be a Hugging Face metric and the name should match the one in the Hugging Face Hub (e.g. `accuracy`, `rouge`, etc.)
- `references_key` and `predictions_key` specify where to find the references and predictions for the metric. By default, predictions are taken from the generation outputs (`resps`) and references from the dataset column specified in `references_key`, but you can customize this as needed.
- You can (or must) also add any additional parameters required by the metric function


#### Example Configuration

Look at `generation/task_configs/gec_dala.yaml` for an example of a complete task configuration file.

---

### Run configuration file

The run configuration file controls how models are executed. It specifies the model to use, which tasks to run, and various execution parameters. They should be places into `generation/run_configs` directory. Here's a breakdown of all available fields:

#### Model Configuration

- `model`: The backend to use for model execution::
  - `hf`: Hugging Face Transformers backend
  - `vllm`: vLLM backend
  - `api`: API-based models (e.g., OpenAI, Anthropic, LM Studio)

- `model_args`: Configuration string for the model. Format is comma-separated key=value pairs e.g.:
  - `pretrained`: Model identifier from Hugging Face
  - `dtype`: Data type for model weights (e.g., `float16`, `bfloat16`, `float32`)
  - `tensor_parallel_size`: Number of GPUs for tensor parallelism
  - `gpu_memory_utilization`: Fraction of GPU memory to use (vLLM only)
  - Other backend-specific arguments
  - Example:

``` yaml
model: vllm
model_args:
  pretrained: meta-llama/Meta-Llama-3-8B-Instruct
  dtype: float16
  tensor_parallel_size: 1
```

#### Task Configuration

- `tasks`: List or single task type to evaluate. Must match the `task` field in task configuration files

- `include_path`: Path to the directory containing task configuration YAML files. The tool will search this directory for matching task types
- Example:

``` yaml
include_path: ../task_configs
tasks:
  - gec_dala
  - cls-gen_dala
  - culture_daisy
```

#### Execution Settings

- `batch_size`: Number of samples to process in parallel. Higher values improve throughput but require more memory

- `device`: Target device for execution:
  - `cuda:0`, `cuda:1`, etc. for specific GPUs
  - `cpu` for CPU execution
  - `cuda` for automatic GPU selection

- `limit`: Maximum number of samples to evaluate per task:
  - Set to `null` for no limit (evaluate full dataset)
  - Set to a number for quick testing/debugging

#### Output and Logging

- `output_path`: Directory where results will be saved. Created automatically if it doesn't exist

- `log_samples`: Whether to save individual sample predictions and outputs to a file

- `predict_only`: When `true`, only generates predictions without computing metrics

#### Generation Settings

- `gen_kwargs`: Generation parameters (will override task-specific generation settings) e.g.:
  - `max_gen_toks`: Maximum tokens to generate
  - `temperature`: Sampling temperature
  - `until`: List of stop sequences that terminate generation (e.g., `["\n"]`, `["</s>"]`)
  - `top_p`: Nucleus sampling threshold
  - `top_k`: Top-k sampling parameter
  - `do_sample`: Whether to use sampling

#### Other Settings

- `num_fewshot`: Number of few-shot examples to include in prompts. Set to `0` for zero-shot evaluation

TODO: check and expand fewshot settings

#### Example Configuration

Look at `generation/run_configs/meta-llama__meta-llama-3-8b-instruct.yaml` for an example of a complete run configuration file.

---

### JustConfig configuration files

The purpose of these files is to easily list which of the defined models and tasks to run for generation and/or evaluation in JustEval. The fields are:

- `force`: whether to generate the outputs for the listed models and tasks even if they already exist. Set to `false` to skip generation for model-task pairs that already have generated outputs saved.

- `models`: list of model names to run. These should match the model names defined in the run configuration file names (e.g. `meta-llama__meta-llama-3-8b-instruct`)

- `tasks`: list of task names to run. These should match the task names defined in the task configuration file names (e.g. `gec_dala`)

An example of a JustConfig configuration file can be found in `just_config.yaml`. You can create as many of these files as you want to easily run different combinations of models and tasks without having to edit the main run configuration files.

---

## Running generations and evaluations

### Run generations

Once you correctly defined model and task configuration files under `generation/run_configs` and `generation/task_configs` directories and you have set up the JustConfig configuration file to specify which models and tasks to run, you can run generations with the following command:

```bash
python just_generate.py --config <just_config_file>
```

This will generate outputs (using lm-evaluation-harness) for all combinations of models and tasks listed in the JustConfig configuration file specified by `--config` flag. The generated outputs will be saved in the folder specified by `output_path` in each model's run configuration file (according to lm-evaluation-harness standards).

### Run evaluations

To run evaluations on the generated outputs, you can use the same JustConfig configuration file used for generations or if you want a subset of the generations you can create a new JustConfig configuration file listing only the models and tasks you want to evaluate. Once you have the JustConfig configuration file ready, you can run evaluations with the following command:

```bash
python just_eval.py --config <just_config_file>
```

This will compute the evaluation metrics specified in each task configuration file for all combinations of models and tasks listed in the JustConfig configuration file specified by `--config` flag. The results will be saved in the `metric_results` folder. Example saved output

```json
{
  {
    "model_id": "google/gemma-3-12b-it",
    "task": "gec_dala",
    "dataset": "giannor/dala_gen_v2",
    "timestamp": "2026-02-16T16-55-43.807652",
    "metrics": {
      "gleu": 0.8017291327935923,
      "exact_match": 0.25
    }
  }
}
```

NOTE: the `timestamp` field matches the timestamp used by lm-evaluation-harness when saving the generated outputs (on outputs and metadata file names), this can be used to link the evaluation results to the corresponding generation outputs.