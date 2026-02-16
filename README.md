# JustEval

<p align="center">
  <img src="assets/simple_logo.png" alt="Description" width="300"/>
</p>

Simple tool for evaluating (for now) generation capabilities of LLMs in two distinct steps: generation and evaluation.

## Generation

The generation relies on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) backend. The setup is fully based on YAML configuration files.

## Installation

You just need to install lm-eval with the models backends you plan to use among: hf, vllm and api. The following will install all, you can pick and choose the ones you need.

```bash
pip install "lm_eval[hf,vllm,api]"
```

Every other library (e.g. transformers, vllm, pytorch, etc.) will be installed as dependencies automatically.

## Run

Once you defined your dataset and run configuration files (as described in the next section), you can run the evaluation with the following command:

```bash
lm-eval run --config <run_config_file>
```

## Configuration files

NOTE: the following documentation is based on lm-evaluation-harness library. For more info on listed or additional fields check their [docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs). In particular:
- For dataset configuration file check [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) and [here (advanced)](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
- For run configuration file check [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/config_files.md)

### Dataset configuration file

The dataset configuration file defines how to load, format, and evaluate your dataset using the lm-evaluation-harness backend. Here's a breakdown of all available fields:

#### Basic Task Configuration

- **task**: The task type identifier. (e.g. `open_generation`). There are no predefined values, this is used by the run configuration files to find the task types requested for evaluation

- **dataset_path**: A Hugging Face dataset identifier (e.g., `giannor/dala`)

- **dataset_name**: The dataset name. Set to `null` if not needed.

- **output_type**: Selects the type of model output for the given task. Defaults to "generate_until". Options are generate_until, loglikelihood, loglikelihood_rolling, and multiple_choice.
  - `generate_until`: Generate tokens until stopping criteria
  - `loglikelihood`: Compute log-likelihood of given completions
  - `multiple_choice`: Select from multiple options

#### Data Splits

- **training_split**: Name of the training split
- **validation_split**: Name of the validation split
- **test_split**: Name of the test split

#### Prompt Engineering

- **doc_to_text**: Template for converting dataset examples to input prompts. Supports:
  - Multi-line prompts using YAML block scalar notation (`|`)
  - Variable interpolation using double curly braces: `{{column_name}}`
  - The variables must match column names in your dataset

- **doc_to_target**: Defines the expected output/target:
  - Can reference a dataset column: `{{label}}`
  - Can be a dummy value if not used for evaluation: `0`

#### Generation Settings

- **generation_kwargs**: Parameters controlling text generation, e.g.:
  - **max_gen_toks**: Maximum number of tokens to generate
  - **temperature**: Sampling temperature
  - **do_sample**: Whether to use sampling
  - **top_p**: Nucleus sampling parameter
  - **top_k**: Top-k sampling parameter

#### Evaluation Metrics

To expand if needed, right now only bypass is used as we generate only.
  
- `bypass`: Skips evaluation entirely

#### Metadata

- **metadata**: Additional information about the configuration:
  - **version**: Version number of the configuration
  - **description**: Human-readable description of the task

#### Example Configuration

Look at `binary_gen_dala.yaml` for an example of a complete dataset configuration file.

### Run configuration file

The run configuration file controls how models are executed and evaluated. It specifies the model to use, which tasks to run, and various execution parameters.

#### Model Configuration

- **model**: The backend to use for model execution::
  - `hf`: Hugging Face Transformers backend
  - `vllm`: vLLM backend
  - `api`: API-based models (e.g., OpenAI, Anthropic, LM Studio)

- **model_args**: Configuration string for the model. Format is comma-separated key=value pairs e.g.:
  - `pretrained`: Model identifier from Hugging Face Hub
  - `dtype`: Data type for model weights (e.g., `float16`, `bfloat16`, `float32`)
  - `tensor_parallel_size`: Number of GPUs for tensor parallelism
  - `gpu_memory_utilization`: Fraction of GPU memory to use (vLLM only)
  - Other backend-specific arguments
  - Example: "pretrained=meta-llama/Meta-Llama-3-8B-Instruct,dtype=float16,tensor_parallel_size=1"

#### Task Configuration

- **tasks**: Comma-separated list or single task type to evaluate. Must match the `task` field in dataset configuration files

- **include_path**: Path to the directory containing dataset configuration YAML files. The tool will search this directory for matching task types

#### Execution Settings

- **batch_size**: Number of samples to process in parallel. Higher values improve throughput but require more memory

- **device**: Target device for execution:
  - `cuda:0`, `cuda:1`, etc. for specific GPUs
  - `cpu` for CPU execution
  - `cuda` for automatic GPU selection

- **limit**: Maximum number of samples to evaluate per task:
  - Set to `null` for no limit (evaluate full dataset)
  - Set to a number for quick testing/debugging

#### Output and Logging

- **output_path**: Directory where results will be saved. Created automatically if it doesn't exist

- **log_samples**: Whether to save individual sample predictions and outputs to a file

- **predict_only**: When `true`, only generates predictions without computing metrics

#### Generation Settings

- **gen_kwargs**: Generation parameters (will override task-specific generation settings) e.g.:
  - **max_gen_toks**: Maximum tokens to generate
  - **temperature**: Sampling temperature
  - **until**: List of stop sequences that terminate generation (e.g., `["\n"]`, `["</s>"]`)
  - **top_p**: Nucleus sampling threshold
  - **top_k**: Top-k sampling parameter
  - **do_sample**: Whether to use sampling

#### Other Settings

- **num_fewshot**: Number of few-shot examples to include in prompts. Set to `0` for zero-shot evaluation

TODO: check and expand fewshot settings

#### Example Configuration

Look at `run_config.yaml` for an example of a complete run configuration file.


