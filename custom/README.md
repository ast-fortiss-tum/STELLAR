# Tutorial - Customization of STELLAR

This tutorial explains how to apply STELLAR to your custom testing problem. The test subject in the example is a mocked natural language understanding (NLU) system. It contains a simplified climate/navigation intent-classifier. 

<div style="background-color:#fff8d6; border:1px solid #e6c200; border-radius:6px; padding:12px;">
Requirements
    
 - **Operating System:** The provided scripts and notebooks have been tested on Linux, Win, and Mac.  
 - **Hardware:** No high-end GPU is required. The example runs on CPU.
 - **Software Environment:** Python 3.11.8 environment with Jupyter Notebook installed.

 </div>

## Getting Started

To understand how STELLAR works, inspect the following notebooks. First, set up the Jupyter environment using the steps below.

[`01_overview.ipynb`](01_overview.ipynb): Explanation of components of the framework.

[`02_customize.ipynb`](02_customize.ipynb): Customization explanations.

## Run the notebooks

Create and activate the Python 3.11 environment.

```bash
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate.ps1
```

Install the requirements.

```bash
pip install -r requirements.txt
pip install jupyter
```

Install the python kernel:

```bash
python -m pip install ipykernel
```

Register the virtual environment as a kernel:

```bash
python3 -m ipykernel install --user --name=venv
```

Start Jupyter Notebook:

```bash
jupyter notebook
```
Select *Kernel > Change Kernel > venv*.

Optionally, configure the LLM endpoint and ```API_KEY``` in ```.env``` for cloud LLM usage.

## How to apply the generator to your problem

1. Copy the `custom/` folder and rename it (e.g. `my_usecase/`).
2. Replace the `Custom*` classes with your implementations (see below).
3. Adjust the feature space in [`configs/features.json`](configs/features.json).
4. Modify the hyperparameters / presets in [`presets.py`](presets.py).
5. Run and inspect the results under `./results` or `custom/results`

## What to customize

| Component | Code | What it does | Replace with |
|---|---|---|---|
| **CustomContentInput / CustomOutputModel** | [`custom_models.py`](custom_models.py) | Provides the structure/schema of test inputs and outputs.| Adopt to the information relevant in your case |
| **CustomSUT** | [`custom_sut.py`](custom_sut.py) | The system under test which receives test inputs and produces test outputs. Here it receives an utterance and classifies the intent. | Send requests to your system (HTTP request, local model, LLM, …). |
| **CustomUtteranceGenerator** | [`custom_generator.py`](custom_generator.py) | Generates test inputs passed to the system, considering different expressions and perturbations. | Modify utterance generation prompts to meet your language/use case.|
| **CustomFitness** | [`custom_fitness.py`](custom_fitness.py) | Evaluates a response given a test input. Here: uses prediction certainty for expected intent. | Modify this function to include e.g. Intent embedding distance between expected and predicted intent.|
| **Feature space** | [`configs/features.json`](configs/features.json) | Specifies the dimensions the search explores. | Your content, ordinal style, and categorical perturbation features  that are of interest for testing. |

### Presets

Several presets are defined for execution:

| Preset | algorithm | population_size | n_generations | max_time | Use when |
|---|---|---|---|---|---|
| `test` | nsga2 | 4 | 2 | 00:00:30 | Checking the pipeline works. |
| `default` | nsga2 | 10 | 10 | 00:10:00 | A first real experiment. |

Print them at any time:

```bash
python -m custom.main --list
```

Select presets:

```bash
# presets
python -m custom.main --preset test
python -m custom.main --preset default
```

You can always override preset configuration by passing flags.
```bash
python -m custom.main 
        --preset default 
        --algorithm random 
        --seed 7 
        --n_generations 5
```

# local or hosted W&B tracking

```bash
python -m custom.main 
        --preset test 
        --wandb_mode offline 
        --wandb_project stellar-custom

python -m custom.main 
        --preset default 
        --wandb_mode online 
        --wandb_entity my-organization 
        --wandb_project stellar-custom
```

For hosted W&B tracking, authenticate with `wandb login` or configure
`WANDB_API_KEY`. `wandb_entity` selects the organization or user account, while
`wandb_project` selects the project within that owner. To use LLM generation,
select a non-`mock` `generator_llm` and configure the selected provider, for
example `OPENAI_API_KEY`.

## Output

Results are written to `results/CUSTOM_.../`:

- `summary_results.csv` — includes the fail rate.
- `all_critical_utterances.json` — the inputs that triggered a failure.
- `overall_metrics.json` — a short summary printed at the end (`fail_rate`,
  `num_failures`, `num_distinct_failing_intents`).

