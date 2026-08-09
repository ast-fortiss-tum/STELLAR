"""Search hyperparameters and ready-made presets.

A :class:`Hyperparameters` bundle fully describes one run. Two presets are
provided:

* ``test``    — tiny and fast; use it to check the pipeline end-to-end.
* ``default`` — a reasonable starting point for a real experiment.

Every field is documented in :data:`HYPERPARAMETER_DOCS` (and in the README).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields
from typing import Dict, Optional

from llm.llms import LLMType


@dataclass
class Hyperparameters:
    # --- search ---
    algorithm: str = "nsga2"                 # "nsga2" (guided) or "random" (baseline)
    population_size: int = 10                # individuals per generation; for "random" it is the number of samples
    n_generations: int = 10                  # number of search iterations (generations)
    max_time: str = "00:10:00"               # wall-clock budget "HH:MM:SS"; search stops at whichever limit is hit first
    seed: int = 1                            # RNG seed for reproducibility

    # --- generation ---
    generator_llm: str = LLMType.MOCK.value  # LLM used by the generator; MOCK keeps everything offline

    # --- experiment tracking ---
    wandb_mode: str = "disabled"             # "disabled", "offline", or "online"
    wandb_project: Optional[str] = None       # optional W&B project name
    wandb_entity: Optional[str] = None        # optional W&B organization or user name

    # --- io ---
    features_config: str = "custom/configs/features.json"  # path to the feature-space definition
    results_folder: Optional[str] = None     # where to write results; None -> framework default ("./results")

    def merged(self, **overrides) -> "Hyperparameters":
        """Return a copy with the given (non-None) fields overridden."""
        data = asdict(self)
        for key, value in overrides.items():
            if value is not None and key in data:
                data[key] = value
        return Hyperparameters(**data)


HYPERPARAMETER_DOCS: Dict[str, str] = {
    "algorithm": "Search algorithm: 'nsga2' evolves inputs toward failures; 'random' samples blindly (baseline).",
    "population_size": "Number of test inputs kept per generation. For 'random' it is the total number of samples.",
    "n_generations": "How many generations (iterations) the search performs.",
    "max_time": "Wall-clock limit as 'HH:MM:SS'. The search stops at the first of n_generations / max_time.",
    "seed": "Random seed; fix it for reproducible runs.",
    "generator_llm": "Model name used by the utterance generator. 'mock' runs offline; swap for e.g. 'gpt-4o-mini'.",
    "wandb_mode": "Weights & Biases mode: 'disabled', 'offline', or 'online'.",
    "wandb_project": "Optional Weights & Biases project name for online or offline experiment tracking.",
    "wandb_entity": "Optional Weights & Biases organization or user name that owns the project.",
    "features_config": "JSON file describing the feature space (categorical + ordinal dimensions).",
    "results_folder": "Output directory. Leave as None to use the framework default under './results'.",
}


PRESETS: Dict[str, Hyperparameters] = {
    # Fast smoke test — finishes in seconds.
    "test": Hyperparameters(
        algorithm="nsga2",
        population_size=4,
        n_generations=2,
        max_time="00:00:30",
        seed=1,
    ),
    # Sensible defaults for a first real experiment.
    "default": Hyperparameters(
        algorithm="nsga2",
        population_size=10,
        n_generations=10,
        max_time="00:10:00",
        seed=1,
    ),
}


def get_preset(name: str = "default", **overrides) -> Hyperparameters:
    """Return a preset by name, with optional field overrides."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return PRESETS[name].merged(**overrides)


def describe_hyperparameters() -> str:
    """Human-readable table of the hyperparameters and preset values."""
    lines = ["Hyperparameters:"]
    for f in fields(Hyperparameters):
        lines.append(f"  - {f.name}: {HYPERPARAMETER_DOCS.get(f.name, '')}")
    lines.append("")
    for preset_name, hp in PRESETS.items():
        lines.append(f"Preset '{preset_name}': {asdict(hp)}")
    return "\n".join(lines)
