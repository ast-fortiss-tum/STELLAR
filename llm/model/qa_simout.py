from dataclasses import dataclass, field
from typing import Any, Dict

from opensbt.simulation.simulator import SimulationOutputBase
from llm.model.models import Utterance

@dataclass
class QASimulationOutput(SimulationOutputBase):
    utterance: Utterance
    model: str
    ipa: str = ""
    response: Any = None
    poi_exists: bool = False
    other: Dict = field(default_factory=dict)  # ensures each instance gets its own dict
