from dataclasses import dataclass, field
from typing import Any, Dict

from opensbt.simulation.simulator import SimulationOutputBase
from llm.model.models import Conversation

@dataclass
class MultiTurnSimulationOutput(SimulationOutputBase):
    conversation: Conversation
    model: str
    ipa: str = ""
    other: Dict = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "conversation": self.conversation.to_dict(),
            "model": self.model, 
            "ipa": self.ipa,
            "other": self.other,
            "timestamp": self.timestamp,
        }