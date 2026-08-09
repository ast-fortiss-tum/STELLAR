"""Data models."""
from typing import Literal
from pydantic import BaseModel, Field
from llm.model.models import ContentInput

Intent = Literal["INTENT_Climate", 
                 "INTENT_Navigation"]

class CustomContentInput(ContentInput):
  """Holds content related information for an utterance.
     Add fields here when your use case requires more information to be passed by the evaluator.
  """
  intent: Intent

class CustomStyleDescription(BaseModel):
  """Style and perturbation features."""
  politeness: Literal["rude", "neutral", "polite"] = "neutral"
  slang: Literal["formal", "neutral", "slangy"] = "neutral"
  implicitness: Literal["explicit", "implicit"] = "explicit"
  verbosity: Literal["short", "medium", "long"] = "short"
  word_perturbation: str = "none"
  char_perturbation: str = "none"

class CustomOutputModel(BaseModel):
  """Holds the output of the custom SUT. This is the model that the SUT returns for each utterance."""
  intent: Intent = Field(..., description="Intent with the highest probability (Predicted Intent).")
  probabilities: dict[Intent, float] = Field(
      ..., description="Normalized probability for each supported intent."
  )
  score: float = Field(..., description="Probability of the predicted intent.")
