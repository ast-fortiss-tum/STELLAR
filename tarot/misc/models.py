from typing import List, Dict, Optional

from pydantic import BaseModel
from llm.model.models import ContentInput, ContentOutput, Coordinates

class NLUContentInput(ContentInput):
    intent: str
    # slots: Optional[dict] = None
    

class NLUContentOutput(ContentInput):
    output: Dict[str,Dict]
    
    
    