import os

DEBUG = False

LLM_TYPE = os.getenv("LLM_TYPE", "DeepSeek-V3-0324")  # gemini-3-flash-preview # DeepSeek-V3-0324 # claude-4-sonnet
MAX_TOKENS = 1024

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "DeepSeek-V3-0324")
N_VALIDATORS = int(os.getenv("N_VALIDATORS", 1))

LLM_OLLAMA = os.getenv("LLM_OLLAMA", "gpt-5-chat")
LLM_SAMPLING = os.getenv("LLM_SAMPLING", "gpt-5-chat")
LLM_MUTATOR = os.getenv("LLM_MUTATOR", "gpt-5-chat")
LLM_CROSSOVER = os.getenv("LLM_CROSSOVER", "gpt-5-chat")
LLM_VALIDATOR = os.getenv("LLM_VALIDATOR", "gpt-5-mini")
LLM_IPA = os.getenv("LLM_IPA", "gpt-5-chat")
LLM_CLASSIFIER = os.getenv("LLM_CLASSIFIER", "DeepSeek-V3-0324")
LLM_GENERATOR = os.getenv("LLM_GENERATOR", "DeepSeek-V3-0324")

CONTEXT = {
    "location": {
        "position": os.getenv("LOCATION_POSITION", "Marienplatz, Munich, Germany"),
        "time": os.getenv("LOCATION_TIME", "2025-03-19T09:00:00Z"),
    }
}