DEBUG = False
LLM_TYPE = "gpt-4o"
MAX_TOKENS = 300
# DEPLOYMENT_NAME = "gpt-35-turbo-1106"
DEPLOYMENT_NAME = "gpt-4o"
#DEPLOYMENT_NAME = "DeepSeek-R1-qcbar"
N_VALIDATORS = 1
# "gpt-35-turbo-1106"
# "gpt-4o"
LLM_OLLAMA = "gpt-4o"
LLM_SAMPLING = "gpt-4o"
# LLM_MUTATOR = "gpt-35-turbo-1106"
LLM_MUTATOR = "gpt-4o" #"gpt-35-turbo-1106"
LLM_CROSSOVER = "gpt-4o"#"gpt-35-turbo-1106"
LLM_VALIDATOR = "gpt-4o"#"gpt-35-turbo-1106"
LLM_IPA = DEPLOYMENT_NAME#"gpt-4o"#"gpt-35-turbo-1106"

CONTEXT = {
    "location" : {
        "position" : "Amathountos Avenue 502, 4520, Limassol, Cyprus",
        "time" : "2025-03-19T09:00:00Z",
    }
}
###############

EXAMPLE = "I am hungry and like Pizza."
USER_PROFILE  = "The user is an IT employee and is lactose intolerant."
