import logging as log
import random
import time
from enum import Enum
from typing import Dict

from llm.config import DEBUG, MAX_TOKENS


class LLMType(Enum):
    MOCK = "mock"
    GPT_3O_MINI = "gpt-3o-mini"
    GPT_35_TURBO = "gpt-35-turbo"
    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_CHAT = "gpt-5-chat"
    LLAMA3_2 = "llama3.2"
    DOLPHIN_MISTRAL = "dolphin-mistral"
    DEEPSEEK_V2 = "deepseek-v2"
    GPT_OSS = "gpt-oss:20b"
    HF = "hf"
    MISTRAL="mistral"
    GEMMA="gemma"
    MISTRAL_7B_INSTRUCT_V02_GPTQ = "Mistral-7B-Instruct-v0.2-GPTQ"
    DEEPSEEK_R1_QCBAR = "DeepSeek-R1-qcbar"
    DEEPSEEK_V3_0324 = "DeepSeek-V3-0324"
    DOLPHIN_21_UNCENSORED = "mainzone/dolphin-2.1-mistral-7b-uncensored"
    DOLPHIN3 = "dolphin3"


ALL_MODELS = [llm.value for llm in LLMType]
GPT5_MODELS = {LLMType.GPT_5, LLMType.GPT_5_MINI,  LLMType.GPT_5_NANO, LLMType.GPT_5_CHAT}
LOCAL_MODELS = {LLMType.GEMMA, LLMType.MISTRAL, LLMType.LLAMA3_2, LLMType.DOLPHIN_MISTRAL, LLMType.DEEPSEEK_V2, LLMType.DOLPHIN3}
OPENAI_MODELS = GPT5_MODELS | {LLMType.GPT_35_TURBO, LLMType.GPT_4, LLMType.GPT_4O, LLMType.GPT_4_1, LLMType.GPT_4O_MINI, LLMType.GPT_3O_MINI}
DEEPSEEK_MODELS = {LLMType.DEEPSEEK_V3_0324, LLMType.DEEPSEEK_R1_QCBAR}


class ModelStatistics:
    statistics = {model: {"tokens": 0, "time": 0, "calls": 0} for model in LLMType}

    @classmethod
    def record_usage(cls, model_type, tokens_used, time_taken):
        cls.statistics[model_type]["tokens"] += tokens_used
        cls.statistics[model_type]["time"] += time_taken
        cls.statistics[model_type]["calls"] += 1

    @classmethod
    def get_statistics(cls, model_type):
        return cls.statistics.get(model_type, {"tokens": 0, "time": 0, "calls": 0})

    @classmethod
    def complete_statistics(cls) -> Dict:
        result = {}
        for llm_type, base_stats in cls.statistics.items():
            result[llm_type.value] = base_stats
            result[llm_type.value]["approximate costs"] = (
                base_stats["tokens"] * 0.0015 / 1000
            )
            result[llm_type.value]["average call time"] = (
                base_stats["time"] / base_stats["calls"]
                if base_stats["calls"] > 0
                else 0
            )
        return result

    @classmethod
    def total_values(cls) -> Dict:
        result = {}
        for key in ["tokens", "time", "calls", "approximate costs"]:
            result[key] = sum(
                stats[key] for stats in cls.complete_statistics().values()
            )
        result["average call time"] = (
            result["time"] / result["calls"] if result["calls"] > 0 else 0
        )
        return result


def pass_llm(
    msg,
    max_tokens=MAX_TOKENS,
    temperature=0,
    llm_type=LLMType.GPT_4O,
    context=None,
    system_message="You are an intelligent system",
):
    prompt = msg
    start_time = time.time()

    if temperature is None:
        temperature = random.random()

    try:
        if llm_type == LLMType.MOCK:
            response_text, total_tokens, elapsed_time = call_mock(
            prompt, "", max_tokens, temperature, system_message
        )
        elif llm_type == LLMType.HF:
            from llm.call_hf import call_hf_llm
    
            response_text, total_tokens, elapsed_time = call_hf_llm(
                prompt, max_tokens, temperature, system_message, context
            )
        elif llm_type in LOCAL_MODELS:
            from llm.call_ollama import call_ollama
    
            response_text, total_tokens, elapsed_time = call_ollama(
                prompt, max_tokens, temperature, llm_type.value, system_message, context
            )
        elif llm_type in OPENAI_MODELS:
            from llm.llm_openai import call_openai
    
            response_text, total_tokens, elapsed_time = call_openai(
                prompt,
                max_tokens,
                temperature,
                system_message,
                context,
                model=llm_type.value,
            )
        elif llm_type in DEEPSEEK_MODELS:
            from llm.call_deepseek import call_deepseek
    
            response_text, total_tokens, elapsed_time = call_deepseek(
                llm_type.value, prompt, max_tokens, temperature, system_message, context
            )
        else:
            raise ValueError(
                f"LLM {llm_type} is not supported. List of supported LLMs: "
                + ", ".join([model.name for model in LLMType])
            )
    except ValueError as e:
        raise e
    except Exception as e:
        print("Error in pass_llm\n", e)
        response_text, total_tokens, elapsed_time = "", 0, 0
    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Record the usage statistics
    ModelStatistics.record_usage(llm_type, total_tokens, elapsed_time)

    # Clean up the response text
    if response_text is None:
        response_text = ""
    response_text = response_text.replace('"', '')

    if DEBUG:
        log.info(f"QUESTION: {prompt}")
        log.info(f"ANSWER: {response_text}")

    log.info(
        f"[Overview] LLM {llm_type} calls: {ModelStatistics.get_statistics(llm_type)['calls']}"
    )
    log.info(
        f"[Overview] LLM {llm_type} token usage: {ModelStatistics.get_statistics(llm_type)['tokens']}"
    )

    return response_text


def call_mock(prompt, role, max_tokens, temperature, system_message):
    output = f"I am just a mock, a random number is {random.randint(239, 239239)}"
    return output, len(output + prompt), 0


if __name__ == "__main__":
    # Define your input
    message = "What is the capital of France?"

    # Call the function with the desired LLM type
    response = pass_llm(
        msg=message,
        llm_type=LLMType.GPT_4 # You can switch this to any other supported enum
    )

    # Print results
    print("Response:", response)
