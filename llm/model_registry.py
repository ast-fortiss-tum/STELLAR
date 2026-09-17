# model_registry.py
import os

from dotenv import load_dotenv

load_dotenv()

""" Include here the models that should be accessable in the framework"""

DEEPSEEK_AZURE_ENDPOINT = "<pass endpoint>"

MODEL_REGISTRY = {
    "mock": {
        "deployment_name": "mock",
    },
    "gpt-3o-mini": {
        "deployment_name": "gpt-3o-mini",
    },
    "gpt-35-turbo": {
        "deployment_name": "gpt-35-turbo",
    },
    "gpt-4o": {
        "deployment_name": "gpt-4o",
    },
    "gpt-4o-mini": {
        "deployment_name": "gpt-4o-mini",
    },
    "gpt-4.1": {
        "deployment_name": "gpt-4.1",
    },
    "gpt-5": {
        "deployment_name": "gpt-5",
    },
    "gpt-5-mini": {
        "deployment_name": "gpt-5-mini",
    },
    "gpt-5-nano": {
        "deployment_name": "gpt-5-nano",
    },
    "gpt-5-chat": {
        "deployment_name": "gpt-5-chat",
    },
    "llama3.2": {
        "deployment_name": "llama3.2",
    },
    "dolphin-mistral": {
        "deployment_name": "dolphin-mistral",
    },
    "deepseek-v2": {
        "deployment_name": "deepseek-v2",
    },
    "hf": {
        "deployment_name": "hf",
    },
    "Mistral-7B-Instruct-v0.2-GPTQ": {
        "deployment_name": "Mistral-7B-Instruct-v0.2-GPTQ",
    },
    "DeepSeek-R1-qcbar": {
        "deployment_name": "DeepSeek-R1-qcbar",
        "api_version": "2024-05-01-preview",
        "azure_endpoint": "<pass endpoint>",
    },
    "DeepSeek-V3-0324": {
        "deployment_name": "DeepSeek-V3-0324",
        "api_version": "2024-05-01-preview",
        "azure_endpoint": DEEPSEEK_AZURE_ENDPOINT,
    },
}
