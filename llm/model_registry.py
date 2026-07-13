# model_registry.py
import os

from dotenv import load_dotenv

load_dotenv()

""" Include here the models that should be accessable in the framework"""

DEFAULT_AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
DEEPSEEK_AZURE_ENDPOINT = "<pass endpoint>"

MODEL_REGISTRY = {
    "mock": {
        "deployment_name": "mock",
        "api_version": "v0",
        "azure_endpoint": None,
    },
    "gpt-3o-mini": {
        "deployment_name": "gpt-3o-mini",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
    },
    "gpt-35-turbo": {
        "deployment_name": "gpt-35-turbo",
        "api_version": "2024-12-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
    },
    "gpt-4o": {
        "deployment_name": "gpt-4o",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
    },
    "gpt-4o-mini": {
        "deployment_name": "gpt-4o-mini",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
    },
    "gpt-4.1": {
        "deployment_name": "gpt-4.1",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
    },
    "gpt-5": {
        "deployment_name": "gpt-5",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
    },
    "gpt-5-mini": {
        "deployment_name": "gpt-5-mini",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
    },
    "gpt-5-nano": {
        "deployment_name": "gpt-5-nano",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
    },
    "gpt-5-chat": {
        "deployment_name": "gpt-5-chat",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
    },
    "llama3.2": {
        "deployment_name": "llama3.2",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": None,
    },
    "dolphin-mistral": {
        "deployment_name": "dolphin-mistral",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": None,
    },
    "deepseek-v2": {
        "deployment_name": "deepseek-v2",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": None,
    },
    "hf": {
        "deployment_name": "hf",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": None,
    },
    "Mistral-7B-Instruct-v0.2-GPTQ": {
        "deployment_name": "Mistral-7B-Instruct-v0.2-GPTQ",
        "api_version": "2023-07-01-preview",
        "azure_endpoint": DEFAULT_AZURE_ENDPOINT,
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
