import os
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import time
from llm.model_registry import MODEL_REGISTRY
import traceback
from azure.core.exceptions import HttpResponseError

load_dotenv()  # By default, loads from .env in current directory

def load_deepseek_client(model_name: str):
    from azure.ai.inference import ChatCompletionsClient

    # Get config from registry
    config = MODEL_REGISTRY.get(model_name)
    if config is None:
        raise ValueError(f"Model '{model_name}' not found in registry")

    endpoint = config["azure_endpoint"]
    api_version = config["api_version"]
    deployment_name = config["deployment_name"]

    if model_name == "DeepSeek-R1-qcbar":
        api_key = os.getenv("DEEPSEEK_R1_API_KEY")
    elif model_name == "DeepSeek-V3-0324":
        api_key = os.getenv("DEEPSEEK_V3_API_KEY")
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
        api_version=api_version
    )

    return client, deployment_name


def call_deepseek(deployment_name: str, 
                  prompt: str, 
                  max_tokens=400, 
                  temperature=0, 
                  system_message=None, 
                  context=None):
    
    client, _= load_deepseek_client(model_name=deployment_name)
    
    if system_message is None:
        raise ValueError("System message must be provided")

    try:
        start_time = time.time()
        formatted_system_msg = system_message.format(context) if context else system_message

        response = client.complete(
            model=deployment_name,
            messages=[
                SystemMessage(content=formatted_system_msg),
                UserMessage(content=prompt)
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        end_time = time.time()

        # Note: azure.ai.inference does not currently return token usage metadata
        response_msg = response.choices[0].message.content
        used_tokens = 0  # Placeholder if token usage is not available

        return response_msg, used_tokens, end_time - start_time

    except HttpResponseError as e:
        print("[DeepSeekClient] HttpResponseError:")
        traceback.print_exc()
        return f"HTTP_RESPONSE_ERROR: {str(e)}", None, -1

    except Exception as e:
        print("[DeepSeekClient] Unhandled exception during DeepSeek call")
        print("Prompt:", prompt)
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    model_name = "DeepSeek-V3-0324"  # or "DeepSeek-V3-0324"
    prompt = "What is so great about Paris?"
    system_message = "You are a helpful assistant."

    client, deployment_name = load_deepseek_client(model_name)

    print(f"Calling model '{model_name}' at deployment '{deployment_name}'...")
    response, tokens, duration = call_deepseek(
        deployment_name=deployment_name,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.7,
        system_message=system_message
    )
    print("reponse:", response, tokens, duration)