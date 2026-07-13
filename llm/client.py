# llm/client.py

import os
import time
import traceback
from openai import AzureOpenAI, BadRequestError
from llm.model_registry import MODEL_REGISTRY
from llm.llms import LLMType, GPT5_MODELS

from dotenv import load_dotenv
load_dotenv()

class OpenAIClient:
    total_token_usage = 0
    total_call_count = 0

    def __init__(self, model_enum: LLMType, api_key: str = None):
        self.model_enum = model_enum
        self.model_name = model_enum.value

        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model {self.model_name} not found in registry")

        self.model_config = MODEL_REGISTRY[self.model_name]

        self.api_version = self.model_config["api_version"]
        self.azure_endpoint = self.model_config["azure_endpoint"]
        self.deployment_name = self.model_config["deployment_name"]

        # Use passed api_key or fallback to environment variable
        self.api_key = api_key or os.getenv(
            f"{self.model_name.upper().replace('-', '_').replace('.', '_')}_API_KEY"
        )
        if not self.api_key:
            raise EnvironmentError(f"API key for model '{self.model_name}' not provided or set in environment")
    
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )

        self.token_usage = 0
        self.call_counter = 0

    def call(self, 
             prompt: str, 
             max_tokens=400, 
             temperature=0, 
             system_message=None, 
             context=None):
        
        if system_message is None:
            raise ValueError("System message must be provided")

        try:        
            self.call_counter += 1
            OpenAIClient.total_call_count += 1  # increment global call count

            start_time = time.time()
            formatted_system_msg = system_message.format(context) if context else system_message
            if LLMType(self.deployment_name) in GPT5_MODELS:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": formatted_system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=1
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": formatted_system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            end_time = time.time()

            used_tokens = response.usage.total_tokens
            self.token_usage += used_tokens
            OpenAIClient.total_token_usage += used_tokens  # increment global token usage

            response_msg = response.choices[0].message.content
            return response_msg, used_tokens, end_time - start_time

        except BadRequestError as e:
            print("[OpenAIClient] BadRequestError:")
            traceback.print_exc()

            error_data = getattr(e, "error", {})
            code = error_data.get("code", "")
            inner_error = error_data.get("innererror", {})
            return f"BADREQUEST_ERROR: {code} \n {inner_error}", self.token_usage, -1

        except Exception as e:
            print("Unhandled exception during OpenAI call")
            print("Model used:", self.model_enum)
            print("Prompt:", prompt)
        
            traceback.print_exc()
            raise e
    
    @classmethod
    def from_deployment_name(cls, deployment_name: str, api_key: str):
        # Find the model enum from the deployment name
        for model_name, config in MODEL_REGISTRY.items():
            if config["deployment_name"] == deployment_name:
                return cls(LLMType(model_name), api_key)
        raise ValueError(f"No model registered with deployment name '{deployment_name}'")