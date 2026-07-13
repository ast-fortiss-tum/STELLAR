from llm.config import DEPLOYMENT_NAME
from dotenv import load_dotenv
import os
from llm.client import OpenAIClient
from llm.llms import LLMType

load_dotenv()  # By default, loads from .env in current directory

def get_key():
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key

def call_openai(prompt, 
                max_tokens = 400, 
                temperature = 0, 
                system_message = None,
                context = None,
                model = DEPLOYMENT_NAME):

    if system_message is None:
        raise ValueError("system_message must be provided")
 
    api_key = get_key()

    # Initialize the client for the specified model (DEPLOYMENT_NAME by default)
    client = OpenAIClient.from_deployment_name(model, api_key=api_key)

    # Call the model and return results
    return client.call(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_message=system_message,
        context=context
    )

def get_openai_client(model):
    api_key = get_key()
    # Initialize the client for the specified model (DEPLOYMENT_NAME by default)
    client = OpenAIClient.from_deployment_name(model, api_key=api_key).client
    return client

if __name__ == "__main__":
    input = "I am in the mood for nice music."
    prompt = """
                Rephrase the following user utterance by using synonyms. Use synonyms that dont change the meaning.
                Provide 2 variations. 

                **Example**

                Input: I am very hungry.
                Ouput: [I am so much hungry., I am very open minded for nice food.]

                **Your turn**

                Input: {}
                Output: 

                """
    print(call_openai(prompt=prompt.format(input),
                      max_tokens=1000,
                      temperature=0.1,
                      system_message="You are a system for replacing words with synonyms." + \
                                    "Only output the result. Dont explain. The results should be human like and natural." +  \
                                    "Replace only single words.",
                      context=None))