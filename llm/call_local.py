from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import torch

class LLMHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.supported_LLMs = [
                "meta-llama/Meta-Llama-3.1-8B",
                "meta-llama/Llama-3.1-8B-Instruct",
                "EleutherAI/gpt-j-6B",
                "EleutherAI/gpt-neo-2.7B",
                "EleutherAI/gpt-neo-1.3B",
                "openlm-research/open_llama_3b",
                "facebook/opt-2.7b",
                "facebook/opt-1.3b",
                "facebook/opt-350m",
                "ibm-granite/granite-3.1-8b-instruct",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "TheBloke/Mistral-7B-v0.1-GGUF",
                "microsoft/phi-2"
            ]
            self.model = None
            self.tokenizer = None
            self._initialized = True

    def load_LLM(self, LLM):
        if LLM not in self.supported_LLMs:
            raise ValueError(f"LLM '{LLM}' is not supported.")
        quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16",  
                        bnb_4bit_use_double_quant=True,    
                        bnb_4bit_quant_type="nf4",          
        )
                    
        self.tokenizer = AutoTokenizer.from_pretrained(LLM, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM,
            device_map="auto",  
            quantization_config=quantization_config,
        )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     LLM, device_map=torch.device("cpu"), torch_dtype=torch.float32
        # )

    def ask_LLM(self, question, max_length=150):  # Reduced max_length for efficiency
        if not self.model or not self.tokenizer:
            raise ValueError("LLM model is not loaded. Please load the model first.")
        
        inputs = self.tokenizer(question, return_tensors="pt")

        start_time = time.time()

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,  # Ensure only one output
            eos_token_id=self.tokenizer.eos_token_id,  # Stops generation at EOS
            num_beams=1,  # No beam search
            do_sample=False,  # No randomness
            return_dict_in_generate=True,  # Get structured output
            output_scores=False  # Disable extra scores
        )

        end_time = time.time()
        response_time = end_time - start_time

        # Decode and ensure only one output
        decoded_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Count input tokens
        input_token_count = len(inputs)  # Number of input tokens

        # Count output tokens
        output_token_count = outputs.sequences.shape[1]  # Number of output tokens

        # Calculate total token count
        total_token_count = input_token_count + output_token_count

        print("total tokens", total_token_count)
        # Force stopping at first occurrence of EOS if needed
        stop_token = self.tokenizer.eos_token
        if stop_token and stop_token in decoded_text:
            decoded_text = decoded_text.split(stop_token)[0].strip()
        return decoded_text, response_time, total_token_count

if __name__ == "__main__":
    # Example usage
    model = "ibm-granite/granite-3.1-8b-instruct"
    model = "facebook/opt-1.3b" 
    # model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    handler = LLMHandler()
    handler.load_LLM(model)

    print(handler.ask_LLM("1 + 1?"))

    