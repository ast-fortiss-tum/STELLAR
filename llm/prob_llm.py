import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class TokenProbEvaluator:
    map_templates = {
        "deepseek-ai/deepseek-coder-6.7b-instruct": """
                                    <|system|>
                                    You are an expert judge of in‑car ChatBot responses. 
                                    You MUST output exactly one token: "Yes" if the answer addresses the user’s utterance, "No" otherwise.
                                    <|user|>
                                    INPUT UTTERANCE: {}\nChatBot ANSWER: {}
                                    <|assistant|>
                                    """,
        "other": """You are an expert judge of in‑car ChatBot responses. 
                    You MUST output exactly one token: \"Yes\" if the answer addresses the user’s utterance, \"No\" otherwise.\n
                    INPUT UTTERANCE: \"{}\"\nChatBot ANSWER: \"{}\"\nJudge ANSWER:""",
    }

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        quantized: bool = False,
        prompt_template: str = "",
        generate: bool = False,
    ):
        self.model_name = model_name
        self.quantized = quantized

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()

        # Optional 4-bit quantization config
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="bitsandbytes_4bit",
            )
            if self.quantized
            else None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if self.quantized:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="sequential",
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="sequential",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.prompt_template = prompt_template
        self.generate = generate

        print("Model initialized on device:", self.model.device)

    def _make_prompt(self, utterance: str, answer: str) -> str:
        template = self.prompt_template
        if len(template) == 0:
            template = (
                self.map_templates[self.model]
                if self.model in self.map_templates
                else self.map_templates["other"]
            )
        prompt = template.format(utterance, answer)
        return prompt

    def evaluate(self, utterance: str, answer: str):
        prompt = self._make_prompt(utterance, answer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        YES_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("Yes")[0])
        NO_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("No")[0])

        if self.generate:
            generated = self.model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_ids = generated.sequences[0]
            new_token_ids = generated_ids[input_len:]
            output_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

            # Calculate probabilities

            if generated.scores:
                first_token_logits = generated.scores[0][0]
                probs = torch.softmax(first_token_logits, dim=-1)

                l_yes = first_token_logits[YES_ID]
                l_no = first_token_logits[NO_ID]
                p_yes = probs[YES_ID].item()
                p_no = probs[NO_ID].item()

                score = torch.sigmoid(l_yes - l_no).item()
                if isinstance(score, float) and not np.isnan(score):
                    prediction = "Yes" if l_yes > l_no else "No"
                else:
                    score = 0.5
                    prediction = "Unknown"
            else:
                p_yes = p_no = score = 0.0
                prediction = "Unknown"

            result = {
                "prediction": prediction,
                "score": score,
                "p_yes": p_yes,
                "p_no": p_no,
                "output_text": output_text,
            }
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
            first_token_logits = outputs.logits[0, -1]
            probs = torch.softmax(first_token_logits, dim=-1)

            l_yes = first_token_logits[YES_ID]
            l_no = first_token_logits[NO_ID]
            p_yes = probs[YES_ID].item()
            p_no = probs[NO_ID].item()

            score = torch.sigmoid(l_yes - l_no).item()
            if isinstance(score, float) and not np.isnan(score):
                prediction = "Yes" if l_yes > l_no else "No"
            else:
                score = 0.5
                prediction = "Unknown"

            result = {
                "prediction": prediction,
                "score": score,
                "p_yes": p_yes,
                "p_no": p_no,
            }

        return result


# Example usage
if __name__ == "__main__":
    evaluator = TokenProbEvaluator(
        model_name="meta-llama/Llama-3.2-3B-instruct", quantized=False
    )
    result = evaluator.evaluate(
        utterance="I am hungry",
        answer="Do you want me to look for a cinema?",
    )
    print(result)
