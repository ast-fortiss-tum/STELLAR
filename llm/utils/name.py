from llm.config import LLM_IPA

def create_problem_name(simulate_function, suffix):
    print(simulate_function)
    # Check if simulate_function is bound to IPA.simulate
    if "IPAOLLAMA" in str(simulate_function):
        return f"IPA_Ollama_{LLM_IPA}_" + suffix   
    elif "IPA_LOS" in str(simulate_function):
        return f"IPA_LOS_{LLM_IPA}_" + suffix    
    elif "YELP" in str(simulate_function):
        return f"IPA_YELP_{LLM_IPA}_" + suffix    
    elif "IPA" in str(simulate_function):
        return f"IPA_{LLM_IPA}_" + suffix   
    else:
        raise ValueError("Simulate function not known.")
    