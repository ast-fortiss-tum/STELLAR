import os
import re
import json
import random
import time

import pronouncing
import pandas as pd

from llm.llms import LLMType
from llm.llms import pass_llm, LLMType

def _get_homophone_pronouncing(word):
    """
    Uses the Carnegie Mellon Pronouncing Dictionary - i.e. American pronounciation.
    """
    pronunciations = pronouncing.phones_for_word(word.lower())
    homophones = set()

    for pronunciation in pronunciations:
        words_with_same_sound = pronouncing.search(f"^{pronunciation}$")
        homophones.update(words_with_same_sound)

    homophones.discard(word.lower())  # original word is not needed
    if not homophones:
        return None
    
    return random.choice(list(homophones))


def _get_homophones_whole_text_prompt(text):
    homophones_prompt = f'''Given the text: "{text}"

                            Replace some words in the text with their homophones (words that sound the same but are spelled differently). 
                            Return ONLY the modified text with homophones substituted.

                            IMPORTANT: 
                            - The homophones you use should be real words
                            - Keep the original capitalization and punctuation
                            - If no suitable homophones exist, return the original text
                            '''
    return homophones_prompt


def _get_fillers_prompt(text):
    fillers_prompt = f'''Given the text: "{text}"

                        Insert 1-2 natural filler words into the text to make it sound more conversational and natural. 
                        Return ONLY the modified text with fillers inserted.

                        Use common filler words like: "uh", "um", "like", "you know", "I mean", "well", "so", "actually", "basically", or others if you think they are relevant.

                        IMPORTANT:
                        - Insert fillers at natural pause points (not in the middle of phrases)
                        - Keep the original meaning and flow
                        - Use fillers that fit the conversational tone
                        - Don't overuse fillers - 1-2 insertions maximum
                        - Maintain original punctuation and capitalization
                     '''
    
    return fillers_prompt


def introduce_homophones_llm(input_text, model = LLMType.GPT_4O_MINI):
    """
    Introduce homophones using LLM by processing the whole text at once.

    Parameters:
        input_text (str): The text where homophones should be introduced.
        model (LLMType): LLM used for introducing homophones.

    Returns:
        str: The modified text with homophones introduced where applicable.
    """
    system_prompt = "You are a helpful assistant that replaces words with their homophones while maintaining text readability."
    prompt = _get_homophones_whole_text_prompt(input_text)

    response = pass_llm(
        msg=prompt,
        system_message=system_prompt,
        llm_type=model,
    )

    # clean up the response - remove any quotes or extra formatting
    result = response.strip()
    
    # if the LLM returns the text in quotes, remove them
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    
    return result

import re
import random

def introduce_homophones_pronouncing_ratio(input_text, homophone_ratio=0.5):
    """
    Replace a percentage of words that have homophones using pronouncing library.
    Only applies replacement probability to words with homophones available.
    """
    words = input_text.split()
    
    # Step 1: Identify all words that have homophones
    homophone_candidates = []
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w]', '', word)
        if _get_homophone_pronouncing(clean_word):
            homophone_candidates.append(i)  # store the index of the word
    
    # Step 2: Decide how many words to replace based on probability
    num_to_replace = int(len(homophone_candidates) * homophone_ratio)
    indices_to_replace = set(random.sample(homophone_candidates, num_to_replace))
    
    # Step 3: Build result with replacements
    result_words = []
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w]', '', word)
        if i in indices_to_replace:
            homophone = _get_homophone_pronouncing(clean_word)
            
            # Preserve capitalization
            if clean_word[0].isupper() and clean_word[1:].islower():
                homophone = homophone.capitalize()
            elif clean_word.isupper():
                homophone = homophone.upper()
            
            # Preserve punctuation
            if word != clean_word:
                punctuation = re.sub(r'\w', '', word)
                homophone += punctuation
            
            result_words.append(homophone)
        else:
            result_words.append(word)
    
    return ' '.join(result_words)

def introduce_homophones_pronouncing(input_text, homophone_probability=0.25):
    """
    Replace words with their homophones using pronouncing library.
    Only applies probability to words that actually have homophones available.
    """
    words = input_text.split()
    result_words = []
    
    for word in words:
        # clean word for homophone lookup (remove punctuation)
        clean_word = re.sub(r'[^\w]', '', word)
        homophone = _get_homophone_pronouncing(clean_word)
        
        if homophone:  # word has homophones available
            if random.random() < homophone_probability:
                if clean_word[0].isupper() and clean_word[1:].islower():
                    homophone = homophone.capitalize()
                elif clean_word.isupper():
                    homophone = homophone.upper()
                
                # preserve punctuation from original word
                if word != clean_word:
                    punctuation = re.sub(r'\w', '', word)
                    result_words.append(homophone + punctuation)
                else:
                    result_words.append(homophone)
            else:
                result_words.append(word)
        else:
            result_words.append(word)
    
    return ' '.join(result_words)


def introduce_fillers(input_text):
    # filler words based on Laserna et al. (2014)
    fillers = ["I mean", "you know", "like", "uh", "um"]

    words = input_text.split()
    if len(words) < 2:
        return input_text  # too short to insert fillers meaningfully

    # choose a random position, but not the very end
    insert_pos = random.randint(0, len(words) - 2)
    filler = random.choice(fillers)

    # insert filler at the chosen position
    modified_words = words[:insert_pos+1]
    return " ".join(modified_words) + f", {filler}, " + " ".join(words[insert_pos+1:])


def introduce_fillers_llm(input_text, model=LLMType.GPT_4O_MINI):
    """
    Introduce filler words using LLM by processing the whole text at once.
    """
    system_prompt = "You are a helpful assistant that inserts natural filler words into text to make it sound more conversational."
    prompt = _get_fillers_prompt(input_text)

    response = pass_llm(
        msg=prompt,
        system_message=system_prompt,
        llm_type=model,
    )
    result = response.strip()
    
    # if the LLM returns the text in quotes, remove them
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    
    return result

def delete_words(input_text, deletion_probability = 0.4):
    """
    Delete some types of words using predefined word lists instead.

    The word lists were provided by discussions with Company Experts.
    """
    articles = {'a', 'an', 'the', 'A', 'An', 'The'}

    pronouns = {
        # personal pronouns
        'i', 'me', 'my', 'mine', 'myself',
        'you', 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself',
        'we', 'us', 'our', 'ours', 'ourselves',
        'they', 'them', 'their', 'theirs', 'themselves',
        # demonstrative pronouns
        'this', 'that', 'these', 'those',
        # interrogative pronouns
        'who', 'whom', 'whose', 'which', 'what',
        # indefinite pronouns
        'all', 'another', 'any', 'anybody', 'anyone', 'anything',
        'both', 'each', 'either', 'everybody', 'everyone', 'everything',
        'few', 'many', 'neither', 'nobody', 'none', 'nothing',
        'one', 'other', 'others', 'several', 'some', 'somebody',
        'someone', 'something',
        # capitalised versions
        'I', 'Me', 'My', 'Mine', 'Myself',
        'You', 'Your', 'Yours', 'Yourself', 'Yourselves',
        'He', 'Him', 'His', 'Himself',
        'She', 'Her', 'Hers', 'Herself',
        'It', 'Its', 'Itself',
        'We', 'Us', 'Our', 'Ours', 'Ourselves',
        'They', 'Them', 'Their', 'Theirs', 'Themselves',
        'This', 'That', 'These', 'Those',
        'Who', 'Whom', 'Whose', 'Which', 'What',
        'All', 'Another', 'Any', 'Anybody', 'Anyone', 'Anything',
        'Both', 'Each', 'Either', 'Everybody', 'Everyone', 'Everything',
        'Few', 'Many', 'Neither', 'Nobody', 'None', 'Nothing',
        'One', 'Other', 'Others', 'Several', 'Some', 'Somebody',
        'Someone', 'Something'
    }

    prepositions = {
        'about', 'above', 'across', 'after', 'against', 'along', 'amid',
        'among', 'amongst', 'around', 'as', 'at', 'before', 'behind',
        'below', 'beneath', 'beside', 'besides', 'between', 'beyond',
        'by', 'concerning', 'considering', 'despite', 'down', 'during',
        'except', 'following', 'for', 'from', 'given', 'in', 'including',
        'inside', 'into', 'like', 'minus', 'near', 'notwithstanding',
        'of', 'off', 'on', 'onto', 'opposite', 'out', 'outside', 'over',
        'past', 'per', 'plus', 'regarding', 'round', 'save', 'since',
        'than', 'through', 'throughout', 'till', 'to', 'toward', 'towards',
        'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus',
        'via', 'with', 'within', 'without',
        # capitalised versions
        'About', 'Above', 'Across', 'After', 'Against', 'Along', 'Amid',
        'Among', 'Amongst', 'Around', 'As', 'At', 'Before', 'Behind',
        'Below', 'Beneath', 'Beside', 'Besides', 'Between', 'Beyond',
        'By', 'Concerning', 'Considering', 'Despite', 'Down', 'During',
        'Except', 'Following', 'For', 'From', 'Given', 'In', 'Including',
        'Inside', 'Into', 'Like', 'Minus', 'Near', 'Notwithstanding',
        'Of', 'Off', 'On', 'Onto', 'Opposite', 'Out', 'Outside', 'Over',
        'Past', 'Per', 'Plus', 'Regarding', 'Round', 'Save', 'Since',
        'Than', 'Through', 'Throughout', 'Till', 'To', 'Toward', 'Towards',
        'Under', 'Underneath', 'Unlike', 'Until', 'Up', 'Upon', 'Versus',
        'Via', 'With', 'Within', 'Without'
    }

    target_words = articles | pronouns | prepositions
    
    words = input_text.split()
    remaining_words = []
    
    for word in words:
        if word in target_words:
            if random.random() > deletion_probability:
                remaining_words.append(word)
        else:
            remaining_words.append(word)
    
    return ' '.join(remaining_words)


def test_perturbation_methods(test_strings, output_name="homophone_test_results"):
    """
    Test different homophone introduction methods, and fillers introduction on a list of test strings.
    """
    results_data = []

    for i, test_string in enumerate(test_strings):
        print(f"=== Test String {i+1} ===")
        print(f"Original: {test_string}")
        print()

        row_data = {"Original Phrase": test_string}
        
        # test introduce_homophones_pronouncing
        start = time.time()
        result_pronouncing = introduce_homophones_pronouncing(test_string)
        end = time.time()
        pronouncing_time = end - start
        print(f"introduce_homophones_pronouncing ({pronouncing_time:.4f}s): {result_pronouncing}")
        
        row_data["Pronouncing"] = result_pronouncing
        row_data["Pron. Time"] = pronouncing_time
        
        # test introduce_homophones_llm with GPT-4o
        start = time.time()
        result_llm = introduce_homophones_llm(test_string, LLMType.GPT_4O)
        end = time.time()
        llm_time = end - start
        print(f"introduce_homophones_llm GPT-4o ({llm_time:.4f}s): {result_llm}")    

        row_data["GPT-4O"] = result_llm
        row_data["4O Time"] = llm_time

        # test introduce_homophones_llm with GPT-4o-mini
        start = time.time()
        result_llm_4o_mini = introduce_homophones_llm(test_string, LLMType.GPT_4O_MINI)
        end = time.time()
        llm_4o_mini_time = end - start
        print(f"introduce_homophones_llm GPT-4o-mini ({llm_4o_mini_time:.4f}s): {result_llm_4o_mini}")

        row_data["GPT-4O-Mini"] = result_llm_4o_mini
        row_data["4O-Mini Time"] = llm_4o_mini_time
        
        # test introduce_homophones_llm with GPT-5
        start = time.time()
        result_llm_5 = introduce_homophones_llm(test_string, LLMType.GPT_5)
        end = time.time()
        llm_5_time = end - start
        print(f"introduce_homophones_llm GPT-5 ({llm_5_time:.4f}s): {result_llm_5}")

        row_data["GPT-5"] = result_llm_5
        row_data["5 Time"] = llm_5_time

        # test introduce_homophones_llm with GPT-5-mini
        start = time.time()
        result_llm_5_mini = introduce_homophones_llm(test_string, LLMType.GPT_5_MINI)
        end = time.time()
        llm_5_mini_time = end - start
        print(f"introduce_homophones_llm GPT-5-mini ({llm_5_mini_time:.4f}s): {result_llm_5_mini}")

        row_data["GPT-5-Mini"] = result_llm_5_mini
        row_data["5-Mini Time"] = llm_5_mini_time

        # test introduce_fillers (random)
        start = time.time()
        result_fillers = introduce_fillers(test_string)
        end = time.time()
        fillers_time = end - start
        print(f"introduce_fillers ({fillers_time:.4f}s): {result_fillers}")

        row_data["Fillers Random"] = result_fillers
        row_data["Fillers Random Time"] = fillers_time
        
        # test introduce_fillers_llm with GPT-4o
        start = time.time()
        result_fillers_llm = introduce_fillers_llm(test_string, LLMType.GPT_4O)
        end = time.time()
        fillers_llm_time = end - start
        print(f"introduce_fillers_llm GPT-4o ({fillers_llm_time:.4f}s): {result_fillers_llm}")

        row_data["Fillers LLM 4O"] = result_fillers_llm
        row_data["Fillers LLM 4O Time"] = fillers_llm_time
        
        # test introduce_fillers_llm with GPT-4o-mini
        start = time.time()
        result_fillers_llm_mini = introduce_fillers_llm(test_string, LLMType.GPT_4O_MINI)
        end = time.time()
        fillers_llm_mini_time = end - start
        print(f"introduce_fillers_llm GPT-4o-mini ({fillers_llm_mini_time:.4f}s): {result_fillers_llm_mini}")

        row_data["Fillers LLM 4O-Mini"] = result_fillers_llm_mini
        row_data["Fillers LLM 4O-Mini Time"] = fillers_llm_mini_time

        results_data.append(row_data)
        print("-" * 50)
        print()
    
    # create DataFrame & save to Excel
    df = pd.DataFrame(results_data)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_filename = os.path.join(current_dir, f"{output_name}.xlsx")
    df.to_excel(excel_filename, index=False)
    print(f"\nResults saved to: {excel_filename}")
    
    return df

if __name__ == "__main__":
    test_strings = [
        "I need to eat something.",
        "I feel like eating something Italian",
        "Is there some Japanese place nearby?",
        "I am lactose intolerant",
        "Wanna pay with my credit card",
        "How is the traffic?",
        "I prefer scenic routes",
        "Find the fastest route to the central train station",
        "Where is the nearest pharmacy located?",
    ]
    results_df = test_perturbation_methods(test_strings, "homophone_test_results_final")


WORD_PERTURBATIONS = {
    "delete_words" : delete_words,
    "introduce_homophones_llm" : introduce_homophones_llm,
    "introduce_homophones_static" : introduce_homophones_pronouncing_ratio,
    "introduce_fillers_llm" : introduce_fillers_llm,
    "introduce_fillers_static" : introduce_fillers
}
