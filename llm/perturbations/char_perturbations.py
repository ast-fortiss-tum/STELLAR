"""
File contains character-level perturbations from METAL framework: https://arxiv.org/abs/2312.06056

Functions were inspired by METAL authors' repo: https://zenodo.org/records/10042353
"""
import random


def introduce_typos(input_text, typo_probability = 0.1):
    string_with_typo = ""

    for char in input_text:
        if random.uniform(0, 1) < typo_probability:
            string_with_typo += chr(random.randint(97, 122))  # random lowercase character
        else:
            string_with_typo += char

    return string_with_typo


def delete_characters(input_text, deletion_probability = 0.1):
    deleted_text = ""

    for char in input_text:
        if random.uniform(0, 1) >= deletion_probability:
            deleted_text += char
    
    return deleted_text


def shuffle_characters(input_text, shuffle_probability = 0.2):
    words = input_text.split()
    new_words = []
    
    for word in words:
        if len(word) <= 3 or random.uniform(0, 1) > shuffle_probability:
            new_words.append(word)
        else:
            first_char = word[0]
            last_char = word[-1]
            middle_chars = list(word[1:-1])
            random.shuffle(middle_chars)
            shuffled_word = first_char + ''.join(middle_chars) + last_char
            new_words.append(shuffled_word)
            
    return ' '.join(new_words)


def add_characters(input_string, addition_probability = 0.1):
    string_with_addition = ""

    for char in input_string:
        if random.uniform(0, 1) < addition_probability:
            string_with_addition += chr(random.randint(97, 122))
            string_with_addition += char
        else:
            string_with_addition += char

    return string_with_addition


def to_leet(input_string, leet_probability = 0.2):
    """
    Implement the basic version of leet (aka 1337) with the characters.

    Full version info: https://en.wikipedia.org/wiki/Leet#Table_of_leet-speak_substitutes_for_normal_letters
    """
    leet_mapping = {
        'a': '4',
        'e': '3',
        'i': '1',
        'o': '0'
    }
    leet_words = []
    words = input_string.split()

    for word in words:
        if random.uniform(0, 1) < leet_probability:
            leet_text = ''
            for char in word:
                lowercase_char = char.lower()
                if lowercase_char in leet_mapping:
                    leet_text += leet_mapping[lowercase_char]
                else:
                    leet_text += char
                    
            leet_words.append(leet_text)
        else:
            leet_words.append(word)
            
    return ' '.join(leet_words)


def add_spaces(input_string, space_probability = 0.3):
    spaced_string = ""

    for char in input_string:
        if random.uniform(0, 1) < space_probability:
            spaced_string += ' '
            spaced_string += char
        else:
            spaced_string += char

    return spaced_string


def swap_characters(input_text, swap_probability = 0.2):
    words = input_text.split()
    new_words = []
    
    for word in words:
        if len(word) <= 3 or random.uniform(0, 1) > swap_probability:
            new_words.append(word)
        else:
            first_char = word[0]
            last_char = word[-1]
            middle_chars = list(word[1:-1])
            char_no = range(len(middle_chars))
            c1, c2 = random.sample(char_no, 2)
            middle_chars[c1], middle_chars[c2] = middle_chars[c2], middle_chars[c1]
            swapped_word = first_char + ''.join(middle_chars) + last_char
            new_words.append(swapped_word)
            
    return ' '.join(new_words)


if __name__ == "__main__":
    # TODO: integrate LLM calling logic
    test_string = """About noon I stopped at the captain’s door with some cooling drinks and medicines. He was lying very much as we had left him, only a little higher, and he seemed both weak and excited."""

    print(f"\noriginal text: {test_string}\n")
    print(f"introduce typos: {introduce_typos(test_string)}\n")
    print(f"delete characters: {delete_characters(test_string)}\n")
    print(f"shuffle characters: {shuffle_characters(test_string)}\n")
    print(f"add characters: {add_characters(test_string)}\n")
    print(f"to 1337 : {to_leet(test_string)}\n")
    print(f"add spaces: {add_spaces(test_string)}\n")
    print(f"swap characters: {swap_characters(test_string)}\n")

CHAR_PERTURBATIONS = {
    # TODO: decide which perturbations are relevent to the voice assistant case
    "introduce_typos" : introduce_typos,  
    "delete_characters" : delete_characters,
    "add_characters" : add_characters,
    "add_spaces" : add_spaces,
    "swap_characters" : swap_characters,
    "shuffle_characters" : shuffle_characters,
}