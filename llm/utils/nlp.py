from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')

def compute_word_length(text: str) -> int:
    try:
        words = word_tokenize(text)
        num = len([w for w in words if w.isalnum()])
    except Exception as e:
        print(e)
        print("Could not compute word length, stalling to default.")
        num = -1
    return num

if __name__ == "__main__":
    data = ["I don't know how to help you.",
            "I found something. You want directions?"
            ]
    for d in data:
        print(compute_word_length(d))