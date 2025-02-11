import re
from lemmatization_for_stopwords import cached_parse

def load_stop_words(filepath: str) -> set:
    with open(filepath, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


RUSSIAN_STOP_WORDS = load_stop_words("db_and_weights/ru_stopwords.txt")


def preprocess_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    words = text.split()
    filtered_words = [cached_parse(word) for word in words if cached_parse(word) not in RUSSIAN_STOP_WORDS]
    return filtered_words
