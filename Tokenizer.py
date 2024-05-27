from nltk.tokenize import word_tokenize


class Tokenizer:
    def tokenize_text(self, text):
        return word_tokenize(text)
