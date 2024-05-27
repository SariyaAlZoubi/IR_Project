import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextProcessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def remove_special_chars(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_urls(self, text):
        return re.sub(r'http\S+|www.\S+', '', text)

    def remove_emojis(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def preprocess(self, text):
        text = self.remove_numbers(text)
        text = self.remove_special_chars(text)
        words = text.lower().split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        words = [self.stemmer.stem(word) for word in words]
        return ' '.join(words)
