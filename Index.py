from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from Text_Processing import DataPreProcessing
from sklearn.feature_extraction.text import CountVectorizer


class Index:
    def indexing(self, documents):
        def custom_tokenizer(text):
            tokens = word_tokenize(text)
            return tokens

        date_pre_processing = DataPreProcessing()
        vectorizer = TfidfVectorizer(preprocessor=date_pre_processing.process_text, tokenizer=custom_tokenizer,
                                     stop_words='english')
        c = CountVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents['text'])
        return tfidf_matrix
