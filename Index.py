from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from Text_Processing import DataPreProcessing
from sklearn.feature_extraction.text import CountVectorizer


class Index:
    def indexing(self, documents):
        date_pre_processing = DataPreProcessing()
        vectorizer = TfidfVectorizer(preprocessor=date_pre_processing.process_text, max_features=25000)
        tfidf_matrix = vectorizer.fit_transform(documents['text'])
        return vectorizer, tfidf_matrix
