from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from Text_Processing import DataPreProcessing
from sklearn.feature_extraction.text import CountVectorizer
import pickle


# تعريف custom_tokenizer خارج الكلاس
def custom_tokenizer(text):
    tokens = word_tokenize(text)
    return tokens


class Index:
    def indexing(self, documents):
        date_pre_processing = DataPreProcessing()
        vectorizer = TfidfVectorizer(preprocessor=date_pre_processing.process_text, max_features=25000)
        tfidf_matrix = vectorizer.fit_transform(documents['text'])

        # حفظ tfidf_matrix
        with open('tfidf_matrix_lifestyle.pkl', 'wb') as file:
            pickle.dump(tfidf_matrix, file)

        # حفظ vectorizer
        with open('vectorizer_lifestyle.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)

        return vectorizer, tfidf_matrix