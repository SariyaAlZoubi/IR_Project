import re
from sklearn.feature_extraction.text import TfidfVectorizer
from Text_Processing import TextProcessing
import pandas as pd
from nltk.tokenize import word_tokenize


class Data_Representation:
    def data_representation(self, documents):
        text_processing = TextProcessing()

        def custom_tokenizer(text):
            tokens = word_tokenize(text)
            return tokens

        vectorizer = TfidfVectorizer(preprocessor=text_processing.preprocess, tokenizer=custom_tokenizer,
                                     stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents['text'])
        df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        print(df)
        vocabulary = vectorizer.get_feature_names_out()
        return tfidf_matrix, vocabulary, vectorizer
