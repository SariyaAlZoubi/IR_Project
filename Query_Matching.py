from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from Text_Processing import DataPreProcessing
import pandas as pd


class Query_Matching:
    def matching_query_documents(self, query, vectorizer, tfidf_matrix):
        query_vector = vectorizer.transform([query])

        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
        df_similarity = pd.DataFrame(cosine_similarities)
        return df_similarity
