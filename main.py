import pickle

import ir_datasets
import pandas as pd
from Index import Index
from Query_Matching import Query_Matching
from Ranking import Ranking
from documents import Documents


class Main:

    def run(self, request):
        document = Documents()

        if request.dataSet == "lifeStyle":
            with open('tfidf_matrix_lifestyle.pkl', 'rb') as file:
                tfidf_matrix = pickle.load(file)

            with open('vectorizer_lifestyle.pkl', 'rb') as file:
                vectorizer = pickle.load(file)
        else:
            with open('tfidf_matrix_recreation.pkl', 'rb') as file:
                tfidf_matrix = pickle.load(file)

            with open('vectorizer_recreation.pkl', 'rb') as file:
                vectorizer = pickle.load(file)

        queryMatching = Query_Matching()

        df_similarity = queryMatching.matching_query_documents(request.query, vectorizer, tfidf_matrix)

        rank = Ranking()

        top_10_similarities = rank.get_top_n_similarities(df_similarity)
        result = document.docs(top_10_similarities)
        return result
