from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from Text_Processing import DataPreProcessing
import pandas as pd


class Query_Matching:
    def matching_query_documents(self, query, index):
        def custom_tokenizer(text):
            tokens = word_tokenize(text)
            return tokens

        date_pre_processing = DataPreProcessing()
        vectorizer = TfidfVectorizer(preprocessor=date_pre_processing.process_text, tokenizer=custom_tokenizer,
                                     stop_words='english')

        query_vector = vectorizer.transform([query])

        cosine_similarities = cosine_similarity(query_vector, index)
        df_similarity = pd.DataFrame(cosine_similarities)

        def get_top_n_similarities(siml_df, n=10):
            top_n_df = pd.DataFrame()
            for query_index in siml_df.index:
                top_n_similarities = siml_df.loc[query_index].nlargest(n)
                top_n_df_query = pd.DataFrame({
                    'query_id': query_index,
                    'doc_id': top_n_similarities.index,
                    'similarity': top_n_similarities.values
                })
                top_n_df = pd.concat([top_n_df, top_n_df_query], ignore_index=True)
            return top_n_df

        top_10_similarities = get_top_n_similarities(df_similarity, n=10)

        return top_10_similarities
