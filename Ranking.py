import pandas as pd


class Ranking:
    def get_top_n_similarities(self, siml_df, n=10):
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
