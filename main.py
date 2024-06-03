import ir_datasets
import pandas as pd
from Index import Index
from Query_Matching import Query_Matching
from Ranking import Ranking
dataset = ir_datasets.load("lotte/lifestyle/dev/forum")
docs = pd.DataFrame(dataset.docs_iter())
documents = docs[:1000]


index = Index()
vectorizer, tfidf_matrix = index.indexing(documents)

query = "SSSSSSsssS"

queryMatching = Query_Matching()

df_similarity = queryMatching.matching_query_documents(query,vectorizer,tfidf_matrix)

rank = Ranking()

top_n_df = rank.get_top_n_similarities(df_similarity)
