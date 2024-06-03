from Evalution import Evaluation


class PrecisionRecall:
    def precision_recall(self, top_10_similarities, qrels , evaluation):
        top_10_similarities['query_id'] = top_10_similarities['query_id'].astype(int)
        qrels['query_id'] = qrels['query_id'].astype(int)
        # Ensure the data types of doc_id columns are consistent
        top_10_similarities['doc_id'] = top_10_similarities['doc_id'].astype(int)
        qrels['doc_id'] = qrels['doc_id'].astype(int)

        query_ids = top_10_similarities['query_id'].unique()

        # Iterate over each query_id and calculate precision and recall at 10
        for query_id in query_ids:
            # Get the top 10 retrieved docs for this query

            retrieved_docs = top_10_similarities[top_10_similarities['query_id'] == query_id]['doc_id'].tolist()

            # Get the relevant docs for this query
            relevant_docs = qrels[(qrels['query_id'] == query_id) & (qrels['relevance'] == 1)]['doc_id'].tolist()

            # Debugging: Print the retrieved_docs and relevant_docs lists to ensure they are correct
            if (query_id < 20):
                print(f"Query ID: {query_id}")
                print(f"Retrieved Docs: {retrieved_docs}")
                print(f"Relevant Docs: {relevant_docs}")

            # Check if relevant_docs is empty
            if not relevant_docs:
                print(f"No relevant documents found for query_id {query_id}")
                continue

            # Calculate precision and recall at 10
            precision, recall = evaluation.precision_recall_at_10(relevant_docs, retrieved_docs)

            # Print the results
            if (query_id < 20):
                print(f"Precision at 10: {precision}")
                print(f"Recall at 10: {recall}")
                print("--------------------------------------------------------------------------")
