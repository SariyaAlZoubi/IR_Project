import pandas as pd


class Ranking:

    def precision_recall_at_10(self, relevant_docs, retrieved_docs):
        k = 10
        # Ensure we do not exceed the length of the retrieved_docs list
        retrieved_k = retrieved_docs[:k]
        # Calculate the number of relevant and retrieved documents
        relevant_and_retrieved = len(set(retrieved_k) & set(relevant_docs))
        # Precision: proportion of retrieved documents that are relevant
        precision = relevant_and_retrieved / k
        # Recall: proportion of relevant documents that are retrieved
        recall = relevant_and_retrieved / len(relevant_docs)
        return precision, recall

    def calculate_map_at_k(self, top_10_similarities: pd.DataFrame, qrels: pd.DataFrame, k: int = 10) -> float:
        # Ensure correct data types
        top_10_similarities = top_10_similarities.astype({"query_id": int, "doc_id": int, "similarity": float})
        qrels = qrels.astype({"query_id": int, "doc_id": int, "relevance": int})

        # Parse qrels to create a dictionary of relevant documents for each query
        qrels_dict = {}
        for entry in qrels.itertuples(index=False):
            query_id = entry.query_id
            doc_id = entry.doc_id
            if query_id not in qrels_dict:
                qrels_dict[query_id] = []
            qrels_dict[query_id].append(doc_id)

        # Parse top_similarity to create a list of predicted documents for each query
        predicted_dict = {}
        for entry in top_10_similarities.itertuples(index=False):
            query_id = entry.query_id
            doc_id = entry.doc_id
            if query_id not in predicted_dict:
                predicted_dict[query_id] = []
            predicted_dict[query_id].append(doc_id)

        # Initialize variables
        Q = len(qrels_dict)  # number of queries
        ap = []

        # Calculate AP for each query
        for q in qrels_dict:
            actual = qrels_dict[q]
            predicted = predicted_dict.get(q, [])
            ap_num = 0
            rel_count = 0

            for x in range(min(k, len(predicted))):
                if predicted[x] in actual:
                    rel_count += 1
                    precision_at_k = rel_count / (x + 1)
                    ap_num += precision_at_k

            if len(actual) > 0:
                ap_q = ap_num / len(actual)
                ap.append(ap_q)

        # Calculate MAP
        map_at_k = sum(ap) / Q
        return round(map_at_k, 4)
