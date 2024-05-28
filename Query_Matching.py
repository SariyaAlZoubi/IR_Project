from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from Text_Processing import TextProcessing


class Query_Matching:
    def matching_query_documents(self, query, index, documents):
        def custom_tokenizer(text):
            tokens = word_tokenize(text)
            return tokens

        text_processing = TextProcessing()
        vectorizer = TfidfVectorizer(preprocessor=text_processing.preprocess, tokenizer=custom_tokenizer,
                                     stop_words='english')

        tf = vectorizer.fit_transform(documents['text'])

        query_vector = vectorizer.transform([query])

        cosine_similarities = cosine_similarity(query_vector, index)

        matched_docs_with_scores = [(doc, score) for doc, score in zip(documents['text'], cosine_similarities[0])]
        matched_docs_with_scores = sorted(matched_docs_with_scores, key=lambda x: x[1], reverse=True)

        return matched_docs_with_scores
