class Documents:
    def docs(self,top_10_similarities):
        query_ids = top_10_similarities['query_id'].unique()
        print(len(query_ids))
        result = []
        f= ["a","b","v","a"]
        for x in f:
            obj = {
                'id': "1",
                'text': x
            }
            result.append(obj)


        return result