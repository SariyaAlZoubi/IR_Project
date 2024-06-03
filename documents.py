from sqlalchemy import create_engine, select, Table, MetaData
from sqlalchemy.orm import sessionmaker
import pandas as pd
from database import create_session, engine


class Documents:
    session = create_session()
    metadata = MetaData()
    metadata.reflect(bind=engine)

    def fetch_documents(self,top_10,type):
        document_list = []
        # Assuming the table name is 'documents'
        documents_table = metadata.tables[type]
        for index, row in top_10.iterrows():
            doc_id = row['doc_id']
            # Query the database to fetch the document with the current doc_id
            query = select([documents_table]).where(documents_table.c.doc_id == doc_id)
            result = session.execute(query).fetchone()

            if result:
                # Convert the result to a dictionary and append to the list
                document_dict = dict(result)
                document_list.append(document_dict)

        return document_list
