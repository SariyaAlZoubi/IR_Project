import pandas as pd
from database import Lifestyle, Recreation, create_session

csv_file_path_recreation = 'recreation_docs.csv'
csv_file_path_lifestyle = 'lifestyle.csv' #

recreation = pd.read_csv(csv_file_path_recreation)
lifestyle = pd.read_csv(csv_file_path_lifestyle)

# Create a session
session = create_session()

for index, row in recreation.iterrows():
    recreation_instance = Recreation(
        doc_id=row['doc_id'],
        text=row['text']
    )
    session.add(recreation_instance)

for index, row in lifestyle.iterrows():
    lifestyle_instance = Lifestyle(
        doc_id=row['doc_id'],
        text=row['text']
    )
    session.add(lifestyle_instance)

# Commit the session to save changes
session.commit()

# Close the session
session.close()
