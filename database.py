from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Create engine
engine = create_engine("mysql+pymysql://root:@127.0.0.1:3306/ir", pool_size=10, max_overflow=20)

# Define base
Base = declarative_base()

# Define the Recreation class
class Recreation(Base):
    __tablename__ = 'recreation_docs'

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    doc_id = Column(String(30), nullable=False)
    text = Column(Text, nullable=False)

# Define the Lifestyle class
class Lifestyle(Base):
    __tablename__ = 'lifestyle_docs'

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    doc_id = Column(String(30), nullable=False)
    text = Column(Text, nullable=False)

# Create tables
Base.metadata.create_all(engine)

# Create a session
def create_session():
    Session = sessionmaker(bind=engine)
    return Session()