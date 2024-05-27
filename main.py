# This is a sample Python script.
from Data_Representation import Data_Representation
import ir_datasets
from Text_Processing import TextProcessing
import pandas as pd

dataset = ir_datasets.load("lotte/lifestyle/dev/forum")

documents = [doc for _, doc in zip(range(1000), dataset.docs_iter())]
docs = pd.DataFrame(dataset.docs_iter())

text = TextProcessing()
first_100_rows = docs[:100]

data_representation = Data_Representation()

data_representation.data_representation(first_100_rows)


