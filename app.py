from fastapi import FastAPI
from pydantic import BaseModel

from main import Main

app = FastAPI()
main = Main()


class Search(BaseModel):
    dataSet: str
    query: str
    wordEmbedding: bool


@app.post('/search')
async def search(request: Search):
    a = main.run(request)
    return a
