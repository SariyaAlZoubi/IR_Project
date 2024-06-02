from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Search(BaseModel):
    dataSet: str
    query: str


@app.post('/search')
async def search(request: Search):
    print(request.dataSet)
    return request
