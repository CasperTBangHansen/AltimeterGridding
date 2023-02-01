from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()



@app.get("/input/{lon}/{lat}/{time}")
async def gridding(lon: float, lat: float, time: float) -> list:
    minutes = time - int(time)
    minutes = minutes * 0.6
    time = int(time) + minutes
    result = [lon,lat,time]
    return result

@app.get("/items/{item_id}")
async def read_item(item_id: int | str):
    results = {"item_id": item_id}
    return results