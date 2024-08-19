from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pickle as pk
import uvicorn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


app = FastAPI()

class postRequest(BaseModel):
    features: list[float]


class postReponse(BaseModel):
    prediction: float

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# Evento OnStart - Carrega modelo
@app.on_event("startup")
async def startup_even():
    global model
    with open("/app/notebooks/modelo_ex21a.pkl", "rb") as f:
        model = pk.load(f)


@app.post("/predict", response_model=postReponse)
async def predict(file: UploadFile):

    imagem = await file.read()
    imagem = imagem.resize((8,8))

    


    return {"file_size": len(file)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)