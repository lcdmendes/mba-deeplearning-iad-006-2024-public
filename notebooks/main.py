from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle as pk
import uvicorn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io


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
async def predict(file: UploadFile = File(...)):

    if file.content_type != "image/png":
        return JSONResponse(status_code=400, content={"message": "Tipo de arquivo inválidos.Aceita apenas arquivos PNG."})

    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = np.array(img)

    img_gray = (np.array(rgb2gray(img)) * 17).astype(int).astype(float)
    vec = img_gray.reshape(1,-1)

    predict = model.predict(vec)

    return {"prediction": predict.item()}
    #return {"prediction": predict}


@app.get("/check")
async def check():
    return {"status" : "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)