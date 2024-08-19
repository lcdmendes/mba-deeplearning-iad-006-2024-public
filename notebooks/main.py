from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle as pk
import uvicorn
from PIL import Image, ImageOps 
import numpy as np
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
    global model_random_forest
    with open("/app/notebooks/model_random_forest.pkl", "rb") as f:
        model_random_forest = pk.load(f)

    global model_xgboost
    with open("/app/notebooks/model_xgboost.pkl", "rb") as f:
        model_xgboost = pk.load(f)


@app.post("/predict", response_model=postReponse)
async def predict(file: UploadFile = File(...)):

    if file.content_type != "image/png":
        return JSONResponse(status_code=400, content={"message": "Tipo de arquivo inválidos. Aceita apenas arquivos PNG."})

    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((8,8))
    img_gray = np.array(ImageOps.grayscale(img) ) * (16/255)
    img_gray = img_gray.astype(int).astype(float)

    vec = img_gray.reshape(1,-1)

    predict = model_random_forest.predict(vec)

    return {"prediction": predict.item()}


@app.get("/check")
async def check():
    return {"status" : "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)