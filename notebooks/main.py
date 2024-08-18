from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pickle as pk
import uvicorn


app = FastAPI()

class postRequest(BaseModel):
    features: list[float]


class postReponse(BaseModel):
    prediction: float


# Evento OnStart - Carrega modelo
@app.on_event("startup")
async def startup_even():
    global model
    with open("/app/notebooks/modelo_ex21a.pkl", "rb") as f:
        model = pk.load(f)


@app.post("/predict", response_model=postReponse)
async def predict(file: UploadFile):
    return {"file_size": len(file)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)