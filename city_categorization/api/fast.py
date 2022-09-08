from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


class ImgTensor(BaseModel):
    X: list


app = FastAPI()

app.state.model = load_model('../../model')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/test")
async def create_image(img: ImgTensor):
    x = np.array(img.X)
    y_pred = str(app.state.model.predict(x))
    return {'prediction': y_pred}


@app.get("/")
def index():
    return {'greeting': 'Hi'}
