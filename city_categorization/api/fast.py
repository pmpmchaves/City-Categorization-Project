from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import pandas as pd


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


@app.post("/predict")
async def create_image(img: ImgTensor):
    X_preprocessed = np.array(img.X)
    prediction = app.state.model.predict(X_preprocessed)
    y_pred_cat = np.argmax(prediction, axis=1)
    y_pred_cat_reduced = pd.Series(y_pred_cat).map(lambda x: 0 if x == 1 else (
        1 if x == 0 or x == 2 or x == 5 or x == 6 else (2 if x == 3 else (
            3 if x == 4 else (4 if x == 7 else 5))))).to_numpy()
    y_pred_list = list(y_pred_cat_reduced)
    return {'prediction': str(y_pred_list)}


@app.get("/")
def index():
    return {'greeting': 'Hi'}
