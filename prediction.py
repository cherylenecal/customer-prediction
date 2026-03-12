import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel
from xgboost import XGBClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
# XGBmodel = open("XGBnSMOTE.pkl", "rb")
# classifier = pickle.load(XGBmodel)
try:
    with open("XGBnSMOTE.pkl", "rb") as model_file:
        classifier = pickle.load(model_file)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")

#encoders
try:
    with open("one_hot.pkl", "rb") as onehot_file:
        one_hot_encoder = pickle.load(onehot_file)
    logger.info("One-hot encoder loaded successfully")
except Exception as e:
    logger.error(f"Error loading one-hot encoder: {e}")

try:
    with open("binary.pkl", "rb") as binary_file:
        label_encoder = pickle.load(binary_file)
    logger.info("Binary encoder loaded successfully")
except Exception as e:
    logger.error(f"Error loading binary encoder: {e}")

try:
    with open("ordinal.pkl", "rb") as ordinal_file:
        ordinal_encoder = pickle.load(ordinal_file)
    logger.info("Ordinal encoder loaded successfully")
except Exception as e:
    logger.error(f"Error loading ordinal encoder: {e}")


class UASMD(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.get("/")
def success():
    return {"message": "Halo Bu Lili <3"}

@app.post("/predict")

def predict(main: UASMD):
    try:
        data = main.dict()
        logger.info(f"Received data: {data}")
        
        # age = data['age']
        # job = data['job']
        # marital = data['marital']
        # education = data['education']
        # default = data['default']
        # housing = data['housing']
        # loan = data['loan']
        # contact = data['contact']
        # month = data['month']
        # day_of_week = data['day_of_week']
        # duration = data['duration']
        # campaign = data['campaign']
        # pdays = data['pdays']
        # previous = data['previous']
        # poutcome = data['poutcome']

        #encode
        one_hot_features = np.array([
            [data['job'], data['marital'], data['education'], data['contact'], data['poutcome']]
        ])
        one_hot_encoded = one_hot_encoder.transform(one_hot_features)
        logger.info(f"One-hot encoded data: {one_hot_encoded}")

        binary_features = np.array([
            [data['housing']],
            [data['loan']]
        ]).flatten()
        binary_encoded = label_encoder.transform(binary_features)
        logger.info(f"Binary encoded data: {binary_encoded}")

        ordinal_features = np.array([
            [data['month'], data['day_of_week']]
        ])
        ordinal_encoded = ordinal_encoder.transform(ordinal_features)
        logger.info(f"Ordinal encoded data: {ordinal_encoded}")

        #input data ke model
        input_data = np.hstack([
            [data['age']],
            one_hot_encoded.flatten(),
            binary_encoded,
            ordinal_encoded.flatten(),
            [data['duration'], data['campaign'], data['pdays'], data['previous']]
        ])
        logger.info(f"Input data for model: {input_data}")

        prediction = classifier.predict(input_data.reshape(1, -1))
        logger.info(f"Model prediction: {prediction}")
        #convert prediction to 'yes' or 'no'
        prediction_label = 'yes' if prediction[0] == 1 else 'no'

        return {'prediction': prediction_label}
    except Exception as e:
        logger.error(f"Error in prediction: {e}")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.0', port=8000)