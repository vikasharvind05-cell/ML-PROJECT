import joblib
import pandas as pd
from pyexpat import model

from train import pipeline


def predict(data: dict):
    df= pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = pipeline.predict_proba(df)[0].tolist()
    return{
      "prediction":int(prediction),
        "probability": {
            "No Survival": probability[0],
            "Survival": probability[1]
        }
    }

