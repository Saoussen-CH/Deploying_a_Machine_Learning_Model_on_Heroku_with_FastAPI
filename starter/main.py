# Put the code for your API here.
import os
import yaml
import pandas as pd
import uvicorn
import pickle

from fastapi import FastAPI
from pydantic import BaseModel


from starter.ml.model import *
from starter.ml.data import *

# FastAPI instance
app = FastAPI()

# Heroku access to DVC data
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")



class Input(BaseModel):
    age : int = 23
    workclass : str = 'Self-emp-inc'
    fnlgt : int = 76516
    education : str = 'Bachelors'
    education_num : int = 13
    marital_status : str ='Married-civ-spouse' 
    occupation : str = 'Exec-managerial'
    relationship : str = 'Husband'
    race : str = 'White'
    sex : str = 'Male'
    capital_gain : int = 0
    capital_loss : int = 0
    hours_per_week : int = 40
    native_country : str = 'United States'

class Output(BaseModel):
    prediction:str

@app.get("/")
def welcome():
    return "Hi, Welcome to Census API"

# Load Pickle files: Model, Encoder and LabelBinarizer   
#@app.on_event("startup")
#async def startup_event(): 
#    global model, encoder, binarizer
model = pickle.load(open("starter/model/model.pkl", "rb"))
encoder = pickle.load(open("starter/model/OneHotEnoder.pkl", "rb"))
binarizer = pickle.load(open("starter/model/LabelBinarizer.pkl", "rb"))

@app.post("/predict", response_model=Output, status_code=200)
def predict(data: Input):

    # Categorical features for transform model
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

    # load predict_data
    request_dict = data.dict(by_alias=True)
    request_data = pd.DataFrame(request_dict, index=[0])
    
    X, _, _, _ = process_data(
                request_data,
                categorical_features=cat_features,
                training=False,
                encoder=encoder,
                lb=binarizer)

    prediction = model.predict(X)
    
    if prediction[0] == 1:
        prediction = "Salary > 50k"
    else:
        prediction = "Salary <= 50k"
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)