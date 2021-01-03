import uvicorn
from fastapi import FastAPI
from banknote import BankNote
import numpy as np 
import pandas as pd 
import pickle

app = FastAPI(debug = True)
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.get('/')
async def index():
    return {'message': 'Hello, World'}

@app.get('/{name}')
async def get_name(name:str):
    return {'message': f'Hello, {name}'}

@app.post('/predict')
async def predict_banknote(data: BankNote):
    data = data.dict()
    print(data)
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    #print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if(prediction[0]>0.5):
        prediction = 'Fake Note'
    else:
        prediction = 'Its a Bank Note'
    return {'prediction': prediction}


if __name__ == '__main__':
    icorn.run(app, host = '127.0.0.1', port = 8000)