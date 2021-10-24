import uvicorn 
from fastapi import FastAPI, Request, requests
import joblib

import pandas as pd
model = joblib.load('model_ref.pkl')

from pydantic import BaseModel
app = FastAPI()

class HumanActivity(BaseModel):
    Ax : float
    Ay : float
    Az : float
    Gx : float
    Gy : float
    Gz : float

@app.get('/')
def index():
    return {'message': 'Welcome to Human Activity Recognition'}

@app.get('/Welcome')
def get_name(name : str):
    return {'Welcome': f'{name}'}

@app.post('/predict_activity')
async def predict_activity(humanactivity : HumanActivity):
    data = pd.DataFrame(humanactivity.dict(), index=[0])
    prediction = model.predict(data)[0]
    output_label = {
        1:'Drinking',2:'Crunchy Food',
        3:'Soft Food',4:'Speaking'}

    output_prediction = output_label.get(prediction)
    
    return {'output_prediction': output_prediction}
    


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)