# 1. Library imports
import uvicorn
from fastapi import FastAPI
from ClassifyNews import ClassifyNews
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("models\model.pickle","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Neels First FastAPI': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/classify')
def classify_news(data:ClassifyNews):
    data = data.dict()
    news = data['newsSummary']

   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([news])
    return {
        'prediction': prediction[0]
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)