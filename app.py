from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd 
import joblib
from src.StellarClassifier.pipeline.prediction_pipeline import PredictionPipeline

app=Flask(__name__)

# Load the saved LabelEncoder
LABEL_ENCODER_PATH = "artifacts/model_trainer/label_encoder.pkl"
label_encoder = joblib.load(LABEL_ENCODER_PATH)

@app.route('/',methods=['GET']) # route to display the home page
def homepage():
    return render_template("index.html")

@app.route('/train',methods=['Get']) # route to train the pipeline
def training():
    os.system("python main.py")
    return  "Training successful"

@app.route('/predict', methods=['POST', 'GET']) # route from web ui
def index():
    if request.method == 'POST':
        try:
            u = float(request.form['u'])
            g = float(request.form['g'])
            r = float(request.form['r'])
            i = float(request.form['i'])
            z = float(request.form['z'])
            redshift = float(request.form['redshift'])

            data=[u,g,r,i,z,redshift]
            data = np.array(data).reshape(1,6)

            obj=PredictionPipeline()
            predict=obj.predict(data) # returns a numerical class

            # Decode the numerical prediction into the original label
            label = label_encoder.inverse_transform([int(predict[0])])[0]

            return render_template('results.html', prediction=label)
        
        except Exception as e: 
            return "Something is wrong"
        
    else: 
        return render_template('index.html')
    
if __name__=="__main__":

    app.run(host="0.0.0.0", port=8080)
