from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd 
import joblib
from src.StellarClassifier.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Load the saved LabelEncoder
LABEL_ENCODER_PATH = "artifacts/model_trainer/label_encoder.pkl"
label_encoder = joblib.load(LABEL_ENCODER_PATH)

@app.route('/', methods=['GET'])  # Route to display the home page
def homepage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # Route to train the pipeline
def training():
    os.system("python main.py")
    return "Training successful"

@app.route('/predict', methods=['POST', 'GET'])  # Route from web UI
def index():
    if request.method == 'POST':
        try:
            # Get input values from the form
            u = float(request.form['u'])
            g = float(request.form['g'])
            r = float(request.form['r'])
            i = float(request.form['i'])
            z = float(request.form['z'])
            redshift = float(request.form['redshift'])

            # Compute magnitude differences
            u_g = u - g
            g_r = g - r
            r_i = r - i
            i_z = i - z

            # Prepare the input data with derived features
            data = [u_g, g_r, r_i, i_z, redshift]
            data = np.array(data).reshape(1, -1)

            obj = PredictionPipeline()
            predict = obj.predict(data)  # Returns a numerical class

            # Decode the numerical prediction into the original label
            label = label_encoder.inverse_transform([int(predict[0])])[0]

            return render_template('results.html', prediction=label)
        
        except Exception as e: 
            return f"An error occurred: {e}"
        
    else: 
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
