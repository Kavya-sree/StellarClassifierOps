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
            # Parse input values
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

            # Prepare input data with raw photometric values and derived features
            data = [u, g, r, i, z, u_g, g_r, r_i, i_z, redshift]
            data = np.array(data).reshape(1, -1)

            obj = PredictionPipeline()
            predict = obj.predict(data)  # Numerical class prediction

            # Decode prediction
            label = label_encoder.inverse_transform([int(predict[0])])[0]

            return render_template('results.html', prediction=label)

        except ValueError as e:
            return f"Invalid input values: {e}", 400
        except Exception as e:
            return f"An unexpected error occurred: {e}", 500

    return render_template('index.html')

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
