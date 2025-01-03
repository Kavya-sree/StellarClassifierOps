# StellarClassifierOps

This project focuses on building a machine learning pipeline to classify stars based on their spectral characteristics. The goal is to develop a robust workflow using MLOps principles to manage data ingestion, data transformation, model training and evaluation.

## Project Overview

This project focuses on classifying celestial objects—stars, galaxies, and quasars—using data from the Sloan Digital Sky Survey (SDSS) DR17. The classification leverages machine learning techniques to analyze and categorize spectral and photometric data, contributing to a deeper understanding of the universe's structure and evolution.

## Background Info

### What is Stellar Classification?**

Stellar classification involves categorizing stars, galaxies, and quasars based on their spectral properties and physical characteristics. This helps in studying their composition, lifecycle, and role in cosmic evolution.

**Astronomy** is the scientific study of the universe and of the objects that exist naturally in space.

**Stars, Quasars, and Galaxies**
The three important stellar objects in astronomy are stars, galaxies, and quasars. They are the fundamental building blocks of the universe.
* Stars are massive luminous spheroids of plasma undergoing nuclear fusion and this process generates enormous amounts of energy, making them shine. They come in various types, sizes, and life cycles, from humble red dwarfs to explosive supernovae.
* Galaxies are sprawling systems of stars, gas, dust, and dark matter bound together by gravity. They range from elegant spiral shapes, like our Milky Way, to colossal elliptical galaxies and irregular formations. Each galaxy is a cosmic metropolis housing millions to trillions of stars.
* Quasars or Quasi-Stellar-Objects (QSO) are at the center of some distant galaxies. They derive their name from their initial starlike appearance when discovered in the late 1950s and early 1960s. It is a supermassive black hole that is growing rapidly by gorging on huge amounts of gas and they are extremely luminous.
 
**About the SDSS DR17 Dataset**

The Sloan Digital Sky Survey (SDSS) is a comprehensive astronomical survey that provides high-quality spectra and imaging for millions of celestial objects. DR17 is its latest release, offering improved data reduction techniques and expanded catalogs for stars, galaxies, and quasars.

## ML Pipeline

1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Trainer
5. Model Evaluation

## Workflows
1. update config.yaml
2. update schema.yaml
3. update params.yaml
4. update the entity
5. update the configuration manager in src config
6. update the components
7. update the pipeline
8. update the main.py
9. update the app.py


## Model Evaluation

The model's performance is evaluated using key metrics to ensure its effectiveness in classifying stars, galaxies, and quasars. The metrics tracked include:

- Accuracy: 96.7%
- Precision: 96.7%
- Recall: 96.7%
- F1 Score: 96.6%


## Installation

1. Clone the repository:

```bash
https://github.com/Kavya-sree/StellarClassifierOps.git
```

2. Navigate to the project directory:

```bash
cd StellarClassifierOps
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to run

To train the model, use the following command:

```bash
python main.py
```

To run the app:

```bash
python app.py
```

## Experiment Tracking with Dagshub and MLflow

This project integrates DagsHub and MLflow to track training experiments and manage model versions:

1. Set up DagsHub Repository

2. Secure configuration with `.env`

This project uses a `.env` file to securely manage MLflow tracking details
* Create a `.env` file in the projects root directory.
* Add the following tracking details in the `.env` file, replacing the placeholders with your actual values:

```python
.env
#  MLflow Details
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/<your-username>/<repo-name>.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "your-username"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your password here"
```
* Ensure the `.env` file is included in `.gitignore` to prevent sensitive information from being commited to version control.

4. Run the Project and log metrics:

```bash
python main.py
```
This script log metrics like accuracy, precision, recall, and F1 score to MLflow.

5. To visualize experiments locally, use the MLflow UI:

```bash
mlflow ui
```