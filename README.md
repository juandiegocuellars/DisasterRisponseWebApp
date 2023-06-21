# Disaster Response Pipeline Project

## Project Summary

This project is made with Figure Eight to develop a web app where an emergency worker can input a new message and get classification results in several categories. This was possible using ETL pipelines and Machine Learning Pipelines to train a supoervised model. 

The objective of this web app is to filter all types of messages during an emergency, and to catalog them for the organizations which can help the people with their urgency. This is a valuable development to save lives and to help the people which needs the most during an emergency.

## Project Components

### 1. ETL pipeline (data folder):
ETL was made on process_data.py which is a pyhton script that makes:
* Loads messages and categories csv files.
* Merge both files.
* Cleans data (outliers, duplicates, etc.)
* Stores in a SQLite database.

### 2. Machine Learning Pipeline (models pipeline):
Pipeline was made on train_classifier.py which is a Python script that makes:
* Loads data from SQLite database.
* Split the data into train and test set.
* Generate a pipeline which proccess text and uses a RandomForestClassifier.
* Trains and Tunes the model using Grid Search.
* Output the result of the test on F1 Score, Precision and Recall.
* Export the model as pickle file.

### 3. Flask Web App (app folder)
This web app was made using python to run.py script to deploy the web app. Also uses HTML scripts and uses the pickle file. This code was made by Udacity.

## Instructions for Web App Execution:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
