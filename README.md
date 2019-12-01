# Disaster Response Pipeline Project
Analyzing messages data for disaster response.

## Overview of files
The repository contains multiple folders:
* app
  - The web application files. Using JQuery and plotly for javascript and visualization and python Flask for backend.
* data
  - Contains the csv files that are loaded into the database along with the python script containing the ETL pipeline (extracting data from csv's, transforming data to tidy format, loading data to database).
* exploratory
  - Contains jupyter notebook files for proof of concept of both ETL and ML pipeline. Not used actively in the application only as documentation and as a 'playground' for new features.
* models
  - The ML pipeline. Code will train algorithm on data from the database generated from ETL pipeline and output model file in .pkl. Scores (F1, recall, etc.) are printed in console.

## Instructions for setup up web application:
The database and the machine learning model files are not included in this repository (size limitations), so you'll have to create them yourselves. To setup the required files do the following:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database (last argument is the database name)
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves (last argument is the model name)
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

## Libraries & versions
Code uses following libraries and has not been tested on anything else:
* Flask 1.1.1
* Pandas 0.24.2
* Scikit-learn 0.21.2
* Plotly 4.3.0
* SQLAlchemy 1.3.5
