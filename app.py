from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import *
from scoring import *



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
test_df = pd.read_csv(os.path.join(test_data_path, "test_data.csv"))

model_path = os.path.join(config['output_model_path'])
with open(os.path.join(model_path, 'trained_model.pkl'), 'rb') as f:
        lr = pickle.load(f)


#######################Home
@app.route("/", methods=["GET"])
def welcome_page():
    return {"Welcome": "Hello World"}

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST'])
def predict():
    data_name = request.files.get('data_name')
    prediction = model_predictions(data_name)

    return jsonify({"Prediction": str(list(prediction))})

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    # lr = request.args.get('model_name')
    # data = request.args.get('data_name')
    f1_score = score_model(lr, test_df)
    return str(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    data_summary = dataframe_summary(test_df)
    return str(data_summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def timing():        
    time = execution_time()
    na_pct = dataframe_missing_data(test_df)
    outdated = outdated_packages_list()

    output = f"""
                Time: {str(time)} <br> 
                % of missing: {str(na_pct)} <br>
                Outdated dependencies: <br>
                {outdated}
              """

    return output #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
