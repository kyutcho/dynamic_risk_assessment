from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path']) 

####################function for deployment
def store_model_into_pickle(model):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    if not os.path.exists(prod_deployment_path):
        os.mkdir(prod_deployment_path)

    # Copy model
    src = os.path.join(model_path, "trained_model.pkl")
    dst = os.path.join(prod_deployment_path, "trained_model.pkl")
    shutil.copy2(src, dst)
    
    # Copy model score
    src = os.path.join(model_path, 'latest_score.txt')
    dst = os.path.join(prod_deployment_path, 'latest_score.txt')
    shutil.copy2(src, dst)

    # Copy ingest file list
    src = os.path.join(dataset_csv_path, 'ingested_files.txt')
    dst = os.path.join(prod_deployment_path, 'ingested_files.txt')
    shutil.copy2(src, dst)
    

if __name__ == '__main__':
    with open(os.path.join(model_path, "trained_model.pkl"), 'rb') as f:
        lr = pickle.load(f)

    store_model_into_pickle(lr)