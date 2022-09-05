import os
import json
import glob

from ingestion import *
from training import *
from scoring import *
from deployment import *
from diagnostics import *
from reporting import *

with open('config.json', 'r') as f:
    config = json.load(f)

ingestion_file_path = os.path.join(config["prod_deployment_path"])
data_csv_path = os.path.join(config["input_folder_path"])

##################Check and read new data
#first, read ingestedfiles.txt
def check_new_files():
    curr_data_lst = []
    with open(os.path.join(ingestion_file_path, "ingested_files.txt")) as f:
        for line in f:
            curr_data_lst.append(line.strip())

    # Check new data in input_folder_path
    os.chdir(data_csv_path)
    new_data_lst = glob.glob(f"*.csv")
    os.chdir('..')

    # Determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    to_ingest_data_lst = set(new_data_lst).difference(set(curr_data_lst))

    # If you found new data, you should proceed. otherwise, do end the process here
    if len(to_ingest_data_lst) != 0:
        merge_multiple_dataframe()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(ingestion_file_path, 'latest_score.txt'), 'r') as f:
    old_f1_score = float(f.read())

train_model()

new_f1_score = score_model(model, data)


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







