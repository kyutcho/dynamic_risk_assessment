import pandas as pd
import numpy as np
import os
import json
import glob
import itertools
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
ext_formats_list = ["csv"]
ingested_file_list = []
df_list = []
root_dir = os.getcwd()

def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    os.chdir(input_folder_path)
    for ext in ext_formats_list:
        data_list = glob.glob(f"*.{ext}")
        ingested_file_list.append(data_list)
        if ext == "csv":
            for data in data_list:
                df = pd.read_csv(data)
                df_list.append(df)

    # Concatenate all dataframes, dedupe, and save as final dataframe
    final_df = pd.concat(df_list)
    final_df.drop_duplicates(inplace=True)
    os.chdir(root_dir)
    if not os.path.exists(output_folder_path):
        path = os.path.join(root_dir, output_folder_path)
        os.mkdir(output_folder_path)
    final_df.to_csv(os.path.join(output_folder_path,'final_data.csv'), index=False)

    file_list = list(itertools.chain.from_iterable(ingested_file_list))
    with open(os.path.join(output_folder_path, 'ingested_files.txt'), 'w') as f:
        f.write('\n'.join(file_list))


if __name__ == '__main__':
    merge_multiple_dataframe()
