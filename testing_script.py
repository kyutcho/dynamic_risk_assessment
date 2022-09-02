import os
import json
import glob
import numpy as np
import pandas as pd

# with open('config.json', 'r') as f:
#     config = json.load(f)
    
# os.chdir(config["input_folder_path"])

# ext_format = "csv"
# result = glob.glob(f"*.{ext_format}")

# print(result)

df = pd.read_csv("practice_data/dataset1.csv")

df2 = df.describe(include=np.number).loc[["mean", "50%", "std"], :]

print(df2.to_dict(orient='list'))

# print(round((df.isnull().sum() / len(df) * 100), 2))