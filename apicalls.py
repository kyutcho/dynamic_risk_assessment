import os
import requests
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

# print(f"{URL}/scoring")

#Call each API endpoint and store the responses
response1 = requests.post(url=f"{URL}/prediction", json={"data_name": os.path.join(test_data_path, "test_data.csv")}).content
response2 = requests.get(url=f"{URL}/scoring").content
response3 = requests.get(url=f"{URL}/summarystats").content
response4 = requests.get(url=f"{URL}/diagnostics").content

# #combine all API responses
responses = f'''
    response1: {response1}
    response2: {response2}
    response3: {response3}
    response4: {response4}
    '''

#write the responses to your workspace
with open(os.path.join(model_path, "api_returns.txt"), "w") as f:
    f.write(responses)
