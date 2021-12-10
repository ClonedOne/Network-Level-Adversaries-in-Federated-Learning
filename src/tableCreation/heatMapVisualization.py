import json 
import pandas as pd

with open("heatMap_data.json","r") as f:
    data = json.load(f)

for k,v in data.items():
    print(k)
    df = pd.read_json(v)
    print(df)
    print("-"*100)