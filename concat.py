from pandas import Series,DataFrame
import pandas as pd
import numpy as np
DATASET_NAME='data.csv'
origin=pd.read_csv(DATASET_NAME)
for i in range(0,999):
    print(i)
    origin['index']+=origin['index'].count()
    origin.to_csv('data.csv',mode='a',header=None,index=False)
