import numpy as np
import pandas as pd
from sklearn import preprocessing

def  convert_list(a):
    x = []
    for i in a:
        x.append(i)

    return x

def encode(a):
    arr = convert_list(a)
    le = preprocessing.LabelEncoder()
    x = le.fit_transform(arr)
    return x
