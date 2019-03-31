import pandas as pd
import numpy as np 
import operator
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pickle

with open('leaf/trained_model','rb') as f:
    model = pickle.load(f)


# data = 