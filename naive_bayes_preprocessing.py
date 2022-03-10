import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from matplotlib.pyplot import figure
from sklearn.impute import SimpleImputer

import seaborn as sns
import statsmodels.api as sm

analysis_name = "Clothing Type Classifier"

#Import into pandas
dataset = pd.read_csv('datasets/dataset_dresses_unlabeled.csv')

#Remove all columns that are all NaN
dataset = dataset.dropna(how='all', axis=1)

#Impute our missing categoricak data (there are lots!)
#imputer = SimpleImputer(strategy="most_frequent")
#dataset = imputer.fit_transform(dataset)

print(dataset.head())

dataset.to_csv('datasets/dataset_dresses_unlabeled_processed.csv')