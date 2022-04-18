import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

import seaborn as sns
import statsmodels.api as sm

analysis_name = "Synthesize Ratings"

#Import into pandas
unlabeled_data = pd.read_csv('datasets/dataset_dresses_unlabeled_cv.csv')
survey_data = pd.read_csv('datasets/survey_results_raw.csv')
rated_dresses = pd.read_csv('datasets/rated_dresses.csv')

def rate_items(survey_):




