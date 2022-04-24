import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

#Import into pandas
unlabeled_predictions = pd.read_csv('datasets/unlabeled_naive_bayes_style_preditions.csv')
unlabeled_cv = pd.read_csv('datasets/dress-dataset-unlabeled-processed-cv.csv')
labeled_predictions = pd.read_csv('datasets/dress-dataset-labeled-processed-cv.csv')

predictions = []

for index, row in unlabeled_predictions.iterrows():
    prediction = row['predict']
    predictions.append(prediction)

unlabeled_cv['Style'] = predictions
print(unlabeled_predictions.head())
unlabeled_cv.to_csv('datasets/dress-dataset-unlabeled-processed-nb-predictions.csv')

all_data = labeled_predictions.append(unlabeled_cv)
all_data = all_data.drop(columns=['Unnamed: 0', 'cv_predicted', 'IndexID'])

all_data.to_csv('datasets/dress-dataset-all-processed-nb-predictions.csv')