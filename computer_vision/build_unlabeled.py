import pandas as pd
import numpy as np

#Import into pandas
unlabeled_processed_dataset = pd.read_csv('../datasets/dress-dataset-unlabeled-processed.csv')
unlabeled_cv_dataset = pd.read_csv('../datasets/dataset_dresses_unlabeled_cv.csv')

unlabeled = []

for index, row in unlabeled_processed_dataset.iterrows():
    link = row['Link']
    cv_row = unlabeled_cv_dataset[unlabeled_cv_dataset['link'] == link]
    cv_value = 'unknown'
    
    if len(cv_row['cv_predicted']) > 0:
        cv_value = cv_row['cv_predicted'].values[0]
    
    print(cv_value)
    #cv_row['cv_predicted'] = cv_value
    unlabeled.append(cv_value)
    #unlabeled.append(cv_row.values.tolist())

unlabeled_processed_dataset['cv_predicted'] = unlabeled
unlabeled_processed_dataset.to_csv('../datasets/dress-dataset-unlabeled-processed-cv.csv')


