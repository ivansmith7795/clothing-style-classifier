import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

import seaborn as sns
import statsmodels.api as sm
pd.set_option('display.max_colwidth', -1)

#Import into pandas

survey_data = pd.read_csv('datasets/survey_results_processed.csv')
rated_dresses = pd.read_csv('datasets/rated_dresses.csv')


def normalize_rated_dresses(rated_dresses):
    
    dataset = rated_dresses.copy()

    # Clear empty field special characters and replace with blank values
    dataset = dataset.replace('--', '', regex=True)

    #Drop unused columns
    dataset = dataset.drop(columns=['Article number'])

    #Remove all columns that are all NaN
    dataset = dataset.dropna(how='all', axis=1)

    #Convert the price column to numeric
    dataset['price'] = dataset['price'].apply(pd.to_numeric, errors='coerce')

    #Get the columns
    datacols = list(dataset.columns)

    # Convert back from numpy to pandas and add the column headers back on
    dataset = pd.DataFrame(dataset, columns = datacols)

    # Save the unlabeled set to a file
    dataset.to_csv('datasets/rated_dresses_processed.csv')

    return dataset

def build_matrix(survey_data):
    
    user_data = []

    # Build our ratings matrix to row-wise item ratings
    for index, row in survey_data.iterrows():
        user_id = row['UserID']

        for i in range(1, 20):
            link = rated_dresses.iloc[[i-1]]['link'].to_string(header=False, index=False)
           
            rating = [user_id, row['UserEmail'], link, i, row['Dress '+str(i)+'/20. How likely is it that you would wear this dress?'][:1]]
            user_data.append(rating)

    columns = ['UserID', 'UserEmail', 'Link', 'DressID', 'Rating']

    user_data = pd.DataFrame (user_data, columns = columns)

     # Save the unlabeled set to a file
    user_data.to_csv('datasets/user_ratings_matrix.csv')

    return user_data

rated_dresss = normalize_rated_dresses(rated_dresses)
user_matrix = build_matrix(survey_data)

#print(user_matrix)