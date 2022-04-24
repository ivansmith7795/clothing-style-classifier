import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

pd.set_option('display.max_colwidth', -1)

#Import into pandas

survey_data = pd.read_csv('datasets/survey-responses.csv')
rated_dresses = pd.read_csv('datasets/dress-data-normalized.csv')


def normalize_rated_dresses(rated_dresses):
    
    dataset = rated_dresses.copy()

    # Clear empty field special characters and replace with blank values
    dataset = dataset.replace('--', np.NaN, regex=True)

     # Clear empty field special characters and replace with blank values
    dataset = dataset.replace(np.NaN, 'Unknown', regex=True)

    #Drop unused columns
    dataset = dataset.drop(columns=['Article number'])

    #Remove all columns that are all NaN
    dataset = dataset.dropna(how='all', axis=1)

    #Convert the price column to numeric
    dataset['Price'] = dataset['Price'].apply(pd.to_numeric, errors='coerce')

    #Get the columns
    datacols = list(dataset.columns)

    # Convert back from numpy to pandas and add the column headers back on
    dataset = pd.DataFrame(dataset, columns = datacols)

    # Save the unlabeled set to a file
    dataset.to_csv('datasets/dress-data-processed.csv')

    return dataset

def build_matrix(survey_data):
    
    user_data = []

    # Build our ratings matrix to row-wise item ratings
    for index, row in survey_data.iterrows():
        user_id = row['UserID']

        # Loop 20 times per question since we know each user was asked 20 questions about 20 different dresses
        for i in range(1, 20):
            
            # Extract the link for storing in the item matrix (we use the link later on to find and compare items and build our recommendations)
            link = rated_dresses.iloc[[i-1]]['Link'].to_string(header=False, index=False)
           
            # Build our new rating row, which contains the user id, user email address, link of the rated dress, index number and rating 
            rating = [user_id, row['UserEmail'], link, i, row['Dress '+str(i)+'/20. How likely is it that you would wear this dress?'][:1]]
            
            # Apped to the user data list
            user_data.append(rating)

    # Define our columns (as described above)
    columns = ['UserID', 'UserEmail', 'Link', 'DressID', 'Rating']

    user_data = pd.DataFrame (user_data, columns = columns)

     # Save the unlabeled set to a file
    user_data.to_csv('datasets/user_ratings_matrix.csv')

    return user_data

rated_dresss = normalize_rated_dresses(rated_dresses)
user_matrix = build_matrix(survey_data)

#print(user_matrix)