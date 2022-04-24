import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

import seaborn as sns
import statsmodels.api as sm

analysis_name = "Clothing Type Classifier"

#Import into pandas
labeled_dataset = pd.read_csv('datasets/dress-dataset-labeled-processed-cv.csv')
unlabeled_dataset = pd.read_csv('datasets/dress-dataset-unlabeled-processed-cv.csv')

def clean_data(dataset, datatype):
    
    # Clear empty field special characters and replace with blank values
    #dataset = dataset.replace('--', np.NaN, regex=True)

    #Drop unused columns
    dataset = dataset.drop(columns=['IndexID', 'Link'])

    #Remove all columns that are all NaN
    dataset = dataset.dropna(how='all', axis=1)

    #Remove the price special characters and convert to numeric
    #dataset = dataset.replace(to_replace ='Â£', value = '', regex = True)
    #dataset = dataset.replace(to_replace ='Â', value = '', regex = True)

    #Convert the price column to numeric
    dataset['Price'] = dataset['Price'].apply(pd.to_numeric, errors='coerce')

    #Get the columns
    datacols = list(dataset.columns)

    # Only impute for the labeled training data (we will use the unlabeled data for prediction)
    if datatype == 'labeled':

        #Impute our missing numeric / categorical data (there are lots!)
        imputer = SimpleImputer(strategy="most_frequent")
        dataset = imputer.fit_transform(dataset)

        # Convert back from numpy to pandas and add the column headers back on
        dataset = pd.DataFrame(dataset, columns = datacols)

    # Save the unlabeled set to a file
    dataset.to_csv('datasets/dataset-dresses-' + datatype + '-processed-bayes.csv')


clean_data(labeled_dataset, 'labeled')
clean_data(unlabeled_dataset, 'unlabeled')