import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

#Import into pandas
all_dresses = pd.read_csv('datasets/dress-dataset-all-raw.csv')
labeled_dresses = all_dresses[all_dresses['Style'].notna()]
unlabeled_dresses = all_dresses[all_dresses['Style'].isnull()]

print(len(labeled_dresses))
print(len(unlabeled_dresses))


def normalize(dataset, datatype):
    
    # Clear empty field special characters and replace with blank values
    dataset = dataset.replace(r'^--', np.NaN, regex=True)

     # Clear empty field special characters and replace with blank values
    #dataset = dataset.replace(np.NaN, 'Unknown', regex=True)

    #Drop unused columns
    dataset = dataset.drop(columns=['Article number'])

    #Remove all columns that are all NaN
    #dataset = dataset.dropna(how='all', axis=1)

    #Convert the price column to numeric
    dataset['Price'] = dataset['Price'].apply(pd.to_numeric, errors='coerce')

    #Get the columns
    datacols = list(dataset.columns)

    # Convert back from numpy to pandas and add the column headers back on
    dataset = pd.DataFrame(dataset, columns = datacols)

    # Save the unlabeled set to a file
    dataset.to_csv('datasets/dress-dataset-'+datatype+'-processed.csv')

    return dataset


normalize(labeled_dresses, 'labeled')
normalize(unlabeled_dresses, 'unlabeled')