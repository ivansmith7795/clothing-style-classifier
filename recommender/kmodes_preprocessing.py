import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

import seaborn as sns
import statsmodels.api as sm

analysis_name = "Survey Preprocessing"

#Import into pandas
survey_data = pd.read_csv('datasets/survey_results_raw.csv')

def get_budget(data):
    averages = []
    for index, row in data.iterrows():
        spends = [int(n) for n in row['What budget do you typically spend on dresses?'].split(",")]
        items = len(spends)
        sum_spends = sum(spends)
        mean = sum_spends / items
        averages.append(round(mean))
    
    data['What budget do you typically spend on dresses?'] = averages
    return data

def clean_data(dataset):
    
    # Clear empty field special characters and replace with blank values
    #dataset = dataset.replace('--', np.NaN, regex=True)

    #Remove all columns that are all NaN
    dataset = dataset.dropna(how='all', axis=1)
    datacols = list(dataset.columns)

    #Remove the price special characters from the spend column
    
    dataset['What budget do you typically spend on dresses?'] = dataset['What budget do you typically spend on dresses?'].str.replace("[â€š<>Ã¢â€šÂ¬]", "")
    dataset['What budget do you typically spend on dresses?'] = dataset['What budget do you typically spend on dresses?'].str.replace("[‚‚,,-]", " ")
    dataset['What budget do you typically spend on dresses?'] = dataset['What budget do you typically spend on dresses?'].str.replace("  ", " ")
    dataset['What budget do you typically spend on dresses?'] = dataset['What budget do you typically spend on dresses?'].str.replace("I would rather not say", " 40")
    dataset['What budget do you typically spend on dresses?'] = dataset['What budget do you typically spend on dresses?'].str[1:]
    dataset['What budget do you typically spend on dresses?'] = dataset['What budget do you typically spend on dresses?'].str.replace(" ", ",")
    dataset['What budget do you typically spend on dresses?'] = dataset['What budget do you typically spend on dresses?'].str.replace(",,", ",")

    dataset = get_budget(dataset)

    dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].str.replace(".", "")
    dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].str.replace(",", "")
    dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].str.replace("cm", "")
    dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].str.replace("m", "")
    dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].str.replace(" ", "")

    dataset['What styles do you identify most with? Limit yourself to picking three looks'] = dataset['What styles do you identify most with? Limit yourself to picking three looks'].str.replace("( -).*","")

    #dataset['What budget do you typically spend on dresses?'] = dataset['What budget do you typically spend on dresses?'].apply(lambda x: mean(map(int, x.split(','))))
    imputer = SimpleImputer(strategy="most_frequent")
    #dataset = imputer.fit_transform(dataset)

    # Lets get the mean price of each user
    #
    
    

    #Normalize the height column
    #dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].replace(to_replace ='cm', value = '', regex = True)
    #dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].replace(to_replace ='m', value = '', regex = True)
    #dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].replace(to_replace =',', value = '', regex = True)
    #dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].replace(to_replace ='.', value = '', regex = True)
    #dataset['What is your height (+-)?'] = dataset['What is your height (+-)?'].replace(to_replace ='vm', value = '', regex = True)

    #Convert the price column to numeric
    #dataset['price'] = dataset['price'].apply(pd.to_numeric, errors='coerce')

    #Get the columns
    
    dataset = pd.DataFrame(dataset, columns = datacols)

    # Save the unlabeled set to a file
    dataset.to_csv('datasets/survey_results_processed.csv')


clean_data(survey_data)