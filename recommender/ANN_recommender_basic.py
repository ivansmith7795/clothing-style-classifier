import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from sklearn.preprocessing import LabelBinarizer

import seaborn as sns
import statsmodels.api as sm
pd.set_option('display.max_colwidth', -1)

#Import into pandas
user_ratings = pd.read_csv('datasets/user_ratings_matrix.csv')
labeled_dresses = pd.read_csv('datasets/dataset_dresses_labeled.csv')
unlabeled_dresses = pd.read_csv('datasets/dataset_dresses_unlabeled.csv')
survey_responses = pd.read_csv('datasets/survey_results_processed.csv')


def combine_data(labeled_data, unlabled_data):

     # Drop our engineered features
    labeled_data = labeled_data.drop(columns=['Styles'])
    unlabled_data = unlabled_data.drop(columns=['cv_predicted'])

    # Consoliate both list into a single pandas dataframe
    combined_data = labeled_data.append(unlabled_data)

    # Clear empty field special characters and replace with blank values
    combined_data = combined_data.replace('--', '', regex=True)

    #Remove the price special characters and convert to numeric
    combined_data = combined_data.replace(to_replace ='Â£', value = '', regex = True)
    combined_data = combined_data.replace(to_replace ='Â', value = '', regex = True)

    #Convert the price column to numeric
    combined_data['price'] = combined_data['price'].apply(pd.to_numeric, errors='coerce')

    return combined_data

def consolidate_data(combine_data):

    consolidated_data = combine_data.copy()

    #Drop all except Zalando filter columns
    consolidated_data = consolidated_data[['brand', 'price', 'color', 'Outer fabric material', 'Collar', 'Sleeve length', 'Length', 'Pattern', 'Fabric']]

    #Remove all columns that are all NaN
    consolidated_data = consolidated_data.dropna(how='all', axis=1)

    return consolidated_data

def hot_encode(dataset):

    encoded = dataset.copy()

    encoded['brand'] = LabelBinarizer().fit_transform(dataset['brand'])
    encoded['price'] = dataset['price']
    encoded['color'] = LabelBinarizer().fit_transform(dataset.color)
    encoded['Outer fabric material'] = LabelBinarizer().fit_transform(dataset['Outer fabric material'])
    encoded['Collar'] = LabelBinarizer().fit_transform(dataset.Collar)
    encoded['Sleeve length'] = LabelBinarizer().fit_transform(dataset['Sleeve length'])
    encoded['Length'] = LabelBinarizer().fit_transform(dataset.Length)
    encoded['Pattern'] = LabelBinarizer().fit_transform(dataset.Pattern)
    encoded['Fabric'] = LabelBinarizer().fit_transform(dataset.Fabric)

    return encoded

def build_ANN_index(dataset):

    vectors = 9

    ann_index = AnnoyIndex(vectors, 'angular')
    for index, row in dataset.iterrows():
        #print(row)
        ann_index.add_item(index, row.tolist())

    ann_index.build(10)
    ann_index.save('models/basic.ann')

def find_neighbours(vectors, item_index, max_neighbours):
    
    model = AnnoyIndex(vectors, 'angular')
    model.load('models/basic.ann')
    neighbours = model.get_nns_by_item(item_index, max_neighbours)

    return neighbours

def find_item_index(link):
    item_index = combined_data[combined_data['link'] == link].index.values.astype(int)

    return item_index

def find_top_recommendations(user_ratings):

    vectors = 9

    for index, row in survey_responses.iterrows():
        user = row['UserID']

        single_user_ratings = user_ratings[user_ratings['UserID'] == user]
        single_user_ratings = single_user_ratings.sort_values('Rating', ascending=False)

        # Get only the top 2
        single_user_ratings = single_user_ratings.head(2)

        recommendations_index = []

        # Get nearest items for the top 2 rated items
        for rating_index, rating_row in single_user_ratings.iterrows():
            item_index = find_item_index(rating_row['Link'][1:])
            neighbors = find_neighbours(vectors, int(item_index[0]), 5)

            for item in neighbors:
                recommendations_index.append(item)
        
        recommendations = []
        
        print(recommendations_index)
        for index in recommendations_index:
            recommend_row = combined_data.iloc[index,:]
            recommendations.append(recommend_row)
        
        recommendations = pd.DataFrame(recommendations, columns = combined_data.columns)
        user_email = row['UserEmail']

        recommendations['User']=user_email

        # Save the user recommendations to a file
        recommendations.to_csv('results/'+ str(user) +'_' + user_email + '_recommendations.csv')



combined_data = combine_data(labeled_dresses, unlabeled_dresses)
consolidated_data = consolidate_data(combined_data)
vectorized_data = hot_encode(consolidated_data)
build_ANN_index(vectorized_data)
recommendations = find_top_recommendations(user_ratings)
