import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from sklearn.preprocessing import LabelBinarizer

pd.set_option('display.max_colwidth', -1)

#Import into pandas
user_ratings = pd.read_csv('datasets/user_ratings_matrix.csv')
dress_data = pd.read_csv('../datasets/dress-dataset-all-processed-nb-predictions.csv')
survey_responses = pd.read_csv('datasets/survey_results_processed.csv')


def process_data (dress_data):

     # Drop our engineered features
    dress_data = dress_data.drop(columns=['Style'])

    # Clear empty field special characters and replace with blank values
    dress_data = dress_data.replace('--', '', regex=True)

    # Clear empty field special characters and replace with blank values
    #dress_data = dress_data.replace(np.NaN, 'Unknown', regex=True)
    dress_data = dress_data.fillna('Unknown')

    #Remove the price special characters and convert to numeric
    dress_data = dress_data.replace(to_replace ='Â£', value = '', regex = True)
    dress_data = dress_data.replace(to_replace ='Â', value = '', regex = True)

    #Convert the price column to numeric
    dress_data['Price'] = dress_data['Price'].apply(pd.to_numeric, errors='coerce')

    return dress_data



def consolidate_data(combine_data):

    consolidated_data = combine_data.copy()

    #Drop all except Zalando filter columns
    consolidated_data = consolidated_data[['Brand', 'Price', 'Color', 'Outer fabric material', 'Collar', 'Sleeve length', 'Dress length', 'Consolidated Prints', 'Fabric']]

    #Remove all columns that are all NaN
    consolidated_data = consolidated_data.dropna(how='all', axis=1)

    return consolidated_data

def hot_encode(dataset):

    encoded = dataset.copy()

    # Encode our basic features into numeric representations
    encoded['Brand'] = LabelBinarizer().fit_transform(dataset['Brand'])
    encoded['Price'] = dataset['Price']
    encoded['Color'] = LabelBinarizer().fit_transform(dataset['Color'])
    encoded['Outer fabric material'] = LabelBinarizer().fit_transform(dataset['Outer fabric material'])
    encoded['Collar'] = LabelBinarizer().fit_transform(dataset['Collar'])
    encoded['Sleeve length'] = LabelBinarizer().fit_transform(dataset['Sleeve length'])
    encoded['Dress length'] = LabelBinarizer().fit_transform(dataset['Dress length'])
    encoded['Consolidated Prints'] = LabelBinarizer().fit_transform(dataset['Consolidated Prints'])
    encoded['Fabric'] = LabelBinarizer().fit_transform(dataset['Fabric'])

    return encoded

def build_ANN_index(dataset):

    vectors = 9

    ann_index = AnnoyIndex(vectors, 'angular')
    for index, row in dataset.iterrows():
        ann_index.add_item(index, row.tolist())

    ann_index.build(10)
    ann_index.save('models/basic.ann')

def find_neighbours(vectors, item_index, max_neighbours):
    
    model = AnnoyIndex(vectors, 'angular')
    model.load('models/basic.ann')
    neighbours = model.get_nns_by_item(item_index, max_neighbours)

    return neighbours


def filter_preferences(dataset, user_id):

    #Drop the options where the user has specified a preference
    user_preferences = survey_responses[survey_responses['UserID'] == user_id]
    user_colors = user_preferences['Out of the following color options, which would you be least likely to wear? Limit your choices to max. 10 options.']
    user_prints = user_preferences['Out of the following prints, which would you be least likely to wear? There is no choice limitation.']
    user_necklines = user_preferences['Out of the following necklines, which would you be least likely to wear? There is no choice limitation.']
    user_collars = user_preferences['Out of the following collars, which would you be least likely to wear? There is no choice limitation.']
    user_materials = user_preferences['Out of the following materials, which would you be least likely to wear? There is no choice limitation.']
    user_dress_type = user_preferences['Out of the following dress types, which would you be least likely to wear? There is no choice limitation.']

    # Remove the disliked colors
    user_colors = user_colors.str.split(',', expand=True)

    for index, color in user_colors.iteritems():
        color = str(color.values[0]).lower()
        matched_dresses = dataset.loc[dataset['Color'] == color]
        dataset = dataset[~dataset.isin(matched_dresses)].dropna()


    # Remove the disliked prints
    user_prints = user_prints.str.split(',', expand=True)

    for index, prints in user_prints.iteritems():
        prints = str(prints.values[0])
        matched_dresses = dataset.loc[dataset['Consolidated Prints'] == prints]
        dataset = dataset[~dataset.isin(matched_dresses)].dropna()


    # Remove the disliked necklines
    user_necklines = user_necklines.str.split(',', expand=True)

    for index, neckline in user_necklines.iteritems():
        neckline = str(neckline.values[0])
        matched_dresses = dataset.loc[dataset['Neckline'] == neckline]
        dataset = dataset[~dataset.isin(matched_dresses)].dropna()

    # Remove the disliked collars
    user_collars = user_collars.str.split(',', expand=True)

    for index, collar in user_collars.iteritems():
        collar = str(collar.values[0]).lower()
        matched_dresses = dataset.loc[dataset['Collar'] == collar]
        dataset = dataset[~dataset.isin(matched_dresses)].dropna()

    # Remove the disliked fabric
    user_materials = user_materials.str.split(',', expand=True)

    for index, fabric in user_materials.iteritems():
        fabric = str(fabric.values[0])
        matched_dresses = dataset.loc[dataset['Fabric'] == fabric]
        dataset = dataset[~dataset.isin(matched_dresses)].dropna()


    # Remove the disliked dress types
    user_dress_type = user_dress_type.str.split(',', expand=True)

    for index, dress in user_dress_type.iteritems():
        dress = str(dress.values[0])
        matched_dresses = dataset.loc[dataset['Type of dress'] == dress]
        dataset = dataset[~dataset.isin(matched_dresses)].dropna()
    
    dataset = dataset.reset_index(drop=True)

    return dataset


def find_top_recommendations(user_ratings):

    vectors = 9

    # Process, consolidate and vectorize our data
    processed_data = process_data(dress_data)
    consolidated_data = consolidate_data(processed_data)
    vectorized_data = hot_encode(consolidated_data)

    # Construct the ANN model from our vectorized data
    build_ANN_index(vectorized_data)

    for index, row in survey_responses.iterrows():
        user = row['UserID']
        print(user)
        single_user_ratings = user_ratings[user_ratings['UserID'] == user]
        single_user_ratings = single_user_ratings.sort_values('Rating', ascending=False)

        # Get recommendations for the top 2 rated dresses per user
        single_user_ratings = single_user_ratings.head(2)

        recommendations_index = []

        # Get nearest items for the top rated dresses
        for rating_index, rating_row in single_user_ratings.iterrows():
            item_index = processed_data[processed_data['Link'] == rating_row['Link'][1:]].index.values.astype(int)

            # Find recommendations for this top rated item
            neighbors = find_neighbours(vectors, int(item_index[0]), 100)
            
            # Add eacg recommendation index to our recommendations list for this user (100 recommendations per item)
            for item in neighbors:
                recommendations_index.append(item)
            
        recommendations = []

        # Now fetch the full row by the index captured above and add to our recommendations list
        for index in recommendations_index:
            recommend_row = processed_data.iloc[index,:]
            recommendations.append(recommend_row)
        
        # Add the column names back to the pandas dataframe
        recommendations = pd.DataFrame(recommendations, columns = processed_data.columns)
        
        # Include the user email address
        user_email = row['UserEmail']
        recommendations['User']=user_email

        # Drop any duplicate recommendations
        recommendations = recommendations.drop_duplicates()

        # Filter our recommendations
        recommendations = filter_preferences(recommendations, user)

        # Save the top 10 user recommendations to a file
        recommendations.head(10).to_csv('results/basic_features/'+ str(user) +'_' + user_email + '_recommendations.csv')


recommendations = find_top_recommendations(user_ratings)
