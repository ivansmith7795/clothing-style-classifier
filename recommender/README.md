# clothing-recommender
A recommendation system using approximate nearest neighbour (ANN) with features derived from the naive bayes style predictor.

# Installation Instructions
The ANN recommender used for this system uses the ANN system developed by Spotify ANNOY which can be found: https://github.com/spotify/annoy 

To install, simply do ** pip install --user annoy ** to pull down the latest version from PyPI.


# Understanding Annoy and Approximate Nearest Neighbhours #

The algorithm allows for the use of Euclidean distance, Manhattan distance, cosine distance, Hamming distance, or Dot (Inner) Product distance. We use cosine distanec for this experiment. We select ANNOY for the speed of predictions with a limited feature space (in this case < 100 features total). For low-dimensionality, approximate nearest neighbours provides the best tradeoff between speed of prediction and accuracy.

ANN works by using random projections and by building up a tree. At every intermediate node in the tree, a random hyperplane is chosen, which divides the space into two subspaces. This hyperplane is chosen by sampling two points from the subset and taking the hyperplane equidistant from them.

We do this k times so that we get a forest of trees. k has to be tuned to your need, by looking at what tradeoff you have between precision and performance.

Hamming distance (contributed by Martin Aumüller) packs the data into 64-bit integers under the hood and uses built-in bit count primitives so it could be quite fast. All splits are axis-aligned.

There are just two main parameters needed to tune Annoy: the number of trees n_trees and the number of nodes to inspect during searching search_k. We use a smaller number of tress as our feature space is limited.

n_trees is provided during build time and affects the build time and the index size. A larger value will give more accurate results, but larger indexes.
search_k is provided in runtime and affects the search performance. A larger value will give more accurate results, but will take longer time to return.
If search_k is not provided, it will default to n * n_trees where n is the number of approximate nearest neighbors. Otherwise, search_k and n_trees are roughly independent, i.e. the value of n_trees will not affect search time if search_k is held constant and vice versa. Basically it's recommended to set n_trees as large as possible given the amount of memory you can afford, and it's recommended to set search_k as large as possible given the time constraints you have for the queries.


- Approximate Nearest Neighbor techniques speed up the search by preprocessing the data into an efficient index and are often tackled using these phases:

- Vector Transformation — applied on vectors before they are indexed, amongst them, there is dimensionality reduction and vector rotation.
In order to this article well structured and somewhat concise, I won't discuss this.

- Vector Encoding — applied on vectors in order to construct the actual index for search, amongst these, there are data structure-based techniques like Trees, LSH, and Quantization a technique to encode the vector to a much more compact form.

- None Exhaustive Search Component — applied on vectors in order to avoid exhaustive search, amongst these techniques there are Inverted Files and Neighborhood Graphs.


# Using ANNOY Recommender for Clothing Classification #

Since the use of nearest neighbour techniques for recommender systems for similar categorical datasets has been employed in industry for some time, a similar approach using content-based recommendations derived from user selections is chosen to approximate the distance between our rated items and all other items in our data set. Approximate nearest neighbours doesn't compute the distance metric for every item in the set, only those that are statistically probable to be neighbours. This has the added advantage of being extremely fast when compared to other techniques, which for production applications where recommendations are needed in near-realtime is important.

Two algorithms are constructed for the purposes of experimentation. The first is a recommender system with limited dimensionality and feature space we will call the 'basic' algorithm. The second is a recommender which leverages additional features engineered from a bayesian classifier we use to derive a new categorical feature called 'style'. This in conjunction with additional features selected from an ecommerce website makeup the dimensions we will use to construct a more complex algorithm we will call 'advanced'.

## Building the Rating Matrix ##

Since our raw responses are exported from Qualtrics (the survey tool used to query users about their preferences and respective item ratings) a row-wise user to item rating matrix is needed to vectorize and ingest the ratings into the approximate nearest neighbour algorithm. We construct a script that normalizes the ratings in the row-wise pairing for this purpose:

First, we import pandas and numpy to manipulate the data and transform it:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
```

Then, we import our raw user response surveys and apparel data which will need to be processed to create the item matrix:

```python
survey_data = pd.read_csv('datasets/survey-responses.csv')
rated_dresses = pd.read_csv('datasets/dress-data-normalized.csv')
```

We construct a function to normalize our apparel data. This function replaces empty column values where "--" is specified with "Unknown". Unused columns (Article number) are also dropped along with any empty columns. Price data is converted to numeric type so hot-encoding is not required. We then save the data with the suffix "processed" for use later on.

```python
def normalize_rated_dresses(rated_dresses):
    
    # Copy the rated dress data 
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

```

The second function is used to build the item ratings matrix. Here we create a loop which iterates over the survey data (exported from Qualtrics). Each column question (which contains our item ratings) is then processed and converted into a row-wise rating (from a scale of 1 through 5). Since we know there are 20 questions, we loop 20 times per survey response to extract all 20 ratings from our survey responses.

The resulting matrix can now be used for building our recommender system using ANN.


```python
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
```


## Basic ANN Recommender ##

The basic recommender, as mentioned above, leverages an ANN classification system to choose items a given user would be interested in. This is largely a content-based classification system but could easily incorporate similar user preferences to make inferences about item purchases. 

The system makes use of the following features, which mimick the ecommerce standard filter options currently available to users:

'Brand', 'Price', 'Color', 'Outer fabric material', 'Collar', 'Sleeve length', 'Dress length', 'Consolidated Prints', 'Fabric'

The script for the basic recommender can be found in the ANN_recommender_basic.py file. Below is an explaination of the code. 

We import the ANNOY algorithm along with pandas an numpy to manipulate the data. Sklearn LabelBinarizer will be used for one-hot encoding our categorical features (which will be described later on):

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from sklearn.preprocessing import LabelBinarizer
```

We then import our datasets including the rating matrix
Once loaded, we clean the data and impute any missing values in the dataset using a simple imputer function. For categorical data, the data is imputed using the most commonly used value. We also set max_colwidth to infinite to avoid truncating links with long URLs:

```python
pd.set_option('display.max_colwidth', -1)

#Import into pandas
user_ratings = pd.read_csv('datasets/user_ratings_matrix.csv')
dress_data = pd.read_csv('datasets/dress-data-processed.csv')
survey_responses = pd.read_csv('datasets/survey_results_processed.csv')
```

We build a series of functions to perform different tasks on our data. The first processes data and removes special characters to normalize our apparel dataset further. Price column is once again directly converted to numeric to avoid the need to one-hot encode:

```python
def process_data (dress_data):

     # Drop our engineered features
    dress_data = dress_data.drop(columns=['Style'])

    # Clear empty field special characters and replace with blank values
    dress_data = dress_data.replace('--', '', regex=True)

    #Remove the price special characters and convert to numeric
    dress_data = dress_data.replace(to_replace ='Â£', value = '', regex = True)
    dress_data = dress_data.replace(to_replace ='Â', value = '', regex = True)

    #Convert the price column to numeric
    dress_data['Price'] = dress_data['Price'].apply(pd.to_numeric, errors='coerce')

    return dress_data

```

We also have a consolidation function which reduces our dimensionality to a limited number of features derived from Zalando. This includes only the basic filtering options currently available on from the website. We select these features to demonstrate the potential accuracy improvement of incorporating additional features in our advanced algorithm later on:

```python

def consolidate_data(combine_data):

    consolidated_data = combine_data.copy()

    #Drop all except Zalando filter columns
    consolidated_data = consolidated_data[['Brand', 'Price', 'Color', 'Outer fabric material', 'Collar', 'Sleeve length', 'Dress length', 'Consolidated Prints', 'Fabric']]

    #Remove all columns that are all NaN
    consolidated_data = consolidated_data.dropna(how='all', axis=1)

    return consolidated_data

```

There is also a one-hot encoding routine which converts our categorical features into their numeric representations. For ingestion into the ANN algorithm, a numeric feature-space is required. In this case, we choose a binarized encoder to preserve the ecludian distance measured during inference as using an ordinal encoder is likely not possible given then unordered nature of the data (in this case):

```python
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

```

We create a function to build the ANN recommender using 9 vectors (our feature space contains 9 dimensions). The model is then saved for future inference:

```python
def build_ANN_index(dataset):

    vectors = 9

    ann_index = AnnoyIndex(vectors, 'angular')
    for index, row in dataset.iterrows():
        ann_index.add_item(index, row.tolist())

    ann_index.build(10)
    ann_index.save('models/basic.ann')
```

A find_neighbours function is constructed to perform our inference. Max neighbours is defined as the number of recommendations. The item index is used to specify the item we want to compare and find neighbours for (rated user item) and vectors is our feature space (in this case, 9):


```python
def find_neighbours(vectors, item_index, max_neighbours):
    
    model = AnnoyIndex(vectors, 'angular')
    model.load('models/basic.ann')
    neighbours = model.get_nns_by_item(item_index, max_neighbours)

    return neighbours

```

Since we a have additional data about our users regarding their specific preferences, we build a filter function to remove recommendations we know they do not like. Since all features are weighted equally in this experiment, filtering is required to remove outlier recommendations using some business logic. There are a series of 6 questions that were asked in the survey that provide the details of this shown below. We simply extract those responses and remove recommendations that match:


```python
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

```

Finally, we construct our find_top_recommendations function. This function applies all other functions above to derive our recommendations list on a per-user basis.

Here, we process our data, consolidate it, vectorize it (using one-hot encoding) and build the ANN model. Once done, we loop over every survey response in our item matrix, and extract all ratings for each user to use as the basis for our recommendation. In this case, the top 2 rated items are selected to perform a recommendation. We use a slightly different methodology for the advanced recommender to be described later on.

Each item has 100 recommendations generated (200 in total). These recommendations are then filtered (based on our filter business logic above) and only the top 10 of the remaining recommendations are then selected. The resulting recommendations are then saved to a file on a per-user basis for capturing qualitative feedback on later on:

```python
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

```

That's it! Future work will include potentially adjusting the feature weights and experimenting with alternate distance metrics to futher tune the model.

## Advanced ANN Recommender ##

The second recommender algorithm constructed as part of this experiment is the 'advanced' algorithm. The algorithm is similar to the basic algorithm with a few key differences. The advanced ANN recommender uses a larger set of features to 'predict' user recommendations based on positively rated items.

The algorithm makes use of the following key features (both engineered and scraped from the ecommerce site) for making recommendations:

'Style', 'Dress name', 'Brand', 'Color', 'Outer fabric material', 'Collar', 'Sleeve length', 'Dress length', 'Consolidated Prints', 'Fabric', 'Price', 'Price Category', 'Lining', 'Care instructions', 'Cut', 'Fastening', 'Type of dress', 'Details', 'Consolidated sleeve length', 'Fit', 'Consolidated fit', 'Back width', 'Sheer?', 'Pockets?', 'Rhinestones?', 'Shoulderpads?', 'Backless?'

Importantly, the algorithm incorporates the 'Style' attribute in conjunction with other un-filterable attributes to make recommendations. The style feature is engineered using a Naive Bayes algorithm which predicts the style of a given garmet using similar features to those listed above, and also leverages a instance segmentation routine which downloads the image of each garmet, analysis the pixels and makes a determination as to the 'style' of a given article of clothing. The computer vision prediction along with the categorical and numeric descriptive attributes are what makes this a novel approach to ecommerce-type recommender systems.

Understanding that we are using engineer features to make recommendations, our hypothesis is tested by allowing the end-user to review our recommendations and score each recommendation on a scale from 1 through 10 where 10 is 'most likely to buy' and one is 'least likely to buy'. We use the scale to gauge and continuously improve the system during retraining while historical data is gathered (i.e. additional purchases and ratings).

The code for the advanced recommender can be found in the ANN_recommender_advanced.py script:

First, we import the required python libraries including pandas, numpy the ANNOY algorithm and LabelBinarizer from Sklearn for one-hot encoding our categorical features:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from sklearn.preprocessing import LabelBinarizer
```


Next, we import our datasets from relavent sources. Note we using the Naive Bayes (nb) output which includes the style classifications for the complete dataset:

```python
#Import into pandas
user_ratings = pd.read_csv('datasets/user_ratings_matrix.csv')
dress_data = pd.read_csv('../datasets/dress-dataset-all-processed-nb-predictions.csv')
survey_responses = pd.read_csv('datasets/survey_results_processed.csv')
```

As with the basic recommender, we perform some preprocessing to remove noisey data:

```python
def process_data (dress_data):


    # Clear empty field special characters and replace with blank values
    dress_data = dress_data.replace('--', '', regex=True)

    # Replace NaN values with 'Unknown'
    dress_data = dress_data.fillna('Unknown')

    #Remove the price special characters and convert to numeric
    dress_data = dress_data.replace(to_replace ='Â£', value = '', regex = True)
    dress_data = dress_data.replace(to_replace ='Â', value = '', regex = True)

    #Convert the price column to numeric
    dress_data['Price'] = dress_data['Price'].apply(pd.to_numeric, errors='coerce')
    dress_data['Price Category'] = dress_data['Price Category'].apply(pd.to_numeric, errors='coerce')

    return dress_data
```

Additionally, we consolidate our data to include 27 vectors, including style for our ANN algorithm to build a recommendation:

```python
def consolidate_data(combine_data):

    consolidated_data = combine_data.copy()

    #Include our advanced features
    consolidated_data = consolidated_data[['Style', 'Dress name', 'Brand', 'Color', 'Outer fabric material', 'Collar', 'Sleeve length', 'Dress length', 'Consolidated Prints', 'Fabric', 'Price', 'Price Category', 'Lining', 'Care instructions', 'Cut', 'Fastening', 'Type of dress', 'Details', 'Consolidated sleeve length', 'Fit', 'Consolidated fit', 'Back width', 'Sheer?', 'Pockets?', 'Rhinestones?', 'Shoulderpads?', 'Backless?']]

    #Remove all columns that are all NaN
    consolidated_data = consolidated_data.dropna(how='all', axis=1)

    return consolidated_data
```

We also hot-encode our categorical features as ANN requires a numeric feature space to measure ecludian (or cosine) distance:

```python
def hot_encode(dataset):

    encoded = dataset.copy()

    # One hot encode all our categorical data to nunmeric representations
    encoded['Style'] = LabelBinarizer().fit_transform(dataset['Style'])
    encoded['Dress name'] = LabelBinarizer().fit_transform(dataset['Dress name'])
    encoded['Brand'] = LabelBinarizer().fit_transform(dataset['Brand'])
    encoded['Color'] = LabelBinarizer().fit_transform(dataset['Color'])
    encoded['Outer fabric material'] = LabelBinarizer().fit_transform(dataset['Outer fabric material'])
    encoded['Collar'] = LabelBinarizer().fit_transform(dataset['Collar'])
    encoded['Sleeve length'] = LabelBinarizer().fit_transform(dataset['Sleeve length'])
    encoded['Dress length'] = LabelBinarizer().fit_transform(dataset['Dress length'])
    encoded['Consolidated Prints'] = LabelBinarizer().fit_transform(dataset['Consolidated Prints'])
    encoded['Fabric'] = LabelBinarizer().fit_transform(dataset['Fabric'])
    encoded['Price'] = dataset['Price']
    encoded['Price Category'] = dataset['Price Category']
    encoded['Lining'] = LabelBinarizer().fit_transform(dataset['Lining'])
    encoded['Care instructions'] = LabelBinarizer().fit_transform(dataset['Care instructions'])
    encoded['Cut'] = LabelBinarizer().fit_transform(dataset['Cut'])
    encoded['Fastening'] = LabelBinarizer().fit_transform(dataset['Fastening'])
    encoded['Type of dress'] = LabelBinarizer().fit_transform(dataset['Type of dress'])
    encoded['Details'] = LabelBinarizer().fit_transform(dataset['Details'])
    encoded['Consolidated sleeve length'] = LabelBinarizer().fit_transform(dataset['Consolidated sleeve length'])
    encoded['Fit'] = LabelBinarizer().fit_transform(dataset['Fit'])
    encoded['Consolidated fit'] = LabelBinarizer().fit_transform(dataset['Consolidated fit'])
    encoded['Back width'] = LabelBinarizer().fit_transform(dataset['Back width'])
    encoded['Sheer?'] = LabelBinarizer().fit_transform(dataset['Sheer?'])
    encoded['Pockets?'] = LabelBinarizer().fit_transform(dataset['Pockets?'])
    encoded['Rhinestones?'] = LabelBinarizer().fit_transform(dataset['Rhinestones?'])
    encoded['Shoulderpads?'] = LabelBinarizer().fit_transform(dataset['Shoulderpads?'])
    encoded['Backless?'] = LabelBinarizer().fit_transform(dataset['Backless?'])
```

We then build our ANN model. Note this is actually a feature map, not a set of weights and parameters to build a function representing the dataset as one would expect with most abstraction models. This is because KNN is inherently a 'lazy learning' method where the entire dataset is loaded into memory to make predictions directly. Also note our vector space is set to 27 to reflect the number of features we use:

```python
def build_ANN_index(dataset):

    vectors = 27

    ann_index = AnnoyIndex(vectors, 'angular')
    for index, row in dataset.iterrows():
        ann_index.add_item(index, row.tolist())

    ann_index.build(10)
    ann_index.save('models/advanced.ann')
```

Our find_neighbours function is constructed to predict nearest neighbours:

```python
def find_neighbours(vectors, item_index, max_neighbours):
    
    model = AnnoyIndex(vectors, 'angular')
    model.load('models/advanced.ann')
    neighbours = model.get_nns_by_item(item_index, max_neighbours)

    return neighbours
```

As with the basic recommender, we also filter out items from our recommendations list our users have explicitly indicated they are not interested in:

```python
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

```

Finally, we perform the steps necessary to make predictions. Data is cleaned, consolidated and encoded. The model is then constructed and the top user ratings are selected to build recommendations.

Note, the only difference from this recommender and the basic recommender are the number of features we include in the system (27 vs 9).

Once recommendations are build (100 identified items for each of the top 2 rated items) will filter out those our user has noted are undesired (based on color selection, print pattern etc) and the remaining top 10 are used to build our item recommendations and saved to a file for review and feedback.


```python
def find_top_recommendations(user_ratings):

    vectors = 27

    processed_data = process_data(dress_data)
    consolidated_data = consolidate_data(processed_data)
    vectorized_data = hot_encode(consolidated_data)
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

            neighbors = find_neighbours(vectors, int(item_index[0]), 100)
            print(rating_row['Rating'])
            for item in neighbors:
                recommendations_index.append(item)
            
        recommendations = []

        for index in recommendations_index:
            recommend_row = processed_data.iloc[index,:]
            recommendations.append(recommend_row)
        
        recommendations = pd.DataFrame(recommendations, columns = processed_data.columns)
        user_email = row['UserEmail']

        recommendations['User']=user_email

        # Drop any duplicate recommendations
        recommendations = recommendations.drop_duplicates()

        # Filter our recommendations
        recommendations = filter_preferences(recommendations, user)

        # Save the top 10 user recommendations to a file
        recommendations.head(10).to_csv('results/advanced_features/'+ str(user) +'_' + user_email + '_recommendations.csv')

```

This concludes the item recommender documentation.
