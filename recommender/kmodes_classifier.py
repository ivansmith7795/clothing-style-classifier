import warnings
warnings.filterwarnings('ignore')

# Importing all required packages
import numpy as np
import pandas as pd

# Data visualization library
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks

from sklearn import preprocessing
from kmodes.kmodes import KModes

#Load our clean dataset
dataset = pd.read_csv('datasets/survey-responses.csv')


#Convert our bool (active) column into a categorical feature
#dataset['active'] = dataset['active'].astype('object')
                    
print(dataset.columns)
print(dataset.info())

#Copy the data
dataset_copy = dataset.copy()

#One-hot encode the dataset
le = preprocessing.LabelEncoder()
dataset = dataset.apply(le.fit_transform)
print(dataset.head())

#Use Cao method of Kmodes to get categorical groupings
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(dataset)

print(fitClusters_cao)

clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = dataset.columns

print(clusterCentroidsDf)

#Finding K cost
cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(dataset)
    cost.append(kmode.cost_)


y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)

plt.savefig('kmodes.png', format="png")

#Looks like K - 2 is the best
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(dataset)

#Combine cluster labels with the original set
dataset = dataset_copy.reset_index()

clustersDf = pd.DataFrame(fitClusters_cao)
clustersDf.columns = ['user_class']
combinedDf = pd.concat([dataset, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)

print(combinedDf.head())

combinedDf.to_csv('datasets/kmodes_labeled_combined.csv')
#Identify each cluster
cluster_0 = combinedDf[combinedDf['user_class'] == 0]
cluster_1 = combinedDf[combinedDf['user_class'] == 1]

print(cluster_0.info())
print(cluster_1.info())

#Plot for comparing the samples per cluster
#plt.subplots(figsize = (15,5))
#sns.countplot(x=combinedDf['heading1'],order=combinedDf['heading1'].value_counts().index,hue=combinedDf['cluster_predicted'])
#plt.savefig('clusters.png', format="png")

