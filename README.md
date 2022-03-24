# clothing-style-classifier
A classification project for various clothing styles using clothing attributes built using a Naive Bayes model and H20 framework.

# Installation Instructions
The python scripts for training and making predictions uses the H20 machine learning framework package found here: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html. This package is selected for its ease of implementation and readible code. While many of the underlying bayesian concepts are abstracted away from the user for the code to work, the training and prediction process are simplified.

To install H20 on your workstation to use this implementation, please use the command:

Run the following commands in a Terminal window to install H2O for Python.

Install dependencies (prepending with sudo if needed):

pip install requests
pip install tabulate
pip install future
Note: These are the dependencies required to run H2O. A complete list of dependencies is maintained in the following file: https://github.com/h2oai/h2o-3/blob/master/h2o-py/conda/h2o/meta.yaml.

Run the following command to remove any existing H2O module for Python.

pip uninstall h2o
Use pip to install this version of the H2O Python module.

pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
Note: When installing H2O from pip in OS X El Capitan, users must include the --user flag. For example:

pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o --user
Optionally initialize H2O in Python and run a demo to see H2O at work.

import h2o
h2o.init()
h2o.demo("glm")

# Training and Prediction Process of Naive Bayes with H20
Naïve Bayes is a classification algorithm that relies on strong assumptions of the independence of covariates in applying Bayes Theorem. The Naïve Bayes classifier assumes independence between predictor variables conditional on the response, and a Gaussian distribution of numeric predictors with mean and standard deviation computed from the training dataset.

Naïve Bayes models are commonly used as an alternative to decision trees for classification problems. When building a Naïve Bayes classifier, every row in the training dataset that contains at least one NA will be skipped completely. If the test dataset has missing values, then those predictors are omitted in the probability calculation during prediction.

The output from Naïve Bayes is a list of tables containing the a-priori and conditional probabilities of each class of the response. The a-priori probability is the estimated probability of a particular class before observing any of the predictors. Each conditional probability table corresponds to a predictor column. The row headers are the classes of the response and the column headers are the classes of the predictor. Thus, in the sample output below, the probability of survival (y) given a person is male (x) is 0.51617440.

When the predictor is numeric, Naïve Bayes assumes it is sampled from a Gaussian distribution given the class of the response. The first column contains the mean and the second column contains the standard deviation of the distribution.

By default, the following output displays:

Output, including model category, model summary, scoring history, training metrics, and validation metrics

Y-Levels (levels of the response column)

A Priori response probabilities

P-conditionals

# Explaination of the Bayes Classifier Used for this Implmentation

The algorithm is presented for the simplified binomial case without loss of generality.

Under the Naive Bayes assumption of independence, given a training set for a set of discrete valued features X (X(i),y(i);i=1,...m)

The joint likelihood of the data can be expressed as:

L(ϕ(y),ϕi|y=1,ϕi|y=0)=Πmi=1p(X(i),y(i))

The model can be parameterized by:

ϕi|y=0=p(xi=1|y=0);ϕi|y=1=p(xi=1|y=1);ϕ(y)

where ϕi|y=0=p(xi=1|y=0) can be thought of as the fraction of the observed instances where feature xi is observed, and the outcome is y=0,ϕi|y=1=p(xi=1|y=1) is the fraction of the observed instances where feature xi is observed, and the outcome is y=1, and so on.

The objective of the algorithm is to maximize with respect to ϕi|y=0, ϕi|y=1, and ϕ(y) where the maximum likelihood estimates are:

ϕj|y=1=Σmi1(x(i)j=1 ⋂yi=1)Σmi=1(y(i)=1)

ϕ_j|y=0=Σmi1(x(i)j=1 ⋂yi=0)Σmi=1(y(i)=0)

ϕ(y)=(yi=1)m

Once all parameters ϕj|y are fitted, the model can be used to predict new examples with features X(i∗). This is carried out by calculating:

p(y=1|x)=Πp(xi|y=1)p(y=1)Πp(xi|y=1)p(y=1)+Πp(xi|y=0)p(y=0)

p(y=0|x)=Πp(xi|y=0)p(y=0)Πp(xi|y=1)p(y=1)+Πp(xi|y=0)p(y=0)

and then predicting the class with the highest probability.

It is possible that prediction sets contain features not originally seen in the training set. If this occurs, the maximum likelihood estimates for these features predict a probability of 0 for all cases of y.

Laplace smoothing allows a model to predict on out of training data features by adjusting the maximum likelihood estimates to be:

ϕj|y=1=Σmi1(x(i)j=1 ⋂yi=1)+1Σmi=1(y(i)=1+2)

ϕj|y=0=Σmi1(x(i)j=1 ⋂yi=0)+1Σmi=1(y(i)=0+2

Note that in the general case where y takes on k values, there are k+1 modified parameter estimates, and they are added in when the denominator is k (rather than 2, as shown in the two-level classifier shown here).

Laplace smoothing should be used with care; it is generally intended to allow for predictions in rare events. As prediction data becomes increasingly distinct from training data, train new models when possible to account for a broader set of possible X values.


# Applied Use for Clothing Classifier

We first training script naive_bayes_train.py to build the model artifact (/models/nb_fit1). This contains our weights and parameters needed to make a prediction, and load the model into memory (optimization memory utilization for production). The number of threads and maximum memory allocated for the training process can be specified after the package is imported:

import h2o
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator

#Specify the number of threads the H20 framework will consume
h2o.init(nthreads = -1, max_mem_size = 8)

Once initialized, we load the data set from the manually annotated set into the program:
#Load the data from the labeled data set
data_csv = "datasets/dataset_dresses_labeled_processed.csv" 
data = h2o.import_file(data_csv)

We then split the labeled set into three parts, training set, validation set and the test set (or holdout set) using a 75 / 15 / 5 split respectively:

#Split out data set into 3 parts for training, validation and testing.
splits = data.split_frame(ratios=[0.7, 0.15], seed=1)  
train = splits[0]
valid = splits[1]
test = splits[2]

The test set in this implementation is required to validate our results and produce a confusion matrix for demonstrating false positives.

Next, we define the response column we're interested in predicting for (style column) and remove it from the set (Naive Bayes requires the response variable is introduced seperately from our independent variables during training)

y = 'Styles'
x = list(data.columns)

x.remove(y)  #remove the response
x.remove('Style options') 
x.remove('link')

Once the data has been seperated into response and dependent variables, we begin training. A seed value of 1 is provided to randomize the underlying shuffling of the dataset:

#Train the model and produce the model file nb_fit1
nb_fit1 = H2ONaiveBayesEstimator(model_id='nb_fit1', seed=1)
nb_fit1.train(x=x, y=y, training_frame=train)


Other parameters can be tuned in the H20NaiveBayesEstimator function if desired, including specifying the number of nfolds for cross-validation testing (the default of 0 was used in this test). Future work may include tuining this parameter for a better result.

Performance metrics and the calculated RMSE score of the model is produced using the holdout set (test set from the remaining 5% of the labeled data set described earlier). RMSE is used as the scoring method and not AUC, since AUC scores have been proven to be unreliable.

#Produce the performance metrics
nb_perf1 = nb_fit1.model_performance(test)

#Print the RMSE score of the model
print("Naive Bayes Estimator:")
print(nb_perf1.rmse())

The permutated variable importance matrix is the produced to show the relative information gain of each of the predictor variables. This is calculated by measuring the distance between prediction errors before and after a feature is permuted; only one feature at a time is permuted.

#calculate variable importance and export to a csv file
`nb_permutation_varimp = nb_fit1.permutation_importance(train, use_pandas=True)`  
`print(nb_permutation_varimp)`  
`nb_permutation_varimp.reset_index(level=0, inplace=True)`  
`frame = h2o.H2OFrame(nb_permutation_varimp)`  
`h2o.export_file(frame, path = "results/naive_bayes_permutation_importance.csv", force=True)`  


We then produce the confusion matrix to better interpret which style has the most false positives when predicting for the test set:

#Retrieve the confusion matrix
`conf_matrix = nb_perf1.confusion_matrix()`  
`print(conf_matrix)`


#Export the confusion matrix

`frame = h2o.H2OFrame(conf_matrix.as_data_frame())`  
`h2o.export_file(frame, path = "results/naive_bayes_confusion_matrix.csv", force=True)`


And finally, we save our model file for prediction:

`model_path = h2o.save_model(model=nb_fit1, path="models", force=True)`





