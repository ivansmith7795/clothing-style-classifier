import h2o
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

h2o.init(nthreads = -1, max_mem_size = 8)
data_csv = "datasets/dataset_dresses_processed.csv" 

data = h2o.import_file(data_csv)

splits = data.split_frame(ratios=[0.7, 0.15], seed=1)  

train = splits[0]
valid = splits[1]
test = splits[2]

y = 'Styles'
x = list(data.columns)

x.remove(y)  #remove the response
x.remove('Style options') 
x.remove('link')

nb_fit1 = H2ONaiveBayesEstimator(model_id='nb_fit1', seed=1)
nb_fit1.train(x=x, y=y, training_frame=train)

nb_fit2 = H2ONaiveBayesEstimator(model_id='nb_fit2', laplace=1, nfolds=100, seed=1)
nb_fit2.train(x=x, y=y, training_frame=train)


nb_perf1 = nb_fit1.model_performance(test)
nb_perf2 = nb_fit2.model_performance(test)
print("Naive Bayes Estimator:")
print(nb_perf1.rmse())
print(nb_perf2.rmse())

# calculate importance
nb_permutation_varimp = nb_fit1.permutation_importance(train, use_pandas=True)

# plot permutation importance (bar plot)
nb_fit1.permutation_importance_plot(train)

# plot permutation importance (box plot)
nb_fit1.permutation_importance_plot(train, n_repeats=15)


rf_fit1 = H2ORandomForestEstimator(model_id='rf_fit1', seed=1)
rf_fit1.train(x=x, y=y, training_frame=train)

rf_fit2 = H2ORandomForestEstimator(model_id='rf_fit2', ntrees=200, seed=1)
rf_fit2.train(x=x, y=y, training_frame=train)


rf_perf1 = rf_fit1.model_performance(test)
rf_perf2 = rf_fit2.model_performance(test)
print("Random Forest Estimator:")
print(rf_perf1.rmse())
print(rf_perf2.rmse())

dl_fit1 = H2ODeepLearningEstimator(model_id='dl_fit1', seed=1)
dl_fit1.train(x=x, y=y, training_frame=train)

dl_fit2 = H2ODeepLearningEstimator(model_id='dl_fit2', 
                                   epochs=200, 
                                   hidden=[10,10], 
                                   stopping_rounds=0,  #disable early stopping
                                   seed=1)
dl_fit2.train(x=x, y=y, training_frame=train)


dl_perf1 = dl_fit1.model_performance(test)
dl_perf2 = dl_fit2.model_performance(test)


print("Deep Learning Estimator:")
print(dl_perf1.rmse())
print(dl_perf2.rmse())