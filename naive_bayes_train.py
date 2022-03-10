import h2o
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator

h2o.init(nthreads = -1, max_mem_size = 8)
data_csv = "datasets/dataset_dresses_labeled_processed.csv" 

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

nb_perf1 = nb_fit1.model_performance(test)

print("Naive Bayes Estimator:")
print(nb_perf1.rmse())

# calculate variable importance
nb_permutation_varimp = nb_fit1.permutation_importance(train, use_pandas=True)
print(nb_permutation_varimp)
nb_permutation_varimp.reset_index(level=0, inplace=True)
frame = h2o.H2OFrame(nb_permutation_varimp)
h2o.export_file(frame, path = "results/naive_bayes_permutation_importance.csv", force=True)

# Retrieve the confusion matrix
conf_matrix = nb_perf1.confusion_matrix()
print(conf_matrix)

frame = h2o.H2OFrame(conf_matrix.as_data_frame())
#frame = frame.reset_index(level=0, inplace=True)
h2o.export_file(frame, path = "results/naive_bayes_confusion_matrix.csv", force=True)

model_path = h2o.save_model(model=nb_fit1, path="models", force=True)


