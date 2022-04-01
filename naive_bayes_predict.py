import h2o
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator

#First, initialize the H2O agent and load the unlabeled samples
h2o.init(nthreads = -1, max_mem_size = 8)
data_csv = "datasets/dataset_dresses_unlabeled_cv.csv" 
data = h2o.import_file(data_csv)

#Set the prdictor columns to every column (since the repsponse variable is not part of this set)
prediction_set = list(data.columns)


# load the model
saved_model = h2o.load_model('models/nb_fit1')

# Predict what the style is for each row of data
predicted_style = saved_model.predict(data)
print(predicted_style)

prediction_table = predicted_style.cbind(data)
h2o.export_file(prediction_table, path = "results/naive_bayes_preditions.csv", force=True)
