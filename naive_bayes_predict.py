import h2o
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator

h2o.init(nthreads = -1, max_mem_size = 8)
data_csv = "datasets/dataset_dresses_unlabeled_processed.csv" 

data = h2o.import_file(data_csv)
prediction_set = list(data.columns)


# load the model
saved_model = h2o.load_model('models/nb_fit1')

predicted_style = saved_model.predict(data)
print(predicted_style)

prediction_table = predicted_style.cbind(data)
h2o.export_file(prediction_table, path = "results/naive_bayes_preditions.csv", force=True)
