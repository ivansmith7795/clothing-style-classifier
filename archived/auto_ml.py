import h2o
from h2o.automl import H2OAutoML

# Start the H2O cluster (locally)
h2o.init(nthreads = -1, max_mem_size = 8)
data_csv = "../datasets/dataset_dresses_labeled_processed.csv" 

data = h2o.import_file(data_csv)

splits = data.split_frame(ratios=[0.7, 0.15], seed=1)  

train = splits[0]
valid = splits[1]
test = splits[2]

y = 'Styles'
x = list(data.columns)

x.remove(y)  #remove the response

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)
h2o.export_file(lb, path = "../results/classification_algo_leaders.csv")

#Export the models (commented out to avoid overwriting the bayes classifier)
##for m in aml.leaderboard.as_data_frame()['model_id']:
##   model = h2o.get_model(m)
    
    # Retrieve the variable importance
##    varimp = model.varimp(use_pandas=True)
##    print(varimp)
##   h2o.export_file(varimp, path = "../models/")

##    confusion_matrix = model.confusion_matrix()
##    h2o.export_file(confusion_matrix, path = "../models/")

##    model_path = h2o.save_model(model=m, path="models", force=True)
