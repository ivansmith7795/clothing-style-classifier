import pandas as pd


df = pd.concat(
	map(pd.read_csv, ['datasets/dataset_1.csv', 'datasets/dataset_2.csv', 'datasets/dataset_3.csv']), ignore_index=True)

print(df)
df.to_csv('datasets/dataset_merged.csv')
