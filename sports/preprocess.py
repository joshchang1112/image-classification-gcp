import pandas as pd

summary = pd.read_csv('sports.csv')
_map = {'train': 'TRAIN', 'valid': 'VALIDATION', 'test': 'TEST'}
summary['data set'] = summary['data set'].apply(lambda x: _map[x])
summary['filepaths'] = summary['filepaths'].apply(lambda x: 'gs://my-storage-bucket-vcm/' + x)
summary = summary[['data set', 'filepaths', 'labels']]

summary.to_csv('sports_automl.csv', header=False, index=False)
