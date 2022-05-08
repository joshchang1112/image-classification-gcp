import pandas as pd
import os

with open('meta/meta/train.txt', 'r') as f:
    train = f.read().split('\n')

class_dict = {}
for data in train:
    data = data.strip()
    if not data:
        break
    _class, _id = data.split('/') 
    class_dict[_class] = class_dict.get(_class, []) + [data]

dataset = []
filepaths = []
class_list = []
for c, path in class_dict.items():
    for i in range(len(class_dict[c])):
        if i < int(len(class_dict[c])*0.8):
            dataset.append('TRAIN')
        else:
            dataset.append('VALIDATION')
        filepaths.append(path[i])
        class_list.append(c)

with open('meta/meta/test.txt', 'r') as f:
    test = f.read().split('\n')

class_dict = {}
for data in test:
    data = data.strip()
    if not data:
        break
    _class, _id = data.split('/') 
    class_dict[_class] = class_dict.get(_class, []) + [data]

for c, path in class_dict.items():
    for i in range(len(class_dict[c])):
        dataset.append('TEST')
        filepaths.append(path[i])
        class_list.append(c)

summary = pd.DataFrame(data={'dataset': dataset, 'filepaths': filepaths, 'class': class_list})
summary['filepaths'] = summary['filepaths'].apply(lambda x: 'gs://my-storage-bucket-vcm/images/'+x+'.jpg') 
summary.to_csv('food_automl.csv', header=False, index=False)

for (_set, c, path) in zip(dataset, class_list, filepaths):
    if not os.path.isdir(os.path.join(_set.lower(), c)):
        os.makedirs(os.path.join(_set.lower(), c))
    os.rename(os.path.join('images', path+'.jpg'), os.path.join(_set.lower(), path+'.jpg'))
