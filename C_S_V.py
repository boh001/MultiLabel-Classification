import pandas as pd
import numpy as np
import os
import glob

dir_csv = '/data1/kaggle-hemorrhage'
dir_train_img = '/data1/kaggle-hemorrhage/stage_1_train_images_png'
train = pd.read_csv(os.path.join(dir_csv, 'stage_1_train.csv'))
test = pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))
train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']
train.head()

png = glob.glob(os.path.join(dir_train_img, '*.png'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)

train = train[train['Image'].isin(png)]
train.to_csv('train.csv', index=False)

test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)
test['Image'] = 'ID_' + test['Image']
test = test[['Image', 'Label']]
test.drop_duplicates(inplace=True)

test.to_csv('test.csv', index=False)