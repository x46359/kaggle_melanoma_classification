import numpy as np
import pandas as pd
import os
import csv
from data.make_data import CustomDataset, make_data
from models.train_models import train, run_training
from models.create_predictions import run_predictions
from paths import *
from utility import *
from models.run_lgb import read_predicts, sample, train, predict
from sklearn.model_selection import train_test_split

# use params
cuda_device=0
batch_size = 50
sample_test = 1000 # recommend minimum of 1000 samples to ensure enough minority samples
n_epochs = 2
n_iter = 30

size_list = [225]

# ['resnet18', 'alexnet', 'vgg16', 'densenet','googlenet','shufflenet', 'resnext50_32x4d', 'wide_resnet50_2']
model_list = ['vgg16']

# loop through model training based on model_list and size_list inputs
for size in size_list:

    # check data exists for train/test
    check_raw_data(data_train)
    check_raw_data(data_test)

    # create train and test loaders
    train_loader, test_loader = make_data(sample_test=sample_test,
                                        size=size,
                                        test_train_split_percent=0.8,
                                        batch_size=batch_size)

    # create csv files to ave model and evaluation info to
    create_csv(model_path/str('model_eval_'+ str(size) + '.csv'))
    create_csv(model_path/str('model_info_' + str(size) + '.csv'))

    # model loop
    for mod in model_list:

        print(f'\nRunning training for {mod}, image size {size}x{size}, and batch size {batch_size}')

        # run trainings
        run_training(mod=mod, 
            cuda_device=cuda_device, 
            size=size, 
            train_loader=train_loader, 
            test_loader=test_loader,
            n_epochs=n_epochs)
        
        # check if file exists, create predictions for train and test datasets
        file_train = str(mod + '_' + str(size) + '_train.csv')
        file_train_path = interim_path_train/file_train
        if os.path.exists(file_train_path):
            print(f'{file_train} already exists')
            pass
        else:
            run_predictions(train_test='train', img_size=size, batch_size=batch_size, gpu=cuda_device, mod=mod, sample_test=sample_test)
        
        file_test = str(mod + '_' + str(size) + '_test.csv')
        file_test_path = interim_path_test/file_test
        if os.path.exists(file_test_path):
            print(f'{file_test} already exists')
            pass
        else:
            run_predictions(train_test='test', img_size=size, batch_size=batch_size, gpu=cuda_device, mod=mod, sample_test=sample_test)


# Empty df for scoring, initial auc score
score_df = pd.DataFrame()
auc = 0

# loop through different ratios and save one with highest roc_auc score.
for sample_ratio in range(20, 60, 5):

    # Create train, valid, test datasets
    X_presplit, y_presplit, ratio = read_predicts('train', interim_path_train, sample_ratio)
    X_train, X_valid, y_train, y_valid = train_test_split(X_presplit, y_presplit, test_size=0.2, random_state=123, stratify=y_presplit)

    X_test, y_test, classes = read_predicts('test', interim_path_test, sample_ratio)

    # run sample function for only training set
    X_train_sample, y_train_sample = sample(X_train, y_train, ratio)    

    # Run grid search
    params = train(X_train_sample, y_train_sample, X_valid, y_valid, n_iter=n_iter)

    # Create predictions
    preds, sdf = predict(X_train_sample, y_train_sample, X_valid, y_valid, X_test, params)

    # final outputs
    output = preds[:,1]
    final = X_test.loc[:, X_test.columns == 'image_name'].copy()

    #need mapping from label encoder, reverse key/value, and replace
    image_map = {y:x for x,y in classes.items()}
    final['image_name'].replace(image_map, inplace=True)
    final['target'] = abs(output)

    # append scores
    sdf['ratio'] = sample_ratio
    score_df = score_df.append(sdf, ignore_index=True)

    # write csv
    new_auc = score_df['roc_auc'].iloc[-1]

    if new_auc > auc:
        auc = new_auc
        final.to_csv('lgb_submission.csv', index=False)

    # print distribution of predicted probabilities
    view_df = final.copy()
    view_df['target'] = view_df.target.round(1)
    print(f'\n\nFinal output distribution for ratio: {sample_ratio}')
    print(view_df.groupby('target').target.count())

