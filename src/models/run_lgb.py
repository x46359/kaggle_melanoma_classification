import numpy as np
import pandas as pd
import lightgbm as lgb
import os.path
import imblearn
import glob
from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier, LGBMModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn import metrics
from paths import *

def read_predicts(test_train, 
                  path, 
                  under_sample_ratio):

    """Reads in all predictions and creates required transformations

    Params
    --------
        train_test (str): inputs are either 'train' or 'test, runs predictions for test or train dataset. 
        path (path): path to interim predictions
        under_sample_ratio (int): various ratios used to determine correct amount of undersamping of majority class


    Returns
    --------
        X (df): dataframe representing independent (non-target) variables
        y (df): dataframe representing target
        ratio (int): ratio to be used for under sampling
        le_name_mapping (dict): dictionary of image_name and label encoding, used for conversion of encoded values in final output
    """

    # file path 
    files = glob.glob(str(path) + '/*.csv')
    raw_df = pd.read_csv(data_raw_path/str(test_train + '.csv'))

    # loop to append all files in respective train/test folder
    for files in files:
        split_it = files.split("\\")[-1]
        target_name = "_".join(split_it.split("_")[:-1])

        predict_df = pd.read_csv(files)
        predict_df = predict_df.rename(columns={'target':'target1'})

        raw_df = raw_df.merge(predict_df, on='image_name')
        raw_df = raw_df.rename(columns={'target1':target_name})

    cat_list = ['patient_id','sex','anatom_site_general_challenge'] # categorical columns

    # label encoding
    le = LabelEncoder()

    # image_name column done separately to keep mapping for conversion label encoded image_name back to string
    raw_df['image_name'] = raw_df['image_name'].astype('str')
    raw_df['image_name'] = le.fit_transform(raw_df['image_name'])
    raw_df['image_name'] = raw_df['image_name'].astype('category') # classify as category
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_))) # keep mapping for conversion later

    # approx_age has NaN values
    raw_df = raw_df.fillna(0) 

    # label encoding for rest of columns in cat_list
    for column in raw_df:
        if column in cat_list:
            raw_df[column] = raw_df[column].astype('str')
            raw_df[column] = le.fit_transform(raw_df[column])
            raw_df[column] = raw_df[column].astype('category')
        elif column == 'image_name':
            raw_df[column] = raw_df[column].astype('category')    

    # if 'test' then return (X, y, mapping for conversion)
    if test_train=='test':
        X = raw_df.loc[:, raw_df.columns != 'target']
        y = raw_df.loc[:, raw_df.columns == 'target']
        
        return X, y, le_name_mapping

    # if 'train' then return (X, y, ratio)    
    else:
        # distribution of target [0,1] and ratio created using input parameter
        dist = raw_df.groupby('target').target.count()
        ratio = (dist[1]/dist[0])*under_sample_ratio

        X = raw_df.loc[:, raw_df.columns != 'target']
        y = raw_df.loc[:, raw_df.columns == 'target']  

        X.drop(columns = ['diagnosis', 'benign_malignant'], inplace=True)

        return X, y, ratio

def sample(X, y, ratio):

    """Undersamples majority and synthetic minority samples using SMOTE

    Params
    --------
        X (df): dataframe representing independent (non-target) variables
        y (df): dataframe representing target
        ratio (int): ratio to be used for under sampling


    Returns
    --------
        X_over (df): dataframe representing independent (non-target) variables, with undersampled majority/SMOTE minority
        y_over (df): dataframe representing target, with undersampled majority/SMOTE minority
    """

    # for sample runs, need to ensure k_neighbors is less than minority samples
    n_minority_samples = y.groupby('target').target.count()[1]

    if n_minority_samples < 5:
        k_neighbors = n_minority_samples-2
    else:
        k_neighbors = 5

    # under sample majority based on ratio
    undersample = RandomUnderSampler(sampling_strategy=ratio, random_state=123)
    X_under, y_under = undersample.fit_resample(X, y)

    # synthetic oversample via SMOTE
    # oversample = BorderlineSMOTE(random_state=123)#, sampling_strategy=.25)#, random_state=123)
    # oversample = SVMSMOTE(random_state=123)#, sampling_strategy=.25)#, random_state=123)
    oversample = SMOTENC(categorical_features=[0,1,2,4],random_state=123, k_neighbors=k_neighbors)#, sampling_strategy=.25)#, random_state=123)
    X_over, y_over = oversample.fit_resample(X_under,y_under)

    return X_over, y_over

def train(X_train, 
          y_train, 
          X_valid, 
          y_valid, 
          n_iter):
    
    """Rund randomized search for best params

    Params
    --------
        X_train (df): dataframe with undersampled majority/SMOTE minority
        y_train (df): dataframe with undersampled majority/SMOTE minority
        X_valid (df): unmodified dataframe for validation
        y_valid (df): unmodified dataframe for validation


    Returns
    --------
        best_params (params): returns roc_auc optimized params
    """

    folds = 3 # folds for cv

    # grid of params to randomly iterate through
    params = {
        'num_leaves': np.arange(10, 50, 5).tolist(),
        'learning_rate': np.arange(.01,.3,.05).tolist(),
        'max_depth': np.arange(5, 20, 3).tolist(),
        'subsample': np.arange(.1,.8,.1).tolist(),
        'subsample_freq': np.arange(1,5,1).tolist(),
        'min_data_per_group': np.arange(1,50,5).tolist(),
        'scale_pos_weight': np.arange(.001,.5,.01).tolist(),
        'colsample_bytree': np.arange(.1,.9,.1).tolist(),
        'reg_lambda': np.arange(.1, 1, .1).tolist()
    }

    # additional model fit params
    fit_params={"early_stopping_rounds":20,
                "eval_metric" : "auc",  
                "eval_set" : [(X_valid, y_valid.values.ravel())],
                "verbose" : 0
               }

    # lgbm model, static params
    lgbm = LGBMClassifier(
            objective='binary',
            verbose =-1,
            seed=123,
            silent=True,
            metric='roc_auc'
    )

    # create folds, run search
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state=123)

    grid = RandomizedSearchCV(estimator=lgbm, param_distributions=params,
                                n_jobs=-1, n_iter=n_iter, cv=skf.split(X_train,y_train.values.ravel()), verbose=1, random_state=123)#,cv=3, verbose=1, random_state=123

    grid.fit(X_train, y_train.values.ravel(), **fit_params)

    return grid.best_params_


def predict(X_train, 
            y_train, 
            X_valid, 
            y_valid, 
            X_test, 
            params):

    """Undersamples majority and synthetic minority samples using SMOTE

    Params
    --------
        X_train (df): dataframe with undersampled majority/SMOTE minority
        y_train (df): dataframe with undersampled majority/SMOTE minority
        X_valid (df): unmodified dataframe for validation
        y_valid (df): unmodified dataframe for validation
        X_test (df): unmodified dataframe used to create final predictions
        params (params): params used to train final LGBM model


    Returns
    --------
        preds (df): dataframe with classification probabilities for each image in test folder
        filtered_df (df): dataframe with summary statistics
    """
    # initialize, run, create predictions
    lgb_model = LGBMClassifier(**params)
    lgb_reg = lgb_model.fit(X_train,y_train.values.ravel())
    valid_preds = lgb_reg.predict(X_valid)
    preds = lgb_reg.predict_proba(X_test)

    # create summary metrics report (precision, recall, f1-score) + roc_auc
    report = metrics.classification_report(y_valid, valid_preds, labels=[0,1], output_dict=True)

    filtered_report = dict((k, report[k]) for k in ['0','1'] if k in report)
    filtered_df = pd.DataFrame.from_dict(filtered_report, orient='index')
    filtered_df['roc_auc'] = roc_auc_score(y_valid, valid_preds, average='weighted')

    return preds, filtered_df




