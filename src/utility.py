import pandas as pd
import numpy as np
import os, os.path
from paths import *


# csv file to append various model runs to
def create_csv(path):
    if os.path.exists(path):
        print(f'\n{path} already exists')
    else:
        print(f'\ncreating file {path}')
        with open(path, "w") as empty:
            pass

def check_raw_data(path):
    # count number of files in folder
    num_of_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

    # test or train folder
    test_train = (str(path).split("\\")[-1])

    # check path exists
    if os.path.exists(path):
        pass
    else:
        print(f'\n\nPlease download data from https://www.kaggle.com/c/siim-isic-melanoma-classification/data')
        exit()        

    # check number of files for test/train
    if test_train=='train':
        if num_of_files==33126:
            pass
        else:
            print(f'\n\nMissing {33126 - num_of_files} image from {test_train}ing data')
            print(f'Please download data from https://www.kaggle.com/c/siim-isic-melanoma-classification/data')
            exit()    
    else:
        if num_of_files==10982:
            pass
        else:
            print(f'\n\nMissing {10982 - num_of_files} image from {test_train}ing data')
            print(f'Please download data from https://www.kaggle.com/c/siim-isic-melanoma-classification/data')
            exit()    
