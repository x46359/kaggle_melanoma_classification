import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from os import walk
from torch.utils.data import Dataset
from PIL import Image
from paths import *

class CustomDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.X = df.iloc[:,0]
        self.y = df.iloc[:,1]
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        img_name = self.X.iloc[i] + '.jpg'
        img_path = data_train/img_name
        img = Image.open(img_path)
        
        if self.transforms:
            data = self.transforms(img)

            return (data, self.y[i])

def make_data(sample_test=0,
              size=225,
              test_train_split_percent=0.8,
              batch_size=6):
    """Create pytorch dataloaders

    Params
    --------
        sample_test (int): used to create subset data for code testing. default is 0 (running all data)
        size (int): used to resize images. input of 225 will resize images to 225x225 pixels.
        test_train_split_percent (float): percentage of data to separate for validation
        batch_size (int): number of images to run in batch. needs to be adjusted for OOM error 

    Returns
    --------
        train_loader (PyTorch dataloader): dataloader for training dataset with augmentations applied
        test_loader (PyTorch dataloader): dataloader for testing dataset 
    """

    # list of all pictures in train/test directory
    ftr = []
    for (dirpath, dirnames, filenames) in walk(data_train):
        ftr.extend(filenames)
        break

    fte = []
    for (dirpath, dirnames, filenames) in walk(data_test):
        fte.extend(filenames)
        break

    # read train csv file, keep relevant columns
    df = pd.read_csv(data_raw_path/'train.csv')
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)[['image_name','target']]

    # subset data if requested
    if sample_test > 0:
        df = df[:sample_test]
        print(f'\nSubsetting data for {sample_test} samples')
    else:
        print(f'\nRunning full data set')

    # create train test split
    separation = int(df.shape[0] * test_train_split_percent)
    tr = df[:separation]
    te = df[separation:].reset_index(drop=True)

    # create augmentation for the training. 
    # augment only the target==1 since they are so limited
    tr_aug = tr[tr.target==1].reset_index(drop=True)
    
    if tr_aug.shape[0] == 0:
        print(f'\nDataset does not have a malignant instance, please increase sample size to > {sample_test}')
        exit()
        
    ratio = tr[tr.target==0].shape[0]/tr_aug.shape[0] 

    # transformations applied to augment dataset
    transformations_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=90),
        # transforms.ColorJitter(),
        transforms.RandomVerticalFlip(),
        transforms.Resize([size,size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # transformations for dataset that doesn't require augmentation
    transformations_no_aug = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # loop through based on ratio (above) and create training dataset
    # first aug_df doesn't need transformations
    tr_no_aug_df = CustomDataset(tr, transforms=transformations_no_aug)
    tr_aug_df = CustomDataset(tr_aug, transforms=transformations_no_aug)
    tr_df = tr_no_aug_df + tr_aug_df

    for i in range(int(ratio/10)):
        tr_aug_df_new = CustomDataset(tr_aug, transforms=transformations_aug)
        tr_df = tr_df + tr_aug_df_new

    # train/test set
    tr_df = CustomDataset(tr, transforms=transformations_no_aug)
    te_df = CustomDataset(te, transforms=transformations_no_aug)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(tr_df, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(te_df, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader