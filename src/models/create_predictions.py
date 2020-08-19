import numpy as np
import pandas as pd
import os
import time
from paths import *
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch import optim
from torchvision import models

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, df, image_path, transforms=None):
        self.X = df
        self.transforms = transforms
        self.image_path = image_path

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        img_name = self.X.iloc[i] + '.jpg'
        img_path = self.image_path/img_name
        img = Image.open(img_path)
        
        if self.transforms:
            data = self.transforms(img)

            return data

# Predict probability function
def predict(data, model):
    with torch.no_grad():
        model.eval()
        out = model(data)
        prob = nn.functional.softmax(out, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)
        return top_p

def run_predictions(train_test, 
                    img_size, 
                    batch_size, 
                    gpu, 
                    mod,
                    sample_test):

    """Setup model parameters based on inputs and runs predict function
       Saves outputs as csv files for use in downstream LightGBM ensemble model

    Params
    --------
        train_test (str): inputs are either 'train' or 'test, runs predictions for test or train dataset. 
        img_size (int): used to resize images. input of 225 will resize images to 225x225 pixels.
        gpu (int): if more than one gpu available, select gpu to run training on
        mod (str): pre-trained model
        batch_size (int): number of images to run in batch. needs to be adjusted for OOM error 
        sample_test (int): used to create subset data for code testing. default is 0 (running all data)

    Returns
    --------
        N/A
    """

    data_interim_pred_path = data_interim_path/'interim_predictions'/train_test

    start_time = time.time()

    # read test/train csv
    df = pd.read_csv(data_raw_path/str(train_test + '.csv'))
    df = df['image_name']

    # subset data if requested
    if sample_test > 0:
        df = df[:sample_test]
        print(f'\nSubsetting prediction data for {sample_test} samples')
    else:
        print(f'\nRunning full data set for predictions')

    # transformations
    transformations_no_aug = transforms.Compose([
        transforms.Resize([img_size,img_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # dataset
    path = eval(str('data_' + train_test))
    data_df = CustomDataset(df, image_path=path, transforms=transformations_no_aug)

    # dataloaders
    data_loader = torch.utils.data.DataLoader(data_df, batch_size=batch_size)

    # read in pre-trained models
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    googlenet = models.googlenet(pretrained=True)
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)

    # set model to mod
    model = eval(mod)
    
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # define linear/classifier layers.
    # refer to jup_notebook folder for details. This was largely manual.
    if mod == 'resnet18':
        model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    if mod == 'alexnet':
        model.classifier[6] = nn.Sequential(
                            nn.Linear(4096, 256), 
                            nn.ReLU(), 
                            nn.Dropout(0.4),
                            nn.Linear(256, 2),                   
                            nn.LogSoftmax(dim=1))
    if mod == 'vgg16':
        model.classifier[6] =  nn.Sequential(
                            nn.Linear(4096, 256), 
                            nn.ReLU(), 
                            nn.Dropout(0.4),
                            nn.Linear(256, 2),                   
                            nn.LogSoftmax(dim=1))    
    if mod == 'densenet':
        model.classifier = nn.Linear(in_features=2208, out_features=2, bias=True)
    if mod == 'googlenet':
        model.fc = nn.Linear(in_features=1024, out_features=2, bias=True)
    if mod == 'shufflenet':
        model.fc = nn.Linear(in_features=1024, out_features=2, bias=True) 
    if mod == 'resnext50_32x4d':
        model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    if mod == 'wide_resnet50_2':
        model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

    # set gpu
    cuda_device=gpu

    # Move to gpu
    model = model.cuda(cuda_device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # file names
    save_file_name = mod + '_' + str(img_size) + '.pt'
    save_file_path = model_path/save_file_name
    final_file_path = model_path/str('final_' + save_file_name)

    # if .pt file already exists, load
    if os.path.isfile(final_file_path):
        checkpoint = torch.load(final_file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    else:
        pass

    # list of final_predictions
    final_predictions = []

    # total iterations (based on batch size) and counter initialization
    total_it = int(data_df.X.size/batch_size)
    counter = 0

    # loop through data_loader
    for i, data in enumerate(data_loader):
        data = data.cuda(cuda_device)
        output = predict(data,model)
        final_predictions.append(output.cpu().numpy())
        print(f'Running iteration {counter} of {total_it} for {save_file_name}', end='\r')
        counter += 1

    # final_predictions to series, join with image names, set/round target probabilities
    final_predictions = pd.Series(np.concatenate(final_predictions).ravel())

    submission = pd.concat([data_df.X, final_predictions], axis=1).reset_index(drop=True)
    submission = submission.rename(columns={0: "target"})
    submission['target'] = (1 - submission.target)#.round(1)

    # create final csv
    file_list = [mod, str(img_size), train_test]
    file_name = data_interim_pred_path/str(mod + '_' + str(img_size) + '_' + train_test + '.csv')
    submission.to_csv(file_name, index=False)

    print("--- %s seconds ---" % (time.time() - start_time))