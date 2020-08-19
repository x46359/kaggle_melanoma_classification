import numpy as np
import pandas as pd
import os
import time
import csv
from PIL import Image
from paths import *
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from timeit import default_timer as timer

# the train function below is borrowed heavily (with minor modifications)
# from WillKoehrsen, jupyter notebook here https://github.com/WillKoehrsen/pytorch_challenge/blob/master/Transfer%20Learning%20in%20PyTorch.ipynb
# model train/eval function
def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_path,
          final_file_path,
          mod,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2,
          cuda_device=0):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_path (str ending in '.pt'): file path to save the model info
        final_file_path (str): file path for optimum model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats
        cuda_device (int): if more than one gpu available, select gpu to run training on
        mod (str): string of cnn model

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    start_epoch = model.epochs

    # Main loop
    for epoch in range(start_epoch, n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # f1 score train/valid
        y_true_train = np.array([])
        y_score_train = np.array([])
        y_true_valid = np.array([])
        y_score_valid = np.array([])

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            data, target = data.cuda(cuda_device), target.cuda(cuda_device)

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # predictions
            _, pred = torch.max(output, dim=1)
            
            # append label/predicted values for f1 scoring later
            y_true = target.cpu().numpy()
            y_score = pred.cpu().numpy()           
            
            y_true_train = np.append(y_true_train, y_true)
            y_score_train = np.append(y_score_train, y_score)

            # accuracy
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    data, target = data.cuda(cuda_device), target.cuda(cuda_device)

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # predictions
                    _, pred = torch.max(output, dim=1)

                    y_true = target.cpu().numpy()
                    y_score = pred.cpu().numpy()           
                    
                    y_true_valid = np.append(y_true_valid, y_true)
                    y_score_valid = np.append(y_score_valid, y_score)
                    
                    # Calculate validation accuracy
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # f1 scores
                f1_train = f1_score(y_true_train, y_score_train, average='weighted')
                f1_valid = f1_score(y_true_valid, y_score_valid, average='weighted')

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc, f1_train, f1_valid])

                # Save Model in case script needs to be run later
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, save_file_path)

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )
                    print(
                        f'\t\tTraining f1 score: {f1_train:.4f} \t Validation f1 score: {f1_valid:.4f}'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # # Save model

                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                }, final_file_path)

                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, save_file_path)
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=['train_loss', 'valid_loss', 'train_acc','valid_acc', 'f1_train', 'f1_valid'])
                        history['model'] = mod

                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start

    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'f1_train', 'f1_valid'])
    history['model'] = mod

    return model, history

def run_training(mod, 
                 cuda_device, 
                 size, 
                 train_loader, 
                 test_loader, 
                 n_epochs):

    """Setup model parameters based on inputs 
       and runs train function for various pre-trained models

    Params
    --------
        mod (str): pre-trained model
        cuda_device (int): if more than one gpu available, select gpu to run training on
        size (int): used to resize images. input of 225 will resize images to 225x225 pixels.
        train_loader (PyTorch dataloader): dataloader for training dataset with augmentations applied
        test_loader (PyTorch dataloader): dataloader for testing dataset 
        n_epochs (int): Number of epochs to train model

    Returns
    --------
        N/A
    """

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
    cuda_device=cuda_device

    # Move to gpu
    model = model.cuda(cuda_device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # file names
    save_file_name = mod + '_' + str(size) + '.pt'
    save_file_path = model_path/save_file_name
    final_file_path = model_path/str('final_' + save_file_name)
    # train_on_gpu = torch.cuda.is_available()

    # if .pt file already exists, load
    if os.path.isfile(save_file_path):
        checkpoint = torch.load(save_file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

        model.epochs = epoch   
    else:
        pass

    model, history = train(
        model,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        save_file_path=save_file_path,
        final_file_path= final_file_path,
        max_epochs_stop=3,
        n_epochs=n_epochs,
        print_every=1,
        cuda_device=cuda_device,
        mod=mod)

    with open(model_path/str('model_eval_'+ str(size) + '.csv'), 'a') as f:
        history.to_csv(f, header=f.tell()==0, index=False)

    with open(model_path/str('model_info_'+ str(size) + '.csv'), 'a') as fd:
        wr = csv.writer(fd)
        wr.writerow([str(model)])