# -*- coding: utf-8 -*-

#### Import packages
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import argparse
import os
import time
from os.path import join
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim
from torchvision import  transforms, models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

class Configuration:
  def __init__(self, args):
    self.experiment_name = "Training the Super Mario Segmentation Model"
    
    # Paramters for the first part
    self.pre_load    = "True" ## Load dataset in memory
    self.pre_trained = "True"
    self.num_classes = 6
    self.ignore_label = 255
    self.training_data_proportion = 0.8 # Proportion of images of the dataset to be used for training

    self.lr    = 0.001  # 0.001 if pretrained weights from pytorch. 0.1 if scratch
    self.epoch = args.epochs     # Play with this if training takes too long
    self.M = [37,42]         # If training from scratch, reduce learning rate at some point

    self.batch_size = args.batch_size  # Training batch size
    self.test_batch_size = 4  # Test batch size
    self.model_file_name = args.output_file
    
    self.dataset_root = args.dataset
    self.download   = False
    
    self.seed = 271828

class MarioDataset(data.Dataset):
    def __init__(self, args, mode, transform_input=transforms.ToTensor(), transform_mask=transforms.ToTensor()):
        self.args = args
        self.folder = args.dataset_root

        #If you change how you create the dataset you may need to modify this:
        self.images_in_dataset = len(os.listdir(self.folder+"/PNG"))
        training_images_no = int(self.images_in_dataset*0.8)
        self.imgs = np.arange(training_images_no) if mode == 'train' else np.arange(training_images_no,self.images_in_dataset)

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set or proportions')

        self.mode = mode
        self.transform_input = transform_input
        self.transform_mask = transform_mask

    # Default trasnformations on train data
    def transform(self, image, mask):

        i, j, h, w = transforms.RandomCrop.get_params(image, (224,224))
        
        image = TF.crop(image,i,j,h,w)
        mask  = TF.crop(mask,i,j,h,w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        return image, mask

    # Default trasnformations on test data
    def test_transform(self, image, mask):
        #224x224 center crop: 
        image = TF.center_crop(image,[224,224])
        mask  = TF.center_crop(mask,[224,224])

        return image, mask
    
    def __getitem__(self, index):

        if self.mode == 'test':
            img = Image.open(+self.folder+"/PNG/"+str(index)+".png").convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return str(index), img

            #Load RGB image
        img = Image.open(self.folder+"/PNG/"+str(index)+".png").convert('RGB')

        if self.mode == 'train':

            #Load class mask
            mask = Image.open(self.folder+"/Labels/"+str(index)+".png")
        else:
            mask = Image.open(self.folder+"/Labels/"+str(index)+".png")

            ##Transform using default transformations
        if self.mode=="train":
              img, mask = self.transform(img,mask)
        else:
              img, mask = self.test_transform(img,mask)

        if self.transform_input is not None:
           img = self.transform_input(img)
        if self.transform_mask is not None:
            mask = 255*self.transform_mask(mask)

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


def train_epoch(args, model, device, train_loader, optimizer, epoch):
    # switch to train mode
    model.train()

    train_loss = []
    counter = 1

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    for _, (images, mask) in enumerate(train_loader):

        images, mask = images.to(device), mask.to(device)

        outputs = model(images)['out']
 
        #Aggregated per-pixel loss
        loss = criterion(outputs, mask.squeeze(1))
        train_loss.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, int(counter * len(images)), len(train_loader.dataset),
                100. * counter / len(train_loader), loss.item(),
                optimizer.param_groups[0]['lr']))
        counter = counter + 1
    
    return sum(train_loss) / len(train_loss) # per batch averaged loss for the current epoch.

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def testing(args, model, device, test_loader):

    model.eval()

    loss_per_batch = []

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    gts_all, predictions_all = [], []
    with torch.no_grad():
        for _, (images, mask) in enumerate(test_loader):

            images, mask = images.to(device), mask.to(device)

            outputs = model(images)['out']

            loss = criterion(outputs,mask.squeeze(1))
            loss_per_batch.append(loss.item())

            # Adapt output size for histogram calculation.
            preds = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            gts_all.append(mask.data.squeeze(0).cpu().numpy())
            predictions_all.append(preds)

    loss_per_epoch = [np.average(loss_per_batch)]

    hist = np.zeros((args.num_classes, args.num_classes))
    for lp, lt in zip(predictions_all, gts_all):
        hist += _fast_hist(lp.flatten(), lt.flatten(), args.num_classes)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))


    mean_iou = np.nanmean(iou)

    print('\nTest set ({:.0f}): Average loss: {:.4f}, mIoU: {:.4f}\n'.format(
        len(test_loader.dataset), loss_per_epoch[-1], mean_iou))

    return (loss_per_epoch, mean_iou)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-o","--output_file",default="MarioSegmentationModel.pth",type=str, help="Name of the model. Will be saved on ./models")
    parser.add_argument("-bs","--batch_size",default=4,type=int, help="Keep it always 2 or more, otherwise it will crash.") ## 
    parser.add_argument("-d","--dataset",required=True, help="Path to dataset",type=str)
    parser.add_argument("-e","--epochs",default=45,type=int,help="Epochs")

    args = parser.parse_args()

    assert args.batch_size != 1, "Please use batch size bigger than 1."

    ## Create arguments object
    args = Configuration(args)
    device = 'cuda'

    # Set random seed for reproducibility
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation
    np.random.seed(args.seed)

    workers = 0 #Anything over 0 will crash on windows. On linux it should work fine.

    trainset = MarioDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    testset = MarioDataset(args, 'val')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    model = models.segmentation.deeplabv3_resnet50(
            pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, 6)

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M, gamma=0.1)

    loss_train_epoch = []
    loss_test_epoch = []
    acc_train_per_epoch = []
    acc_test_per_epoch = []
    new_labels = []

    cont = 0

    if not os.path.isdir("./models/"):
        os.makedirs("models")

    for epoch in tqdm(range(1, args.epoch + 1),desc = "DeepLabV3_Resnet50 training, epoch"):
        st = time.time()
        loss_per_epoch = train_epoch(args,model,device,train_loader,optimizer,scheduler)

        loss_train_epoch += [loss_per_epoch]

        scheduler.step()

        loss_per_epoch_test, acc_val_per_epoch_i = testing(args,model,device,test_loader)

        loss_test_epoch += loss_per_epoch_test
        acc_test_per_epoch += [acc_val_per_epoch_i]

        if epoch == 1:
            best_acc_val = acc_val_per_epoch_i
            
        else:
            if acc_val_per_epoch_i > best_acc_val:
                best_acc_val = acc_val_per_epoch_i

        if epoch==args.epoch:
            torch.save(model.state_dict(), "./models/"+args.model_file_name)

        

        cont += 1

    ##Accuracy
    acc_test  = np.asarray(acc_test_per_epoch)

    #Loss per epoch
    loss_test  = np.asarray(loss_test_epoch)
    loss_train = np.asarray(loss_train_epoch)

    numEpochs = len(acc_test)
    epochs = range(numEpochs)

    plt.figure(2)
    plt.plot(epochs, loss_test, label='Test Semantic, min loss: ' + str(np.min(loss_test)))
    plt.plot(epochs, loss_train, label='Train Semantic, min loss: ' + str(np.min(loss_train)))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.show()


