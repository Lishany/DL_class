__author__ = 'lishanyu'

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import torch
from torch.utils.data import TensorDataset,Dataset,DataLoader
import torch.nn as nn
from torch.autograd import Variable
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import gaussian
import torchvision.transforms as transforms
from scipy import ndimage
from torchvision import models
import torch.nn.functional as F
import math
import random


server = 0## whether run it in server 0: no; 1 yes

'''
augment = (random.random()>0.4)
if augment:
    print("DO Augment, and type is ")
else:
    print("NOT Augment")

if augment:
    augment_type = random.choice([1,2,3])
    if augment_type == 1:
        print("horizon")
    else:
        if augment_type == 2:
            print("horizon+shift")
        else:
            print("horizon+shift+rotate")
'''

#ttemp1 = random.sample([32, 64], 1)
#ttemp2 = random.sample([64, 128], 1)
num_neuro = [32,64,64,128]
fc_num = 64
print("4CNN num of neuros",num_neuro,fc_num)
print("fc activation is Sigmoid, output activation is Sigmoid")

'''
acti_type = 3
if acti_type==1:
    print("fc activation is ReLU")
else:
    if acti_type == 2:
        print("fc activation is Tanh")
    else:
        print("fc activation is Sigmoid")
'''
'''
output_type = random.choice([1,2])
if output_type == 1:
    print("output sigmoid")
else:
    print("out put tanh+1.2")
'''
BATCH_SIZE = 32  ## minibatch size
print("batch size is ",BATCH_SIZE)

#print("GREY, gaussian,tv_denoise")

print("============")
if server == 0:
    import first_HW_dl_zhu.data_pre as gendata
    datapath= '/Users/lishanyu/Desktop/kaggle/iceberg/data/processed/'   ## the root path of train.json in computer
    savepath = "/Users/lishanyu/Desktop/DeepLearning_MichaelZhu/"
else:
    import outer_pythoncode.data_pre as gendata
    datapath= '/data/user45/data_iceberg/'  ## the root path of train.json in server
    savepath = "/data/user45/hw1_zhu/"

NUM_EPOCH = 40  ## the number of epoch

val_share = 0.2  ## the ratio of validation data over all data
n_channel = 2  ## the number of channel of input data
kkk = 5 #几步以内最小  validation score will be minimal within 12 steps, a criterion of stopping
lr = 0.001


def denoise(X, weight, multichannel):
    return np.asarray([denoise_tv_chambolle(item, weight=weight, multichannel=multichannel) for item in X])

def smooth(X, sigma):
    return np.asarray([gaussian(item, sigma=sigma) for item in X])

'''
class RandomShift(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if np.random.random() < 0.3:
            image=ndimage.shift(image,(0,np.random.randint(-3,3),np.random.randint(-3,3)),mode='wrap')
        return {'image': image, 'labels': labels}

class RandomRotate(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if np.random.random() < 0.3:
            image=ndimage.rotate(image,axes =(2,1),angle = np.random.randint(-8,8),mode="wrap")
            if len(image[0])>75:
                if len(image[0])==76:
                    image = image[:,1:,1:]
                else:
                    if (len(image[0])-75)%2 ==0:
                        nn = (len(image[0])-75)//2
                        image = image[:,nn:-nn,nn:-nn]
                    else:
                        nn = (len(image[0])-76)//2
                        image = image[:,(nn+1):-nn,(nn+1):-nn]
        return {'image': image, 'labels': labels}

class RandomZoom(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if np.random.random() < 0.3:
            image=ndimage.zoom()
        return {'image': image, 'labels': labels}
'''
class RandomZoom(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if np.random.random() < 0.3:
            image=ndimage.zoom()
        return {'image': image, 'labels': labels}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if np.random.random() < 0.3:
            image=np.flip(image,2)

        return {'image': image, 'labels': labels}

class Identityy(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        return {'image': image, 'labels': labels}

class ToTensor(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = F.to_tensor(image)
        return {'image': image, 'labels': labels}

'''
if acti_type==1:
    our_acti = nn.ReLU
else:
    if acti_type == 2:
        our_acti = nn.Tanh
    else:
        our_acti = nn.Sigmoid
'''

## Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channel, num_neuro[0], kernel_size=3, stride=1, padding=0, bias = False),
            nn.BatchNorm2d(num_neuro[0]),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_neuro[0], num_neuro[1], kernel_size=3, stride=1, padding=0, bias = False),
            nn.BatchNorm2d(num_neuro[1]),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_neuro[1], num_neuro[2], kernel_size=3, stride=1, padding=0, bias = False),
            nn.BatchNorm2d(num_neuro[2]),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.3))

        self.conv4 = nn.Sequential(
            nn.Conv2d(num_neuro[2], num_neuro[3], kernel_size=3, stride=1, padding=0, bias = False),
            nn.BatchNorm2d(num_neuro[3]),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.3))
        self.linear = nn.Sequential(
            nn.Linear(4*num_neuro[3],fc_num),
            nn.Dropout(0.2),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(fc_num, 1),
        )
        self.sig = nn.Sigmoid()
        #self.tan = nn.Tanh()
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = self.pool4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.fc(out)
        out = self.sig(out)
        '''
        if output_type == 1:
            out = self.sig(out)
        else:
            out = (self.tan(out) + 1) / 2
        '''
        #print(out)
        return out


model = CNNModel()
print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


## Augmentation or random transformation
class read_data(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data= data
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx,:,:,:], 'labels': np.asarray([self.labels[idx]])}
        if self.transform:
            sample = self.transform(sample)
            ttemp = sample['image']
            bband1 = gaussian(denoise_tv_chambolle(ttemp[0], weight=0.05, multichannel=False), 0.5)
            bband2 = gaussian(denoise_tv_chambolle(ttemp[1], weight=0.05, multichannel=False), 0.5)
            sample['image']=np.stack([bband1,bband2],axis = 0)
        return sample



criterion = nn.BCELoss()

all_losses = []
val_losses = []


if __name__ == '__main__':

    data = pd.read_json(datapath + "train.json")
    #r = random.random
    #random.seed(2)
    #data = shuffle(data,random=r)  # otherwise same validation set each time!
    #data = data.reindex(np.random.permutation(data.index))
    data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

    band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
    band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
    train_targets = np.array(data['is_iceberg'].values, dtype=np.float32)
    #band_1, band_2, train_targets = shuffle(band_1, band_2, train_targets, random_state=271)
    band_1, band_2, train_targets = shuffle(band_1, band_2, train_targets)

    val_offset = int(band_1.shape[0] * (1 - val_share))
    print("Offest:" + str(val_offset))
    train_band1 = band_1[:val_offset]
    train_band2 = band_2[:val_offset]
    train_tg = train_targets[:val_offset]

    val_band1 = band_1[val_offset:]
    val_band2 = band_2[val_offset:]
    val_tg = train_targets[val_offset:]

    ## Preparing training data
    full_img_tr = np.stack([train_band1, train_band2], axis=1)
    '''
    if augment:
        if augment_type == 1:
            train_dataset = read_data(data=full_img_tr, labels=train_tg,  transform=transforms.Compose([
                RandomHorizontalFlip()
                # ,RandomRotate()
                # ,RandomShift()
            ]))
        else:
            if augment_type == 2:
                train_dataset = read_data(data=full_img_tr, labels=train_tg, transform=transforms.Compose([
                    RandomHorizontalFlip()
                    # ,RandomRotate()
                    ,RandomShift()
                ]))
            else:
                train_dataset = read_data(data=full_img_tr, labels=train_tg, transform=transforms.Compose([
                    RandomHorizontalFlip()
                    ,RandomShift()
                    ,RandomRotate()
                ]))
        
    else:
        train_dataset = read_data(data=full_img_tr, labels=train_tg,  transform=transforms.Compose([Identityy()]))
    '''
    train_dataset = read_data(data=full_img_tr, labels=train_tg, transform=transforms.Compose([
        RandomHorizontalFlip()
        # ,RandomRotate()
        # ,RandomShift()
    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    ## Preparing validation data
    val_img_tr = np.stack([val_band1, val_band2], axis=1)
    val_dataset = read_data(data=val_img_tr, labels=val_tg,transform=transforms.Compose([Identityy()]))
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


    best_model = model
    best_val = 100
    cur_step = 0
    best_step = 0

    for epoch in range(NUM_EPOCH):
        #print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCH))
        #print('*' * 5 + ':')
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for i, data in enumerate(train_loader, 1):##此处i从1开始

            img, label = data['image'].float(),data['labels'].float()
            img, label = Variable(img), Variable(label)  # RuntimeError: expected CPU tensor (got CUDA tensor)
            out = model(img)
            label = label.float()
            loss = criterion(out, label)
            running_loss += loss.data[0] * label.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if i % 10 == 0:
                #temp = running_loss / (BATCH_SIZE * i-BATCH_SIZE+label.size(0))
                #all_losses.append(temp)
                #print('[{}/{}] Loss: {:.6f}'.format(epoch + 1, NUM_EPOCH,temp))
            if i % 100 == 0:
                print("i is ",i)
        print('Finish {} epoch, Loss: {:.6f}'.format(epoch + 1, (running_loss / (BATCH_SIZE * i-BATCH_SIZE+label.size(0)))))

        model.eval()
        eval_loss = 0
        eval_acc = 0
        total_num = 0
        acu_num = 0
        for i, data in enumerate(val_loader, 1):
            img, label = data['image'].float(),data['labels'].float()
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
            label = label.float()
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)
            total_num +=label.size(0)
            numlabel = out.data.numpy()
            numlabel[numlabel>0.5]=1
            numlabel[numlabel<0.5] = 0
            temp = numlabel - label.data.numpy()
            acu_num = acu_num+sum(temp == 0)
        print(acu_num,total_num)
        cur_step += 1
        tmp = eval_loss / (total_num)
        if tmp < best_val:
            best_model = model
            best_step = cur_step
            best_val = tmp
        else:
            if best_step < cur_step - kkk and best_val<0.15:
                print("best val = ",best_val)
                print("cur_step = ",cur_step)
                print('VALIDATION Loss: {:.6f}'.format(eval_loss / (total_num)))
                val_losses.append(tmp)
                print("break")
                break
        print('VALIDATION Loss: {:.6f}'.format(eval_loss / (total_num)))
        val_losses.append(tmp)
        if epoch == 20:
            if best_val > 0.24:
                break
    print("BEST VAL is ",best_val)
    if best_val < 0.14:
        temp_img = []
        temp_label = []
        wrong_loss = []
        total_num = 0
        acu_num = 0
        for i, data in enumerate(val_loader, 1):
            img, label = data['image'].float(),data['labels'].float()
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
            label = label.float()
            out = best_model(img)
            loss = criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)
            total_num +=label.size(0)
            numlabel = out.data.numpy()
            numlabel[numlabel>0.5]=1
            numlabel[numlabel<0.5] = 0
            temp = numlabel - label.data.numpy()
            acu_num = acu_num+sum(temp == 0)
            id = np.where(temp != 0)[0]
            temp_img.append(img.data.numpy()[id])
            temp_label.append(label.data.numpy()[id])
            #wrong_loss.append(((1-out)*(1-label)+(out)*(label)).data.numpy()[id])
            wrong_loss.append(out.data.numpy()[id])
        print(acu_num,total_num)
        a = np.vstack(temp_img)
        b = np.squeeze(np.vstack(temp_label))
        c = np.squeeze(np.vstack(wrong_loss))
        np.save(savepath+str(best_val)+"filename_img.npy", a)
        np.save(savepath + str(best_val) + "filename_label.npy", b)
        np.save(savepath + str(best_val) + "filename_loss.npy", c)
        print("PRINT WRONG INFO")