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

augment = (random.random()>0.45)
ttypes = random.choice([1,2,3,4]) # raw grey
model_type = 1
    #random.choice([1,2,3,4]) # densenet cnn_bn  cnn_nobn cnn_nobn_multiconv

if augment:
    print("DO Augment")
else:
    print("NOT Augment")

if model_type==1:
    print("Densenet")
    grate = random.choice([8,10,12,16])# growth_rate
    print("grow_rate is ",grate)
    dense_num = random.sample([1,2,1,2,1], 1)+random.sample([1,2,3,4,1,2,3], 1)+[2]+random.sample([1,2,3,1,2], 1)
    print(dense_num)
else:
    print("4CNN")
    num_neuro = [64, 64, 128, 64]
    ttemp = random.sample([16, 32, 64], 1)
    ttemp1 = random.sample([32, 32, 64, 64, 128], 1)
    ttemp2 = random.sample([64, 128, 256, 64, 128, 64, 128, 256, 64, 128], 2)
    num_neuro = ttemp + ttemp1 + ttemp2
    print(num_neuro)
    if model_type == 2:
        print("cnn+bn")
    else:
        if model_type ==3:
            print("cnn+noBN")
        else:
            if model_type==4:
                print("cnn+noBN+multiConv")

BATCH_SIZE = random.choice([32,64])  ## minibatch size
print("batch size is ",BATCH_SIZE)

if ttypes==1:
    print("GREY, gaussian,tv_denoise")
if ttypes == 2:
    print("ZScore")
if ttypes == 3:
    print("Raw")
if ttypes == 4:
    print("zscore2,/max-min")

if server == 0:
    import first_HW_dl_zhu.data_pre as gendata
    datapath= '/Users/lishanyu/Desktop/kaggle/iceberg/data/processed/'   ## the root path of train.json in computer
else:
    import outer_pythoncode.data_pre as gendata
    datapath= '/data/user45/data_iceberg/'  ## the root path of train.json in server

NUM_EPOCH = 30  ## the number of epoch

val_share = 0.2  ## the ratio of validation data over all data
n_channel = 2  ## the number of channel of input data
kkk = 4 #几步以内最小  validation score will be minimal within 12 steps, a criterion of stopping
lr = 0.001


def denoise(X, weight, multichannel):
    return np.asarray([denoise_tv_chambolle(item, weight=weight, multichannel=multichannel) for item in X])

def smooth(X, sigma):
    return np.asarray([gaussian(item, sigma=sigma) for item in X])

def grey_feature(band_1,band_2, smooth_gray=0.5, weight_gray=0.05):#smooth_gray provides regularization.. large提供正则，但影响收敛
    band_1 = smooth(denoise(band_1, weight_gray, False), smooth_gray)
    print('Gray 1 done')
    band_2 = smooth(denoise(band_2, weight_gray, False), smooth_gray)
    print('Gray 2 done')
    nnum = 2
    return np.stack([band_1,band_2],axis = 1), nnum

def zscore(X):
    return((X-np.mean(X))/np.std(X))

def zscore2(X):
    return((X-np.mean(X))/(np.max(X)-np.min(X)))

class RandomShift(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if np.random.random() < 0.8:
            image=ndimage.shift(image,(0,np.random.randint(-3,3),np.random.randint(-3,3)),mode='wrap')
        return {'image': image, 'labels': labels}

class RandomRotate(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if np.random.random() < 0.7:
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
        if np.random.random() < 0.5:
            image=ndimage.zoom()
        return {'image': image, 'labels': labels}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if np.random.random() < 0.5:
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

        self.fc = nn.Sequential(
            nn.Linear(4*num_neuro[3],64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64,1),
        )
        self.sig = nn.Sigmoid()
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
        out = self.fc(out)
        out = self.sig(out)
        return out

class CNNModel_noBN(nn.Module):
    def __init__(self):
        super(CNNModel_noBN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channel, num_neuro[0], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_neuro[0], num_neuro[1], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_neuro[1], num_neuro[2], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.3))
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_neuro[2], num_neuro[3], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.3))

        self.fc = nn.Sequential(
            nn.Linear(4*num_neuro[3],64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64,1),
        )
        self.sig = nn.Sigmoid()
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
        out = self.fc(out)
        out = self.sig(out)
        return out

class CNNModel_noBN_mulconv(nn.Module):
    def __init__(self):
        super(CNNModel_noBN_mulconv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channel, num_neuro[0], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_neuro[0], num_neuro[0], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_neuro[0], num_neuro[1], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_neuro[1], num_neuro[1], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_neuro[1], num_neuro[2], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.3))
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_neuro[2], num_neuro[3], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Dropout(0.3))

        self.fc = nn.Sequential(
            nn.Linear(4*num_neuro[3],64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64,1),
        )
        self.sig = nn.Sigmoid()
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
        out = self.fc(out)
        out = self.sig(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

n_channels = 2


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=1):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = n_channels * growth_rate
        self.conv1 = nn.Conv2d(n_channels, num_planes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])

        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)

        num_planes = out_planes
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])

        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        '''
        num_planes = out_planes
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])

        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        '''
        num_planes = out_planes
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])

        num_planes += nblocks[3] * growth_rate
        self.bn = nn.BatchNorm2d(num_planes)

        self.linear =nn.Sequential(
            nn.Linear((2+nblocks[0]+2*nblocks[1]+4*nblocks[3])*growth_rate*4, 64),##num_out*16*growth_rate
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        #self.linear = nn.Linear(3328, num_classes)

        self.sig = nn.Sigmoid()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        #print('ooout',out.size())
        out = self.trans2(self.dense2(out))
        #print('ooout2',out.size())
        #out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        #print('ooout3',out.size())
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        #print("out",out.data.shape)
        out = self.linear(out)
        #print('ooout',out.size())
        out = self.sig(out)
        return out

def DenseNet_yu():
    return DenseNet(Bottleneck, dense_num, growth_rate=grate)##注意，少一层，第三个参数无效的。


if model_type ==1:
    model = DenseNet_yu()
    print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
else:
    if model_type == 2:
        model = CNNModel()
        print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        if model_type == 3:
            model = CNNModel_noBN()
            print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            if model_type == 4:
                model = CNNModel_noBN_mulconv()
                print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)


print(model)





## Augmentation or random transformation
class read_data(Dataset):
    def __init__(self, data, labels, types, transform=None):
        self.data= data
        self.labels = labels
        self.transform = transform
        self.types = types
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx,:,:,:], 'labels': np.asarray([self.labels[idx]])}
        if self.transform:
            sample = self.transform(sample)
            ttemp = sample['image']

            if self.types==1:
                bband1 = gaussian(denoise_tv_chambolle(ttemp[0], weight=0.05, multichannel=False), 0.5)
                bband2 = gaussian(denoise_tv_chambolle(ttemp[1], weight=0.05, multichannel=False), 0.5)
            else:
                if self.types==2:
                    bband1 = zscore(ttemp[0])
                    bband2 = zscore(ttemp[1])
                else:
                    if self.types==3:
                        bband1 = ttemp[0]
                        bband2 = ttemp[1]
                    else:
                        if self.types == 4:
                            bband1 = zscore2(ttemp[0])
                            bband2 = zscore2(ttemp[1])

            sample['image']=np.stack([bband1,bband2],axis = 0)
        return sample



criterion = nn.BCELoss()

all_losses = []
val_losses = []


if __name__ == '__main__':

    data = pd.read_json(datapath + "train.json")
    data = shuffle(data)  # otherwise same validation set each time!
    data = data.reindex(np.random.permutation(data.index))
    data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

    band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
    band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
    train_targets = np.array(data['is_iceberg'].values, dtype=np.float32)
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

    if augment:
        train_dataset = read_data(data=full_img_tr, labels=train_tg, types= ttypes, transform=transforms.Compose([
            RandomHorizontalFlip()
            # ,RandomRotate()
            # ,RandomShift()
            # ,transforms.RandomRotation(5)
        ]))
    else:
        train_dataset = read_data(data=full_img_tr, labels=train_tg, types= ttypes, transform=transforms.Compose([Identityy()]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    ## Preparing validation data
    val_img_tr = np.stack([val_band1, val_band2], axis=1)
    val_dataset = read_data(data=val_img_tr, labels=val_tg,types= ttypes, transform=transforms.Compose([Identityy()]))
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
        for i, data in enumerate(val_loader, 1):
            img, label = data['image'].float(),data['labels'].float()
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
            label = label.float()
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)
            total_num +=label.size(0)
        cur_step += 1
        tmp = eval_loss / (total_num)
        if tmp < best_val:
            best_model = model
            best_step = cur_step
            best_val = tmp
        else:
            if best_step < cur_step - kkk and best_val<0.20:
                print("best val = ",best_val)
                print("cur_step = ",cur_step)
                print('VALIDATION Loss: {:.6f}'.format(eval_loss / (total_num)))
                val_losses.append(tmp)
                print("break")
                break
        print('VALIDATION Loss: {:.6f}'.format(eval_loss / (total_num)))
        val_losses.append(tmp)