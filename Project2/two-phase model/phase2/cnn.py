
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

save = True
serve = False
BATCH_SIZE = 1024*8
n_epoches = 50
jieba = True
mink = 30
usermink = 5
offset = 0.92
isval = False
attk = 3
attk2 = 3
attk3 = 3
hidden_dim3 = 8
hidden_dim4 = 4
TIMEDIM = 33
ispre = True
file_num = 5
ffnum = [4,5,6,7]

import datapreprocess_jieba as dp
basepath = '/home/syugroup/LishanYu/hw2/'
jiebapath = basepath+"jieba_frequency.txt"
#userpath = basepath+"frequency_userID.txt"
#userpath = basepath+"userID_stat.txt"
userpath = basepath+"user_ana.txt"
txtName = basepath + 'phase2/resultmlpmodel_file'+str(file_num)+'pluscnn.txt'
class Loss_3(nn.Module):
    def forward(self, input_tensor,label):
        #input_tensor = (input_tensor**2)
        temp=label+Variable(torch.ones(label.size()[0], 3).cuda() * torch.FloatTensor([5, 3, 3]).cuda())
        temp = temp*Variable(torch.FloatTensor([2,4,4]).cuda())

        out = torch.abs(input_tensor-label)
        out = out/temp
        out = 1-torch.sum(out,1)
        total = torch.sum(label,1)+1
        total[total>100]=101
        id = torch.arange(0,out.size()[0]).long().cuda()[out.data<0.8]
        if id.dim()==0:
            return(1-torch.sum(out)/label.size()[0],Variable(torch.Tensor([1.0]).cuda()))
        out2 = (1-out[id])*total[id]
        #tmp = 1-torch.sum(total[id])/torch.sum(total)
        out3 = torch.abs(input_tensor.round()-label)
        out3 = out3/temp
        out3 = 1-torch.sum(out3,1)
        id2 = torch.arange(0,out3.size()[0]).long().cuda()[out3.data<0.8]
        if id2.dim()==0:
            return(1-torch.sum(out)/label.size()[0],Variable(torch.Tensor([1.0]).cuda()))
        tmp = 1-torch.sum(total[id2])/torch.sum(total)
        return(torch.sum(out2)/torch.sum(total[id]),tmp)

loss_pred = Loss_3()


import pandas as pd
import numpy as np

datapath = basepath + 'train_jeiba7/train_jeiba'+str(file_num)+'.json'

origin_train = pd.read_json(basepath + 'train_jeiba7/train_jeiba'+str(ffnum[0])+'.json',typ = 'frame')
i = 1
while i < len(ffnum):
    temp = pd.read_json(basepath + 'train_jeiba7/train_jeiba'+str(ffnum[i])+'.json',typ = 'frame')
    origin_train = origin_train.append(temp,ignore_index=True)
    i = i + 1


#train = pd.read_json(datapath,typ = 'frame')
if isval:
    time = origin_train['time']
    valid = []
    trainid = []
    for i in range(len(time)):
        if time[i][6] == '7':
            valid.append(i)
        else:
            trainid.append(i)
    train = origin_train.iloc[trainid]
    val = origin_train.iloc[valid]
else:
    train = origin_train

textdict = dp.text_jiebaDictionary(jiebapath, mink)  ## mink 以上
userdict = dp.userDictionary(userpath,usermink)


train_loader =  dp.TextClassDataLoadernotforrnn(train, userdict,textdict, batch_size=BATCH_SIZE)
maxlen = train_loader.max_length
print(maxlen)
if isval:
    val_loader = dp.TextClassDataLoadernotforrnn(val, userdict,textdict,maxlen = maxlen,batch_size=BATCH_SIZE)



class EDModel(nn.Module):
    def __init__(self, len_userdic,len_textdic,final_dim = 30,user_embedding_dim=50,enc_dim=40,text_embedding_dim = 50,time_embedding_dim = 10,num_layers = 1):
        super(EDModel, self).__init__()
        self.model_name = 'Encoder'
        self.user_embedding_dim = user_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.user_embeddings = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
        self.time_embeddings = nn.Linear(TIMEDIM,self.time_embedding_dim,bias = True)
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+self.time_embedding_dim,enc_dim),
                                nn.Dropout(0.3),nn.ReLU())
        self.cov1 = nn.Sequential(nn.Conv1d(1,out_channels=hidden_dim3,kernel_size=attk*self.text_embedding_dim,
            padding=int(attk/2)*self.text_embedding_dim,
            stride=self.text_embedding_dim),nn.Dropout(0.3),nn.ReLU())
        self.pool1 = nn.Sequential(nn.MaxPool1d(kernel_size = 2,stride = 2))
        self.cov2 = nn.Sequential(nn.Conv1d(hidden_dim3,hidden_dim4,
            kernel_size=attk2,padding=int(attk2/2)),nn.Dropout(0.3),nn.Tanh())
        self.pool2 = nn.Sequential(nn.MaxPool1d(kernel_size = 2,stride = 2))
        self.cov3 = nn.Sequential(nn.Conv1d(hidden_dim4,1,
            kernel_size=attk3,padding=int(attk3/2)),nn.Dropout(0.3),nn.Tanh())
        self.pool3 = nn.Sequential(nn.MaxPool1d(kernel_size = 2,stride = 2))
        self.mlp = nn.Sequential(nn.Linear(int(maxlen/8)+enc_dim+5+17,final_dim),
                                nn.Dropout(0.3),nn.ReLU())
        self.mlpcoe = nn.Sequential(nn.Linear(int(maxlen/8)+enc_dim+5+17,final_dim),
                                nn.Dropout(0.3),nn.Sigmoid())
        self.mlpuser = nn.Sequential(nn.Linear(int(maxlen/8)+enc_dim+5+17,3),
                                nn.Dropout(0.6),nn.ReLU())
    def forward(self, sen,user,time,seq_lengths,num4,textnum):
        user_embeds = self.user_embeddings(user)
        time_embeds = self.time_embeddings(time)
        input_embeds = torch.cat((user_embeds,time_embeds),1)
        enc_embeds = self.enc_mlp(input_embeds)###context important!!
        
        enc_embeds = torch.cat((enc_embeds,num4,textnum),1)
        text_embeds = self.text_embeddings(sen)
        sen_embeds = text_embeds.view((text_embeds.size()[0],-1))
        #print(sen_embeds.size())
        out = self.cov1(sen_embeds.unsqueeze(1))
        #print(out.size())
        out = self.pool1(out)
        #print(out.size())
        out = self.cov2(out)
        #print(out.size())
        out = self.pool2(out)
        #print(out.size())
        out = self.cov3(out)
        #print(out.size())
        out = self.pool3(out)
        #print(out.size())
        out = out.view((user.size()[0],-1))
        user_text = torch.cat((enc_embeds,out), 1)
        out_user2 = self.mlpuser(user_text)
        label3 = self.mlp(user_text)
        coe = self.mlpcoe(user_text)
        label3 = label3*coe
        label3 = label3.view((sen.size()[0],3,-1))
        label3 = torch.sum(label3,2)/10

        label3 = label3 + out_user2
        return label3


model = EDModel(userdict.__len__(),textdict.__len__()).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.008)
print(model)
valist = []
talist = []
tllist=[]
vllist=[]
for ep in range(n_epoches):
    print("epoch ",ep)
    all_loss = 0
    accuracy = 0
    model.train()
    for i, (bsen, seq_lengths, buser, btime, ll,num4,textnum) in enumerate(train_loader,1):
        # measure data loading time
        #data_time.update(time.time() - end)
        bsen = Variable(bsen)
        buser = Variable(buser)
        btime = Variable(btime)
        num4 = Variable(num4)
        textnum = Variable(textnum)
        blabel= Variable(ll.float())
        # compute output
        label3= model(bsen,buser,btime, seq_lengths,num4,textnum)
        loss,accu = loss_pred(label3,blabel)
        #print("acc:",1-loss2.data/label3.size()[0])
        #all_loss += 1-loss2.data/label3.size()[0]
        all_loss += loss.data
        accuracy += accu.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%5==0:
            print("loss ",all_loss.cpu().numpy().item()/5,"a acc ", accuracy.cpu().numpy().item() / 5)
            tllist.append(all_loss.cpu().numpy().item()/5)
            talist.append(accuracy.cpu().numpy().item()/5)
            all_loss = 0
            accuracy = 0
    if isval:
        print("validation result")
        model.eval()
        all_loss = 0
        accuracy = 0
        for i, (bsen, seq_lengths, buser, btime, ll,num4,textnum) in enumerate(val_loader,1):
            # measure data loading time
            # data_time.update(time.time() - end)
            bsen = Variable(bsen)
            buser = Variable(buser)
            btime = Variable(btime)
            num4 = Variable(num4)
            textnum = Variable(textnum)
            blabel = Variable(ll.float())
            # compute output
            label3 = model(bsen, buser, btime, seq_lengths,num4,textnum)
            loss, accu = loss_pred(label3, blabel)
            # print("acc:",1-loss2.data/label3.size()[0])
            # all_loss += 1-loss2.data/label3.size()[0]
            all_loss += loss.data
            accuracy += accu.data
        vllist.append(all_loss.cpu().numpy().item()/i)
        valist.append(accuracy.cpu().numpy().item()/i)
        print("loss ", all_loss.cpu().numpy().item() / i,"a acc ", accuracy.cpu().numpy().item() / i)
        all_loss = 0
        accuracy = 0

#print(torch.Tensor([tllist,vllist,talist,valist]))

if ispre:
    datapath2 = basepath + 'test_jeiba.json'
    test = pd.read_json(datapath2,typ = 'frame')
    with open('/home/syugroup/LishanYu/hw2/phase2/phase2_class.txt','r') as df:
        count = 0
        testid = []
        for line in df:
            kv = float(line)
            if kv == file_num:
                testid.append(count)
            count += 1
    test = test.iloc[testid]
    test_loader = dp.TestDataLoadernotforrnn(maxlen,test,userdict,textdict,batch_size=BATCH_SIZE)
    sen_result = []
    model.eval()
    for i, (bsen, seq_lengths,buser,btime,perm_id,num4,textnum) in enumerate(test_loader,1):
        bsen = Variable(bsen)
        buser = Variable(buser)
        btime = Variable(btime)
        num4 = Variable(num4)
        textnum = Variable(textnum)
        oid,oid_in_perm = perm_id.sort(0, descending = False)
        label3 = model(bsen, buser, btime, seq_lengths,num4,textnum)
        label3 = label3[oid_in_perm]
        pred = label3.data.round()
        pred = pred.cpu().numpy()
        sen_result.append(pred)
    if i > 1:
        sen_result = np.concatenate(sen_result)
    else:
        sen_result = sen_result[0]
    import math
    #i = 0
    #print(test["user"].iloc[i]+"\t"+test["weibo"].iloc[i]+"\t"+str(sen_result[i][0])+","+str(sen_result[i][1])+","+str(sen_result[i][2]))
    def savePred(result,test,txtName):
        f = open(txtName, "w+")
        i = 0
        for i in range(sen_result.shape[0]):
            f.write(test["user"].iloc[i]+"\t"+test["weibo"].iloc[i]+"\t"+str(int(sen_result[i][0]))+","+str(int(sen_result[i][1]))+","+str(int(sen_result[i][2]))+"\n")
        f.close()
    savePred(sen_result,test,txtName)