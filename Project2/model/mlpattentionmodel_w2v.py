
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from gensim.models import Word2Vec
w2vmodel = Word2Vec.load('/home/syugroup/LishanYu/hw2/word2vec')

save = True
serve = False
BATCH_SIZE = 512
n_epoches = 6
jieba = True
mink = 12
usermink = 10
offset = 0.92
isval = True
attk = 3
attk2 = 7
hidden_dim3 = 4
w2vfix = False
TIMEDIM = 33


import datapreprocess_jieba as dp
basepath = '/home/syugroup/LishanYu/hw2/'
jiebapath = basepath+"jieba_frequency.txt"
#userpath = basepath+"frequency_userID.txt"
userpath = basepath+"userID_stat.txt"
txtName = basepath + "lishan/resultmlp_attention_model_jiebapartuser.txt"


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

datapath = basepath + 'train_jeiba.json'
datapath2 = basepath + 'test_jeiba.json'

#train = pd.read_json(datapath)
if isval:
    origin_train = pd.read_json(datapath,typ='frame')
    #origin_train = origin_train.sample(frac = 1)
    #trainlen = int(len(origin_train)*offset)
    #train = origin_train.iloc[:trainlen]
    #val = origin_train.iloc[trainlen:]
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
    train = pd.read_json(datapath,typ='frame')

#train.ix[1]
#df1 = train.iloc[:5]
#df1.sample(frac=1)
#df1.sample(frac=1).reset_index(drop=True)  

test = pd.read_json(datapath2,typ='frame')

textdict = dp.text_jiebaDictionary(jiebapath, mink)  ## mink 以上
userdict = dp.userDictionary(userpath,usermink)

w2vdict = []
for i in range(len(textdict.idx2word)):
    word = textdict.idx2word[i]
    w2vdict.append(w2vmodel.wv[word])
w2vdict = np.stack(w2vdict)

'''
identifyuser = dp.Identifyuser(train, userdict,textdict, 8, isval)
train_loader =  dp.TextClassDataLoader(train, userdict,textdict,identifyuser, batch_size=BATCH_SIZE)
if isval:
    val_loader = dp.TextClassDataLoader(val, userdict,textdict, identifyuser,batch_size=BATCH_SIZE)
'''
train_loader =  dp.TextClassDataLoader_notidentify(train, userdict,textdict, batch_size=BATCH_SIZE)
if isval:
    val_loader = dp.TextClassDataLoader_notidentify(val, userdict,textdict,batch_size=BATCH_SIZE)


class EDModel(nn.Module):
    def __init__(self, len_userdic,len_textdic,final_dim = 70,user_embedding_dim=80,enc_dim=50,text_embedding_dim = 140,time_embedding_dim = 25,num_layers = 1):
        super(EDModel, self).__init__()
        self.model_name = 'Encoder'
        self.user_embedding_dim = user_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        self.text_embeddings.weight = nn.Parameter(torch.FloatTensor(w2vdict))
        self.user_embeddings = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
        self.time_embeddings = nn.Linear(TIMEDIM,self.time_embedding_dim,bias = True)
        self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+self.time_embedding_dim,enc_dim),
                                nn.Dropout(0.3),nn.ReLU())

        self.att_mlp = nn.Sequential(nn.Conv1d(1,out_channels=hidden_dim3,kernel_size=attk*self.text_embedding_dim,padding=int(attk/2)*self.text_embedding_dim,stride=self.text_embedding_dim),nn.Dropout(0.3),nn.ReLU())
        self.att_mlp2 = nn.Sequential(nn.Conv1d(hidden_dim3,1,kernel_size=attk2,padding=int(attk2/2)),nn.Dropout(0.3),nn.Tanh())
        self.att_softmax = F.softmax
        self.mlp = nn.Sequential(nn.Linear(text_embedding_dim+enc_dim+9,final_dim),
                                nn.Dropout(0.3),nn.ReLU())
        self.mlp1 = nn.Sequential(nn.Linear(final_dim, 3),
                                     nn.Dropout(0.3))

    #def init_hidden(self):
    #    return (autograd.Variable(torch.zeros(1, 1, self.embedding_dim0)),autograd.Variable(torch.zeros(1, 1, self.embedding_dim0)))

    def forward(self, sen,user,time,seq_lengths,num4,textnum):
        user_embeds = self.user_embeddings(user)
        time_embeds = self.time_embeddings(time)
        input_embeds = torch.cat((user_embeds,time_embeds),1)
        enc_embeds = self.enc_mlp(input_embeds)###context important!!
        enc_embeds = torch.cat((enc_embeds,num4,textnum),1)
        text_embeds = self.text_embeddings(sen)
        sen_embeds = text_embeds.view((text_embeds.size()[0],-1))
        #print(sen_embeds.size())
        atten = self.att_mlp(sen_embeds.unsqueeze(1))
        #print(atten.size())
        atten = self.att_mlp2(atten)
        atten = atten.squeeze(1)
        atten = self.att_softmax(atten).unsqueeze(2)
        #print(atten.size())
        sen_embeds = text_embeds*atten
        sen_embeds = torch.sum(text_embeds, 1)#/seq_lengths
        user_text = torch.cat((enc_embeds,sen_embeds), 1)
        label3 = self.mlp(user_text)
        label3 = self.mlp1(label3)
        return label3

model = EDModel(userdict.__len__(),textdict.__len__()).cuda()
if w2vfix:
    i = 0
    for param in model.parameters():
        if i == 0:
            param.requires_grad = False
        else:
            param.requires_grad = True
        i = i + 1
    op_parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(op_parameter, lr=0.008)
else:
    optimizer = optim.Adam(model.parameters(), lr=0.008)

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
        if i%100==0:
            print("loss ",all_loss.cpu().numpy().item()/100,"a acc ", accuracy.cpu().numpy().item() / 100)
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
            if i % 40 == 0:
                print("loss ", all_loss.cpu().numpy().item() / 40,"a acc ", accuracy.cpu().numpy().item() / 40)
                all_loss = 0
                accuracy = 0



#print(model)

if save:
    test_loader = dp.TestDataLoader(test,userdict,textdict,identifyuser,batch_size=BATCH_SIZE)
    sen_result = []
    model.eval()
    for i, (bsen, seq_lengths,buser,btime,perm_id,num4,textnum) in enumerate(test_loader,1):
        bsen = Variable(bsen)
        buser = Variable(buser)
        btime = Variable(btime)
        num4 = Variable(num4)
        textnum = Variable(textnum)
        oid,oid_in_perm = perm_id.sort(0, descending = False)
        label3 = model(bsen,buser,btime, seq_lengths,num4,textnum)#input_em :(len(input_em)-1)
        label3 = label3[oid_in_perm]
        label3 = label3.data.cpu()
        label3[label3<0]=0
        sen_result.append(label3.numpy())
        #if i == 2:
        #    break
    sen_result = np.concatenate(sen_result)


    import math
    def savePred(result,test,txtName):
        f = open(txtName, "w+")
        i = 0
        for i in range(sen_result.shape[0]):
            f.write(test["user"][i]+"\t"+test["weibo"][i]+"\t"+str(math.floor(sen_result[i][0]))+","+str(math.floor(sen_result[i][1]))+","+str(math.floor(sen_result[i][2]))+"\n")
        f.close()
    savePred(sen_result,test,txtName)
