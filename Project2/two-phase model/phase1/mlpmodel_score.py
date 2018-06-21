
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

save = True
serve = False
BATCH_SIZE = 1024*4
n_epoches = 8
jieba = True
mink = 30
usermink = 5
TIMEDIM = 31
isval = False
if serve:
    if jieba:
        import outer_pythoncode.datapreprocess_jieba as dp
    else:
        import outer_pythoncode.datapreprocess as dp
    basepath = '/data/user45/hw2/'
else:
    if jieba:
        import datapreprocess_jieba as dp
    else:
        import datapreprocess as dp
    basepath = '/home/syugroup/LishanYu/hw2/'
jiebapath = basepath+"jieba_frequency.txt"
#userpath = basepath+"userID_stat.txt"
userpath = basepath+"user_ana.txt"
if jieba:
    txtName = basepath + "lishan/class_resultmlpmodel_score.txt"
else:
    txtName = basepath + "resultmlpmodel_score.txt"


class Loss_3(nn.Module):
    def forward(self, input_tensor,label):
        out = torch.abs(input_tensor-label)
        temp=label+Variable(torch.ones(label.size()[0], 3).cuda() * torch.FloatTensor([5, 3, 3]).cuda())
        temp = temp*Variable(torch.FloatTensor([2,4,4]).cuda())
        out = out/temp
        out = 1-torch.sum(out,1)
        total = torch.sum(label,1)+1
        total[total>100]=101
        id = torch.arange(0,out.size()[0]).long().cuda()[out.data<0.8]
        out2 = (1-out[id])*total[id]
        tmp = 1-torch.sum(total[id])/torch.sum(total)
        return(torch.sum(out2)/torch.sum(total[id]),tmp)

class Loss_2(nn.Module):
    def forward(self, input_tensor,label):
        #out = torch.abs(input_tensor-label**2)
        out = (input_tensor-label)**2
        #out = torch.max(out,1)[0]
        return(torch.sum(out)/label.size()[0])

class EDModel(nn.Module):
    def __init__(self, len_userdic,len_textdic,final_dim = 64,user_embedding_dim=60,enc_dim=40,text_embedding_dim = 100,time_embedding_dim = 10,num_layers = 1):
        super(EDModel, self).__init__()
        self.model_name = 'Encoder'
        self.user_embedding_dim = user_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.user_embeddings = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
        self.time_embeddings = nn.Linear(TIMEDIM,self.time_embedding_dim,bias = True)
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+self.time_embedding_dim,enc_dim),
                                nn.Dropout(0.3),nn.LeakyReLU())
        self.mlp = nn.Sequential(nn.Linear(text_embedding_dim+enc_dim+17+5,final_dim),
                                nn.Dropout(0.3),nn.LeakyReLU())
        self.mlp1 = nn.Sequential(nn.Linear(final_dim, 3),
                                     nn.Dropout(0.3))
    def forward(self, sen,user,time,seq_lengths,num4,textnum):
        user_embeds = self.user_embeddings(user)
        time_embeds = self.time_embeddings(time)
        input_embeds = torch.cat((user_embeds,time_embeds),1)
        enc_embeds = self.enc_mlp(input_embeds)###context important!!
        enc_embeds = torch.cat((enc_embeds,num4,textnum),1)
        text_embeds = self.text_embeddings(sen)
        sen_embeds = torch.sum(text_embeds, 1)#/seq_lengths
        #print(sen_embeds.size())
        #print(seq_lengths.size())
        #sen_embeds = torch.t(torch.div(torch.t(sen_embeds),seq_lengths.long()))
        #print(seq_lengths)
        sen_embeds = torch.t(torch.t(sen_embeds)/Variable(seq_lengths).float())
        ###  text 可以乘tf-idf
        user_text = torch.cat((enc_embeds,sen_embeds), 1)
        label3 = self.mlp(user_text)
        label3 = self.mlp1(label3)
        label3 = torch.sum(label3,1)
        return label3

#loss_pred = Loss_3()
loss_pred = Loss_2()
import pandas as pd
import numpy as np

datapath = basepath + 'train_jeiba.json'
datapath2 = basepath + 'test_jeiba.json'
classlabel='/home/syugroup/LishanYu/hw2/classlabel.txt'
flow = []
with open(classlabel,'r') as df:
    for line in df:
        kv = line.split('\t')
        flow.append(int(kv[1]))
#train = pd.read_json(datapath)
if isval:
    origin_train = pd.read_json(datapath,typ='frame')
    time = origin_train['time']
    valid = []
    trainid = []
    trainflow = []
    valflow = []
    for i in range(len(time)):
        if time[i][6] == '7':
            valid.append(i)
            valflow.append(flow[i])
        else:
            trainid.append(i)
            trainflow.append(flow[i])
    train = origin_train.iloc[trainid]
    val = origin_train.iloc[valid]
else:
    train = pd.read_json(datapath,typ = 'frame')
    trainflow = flow

#train.ix[1]
#df1 = train.iloc[:5]
#df1.sample(frac=1)
#df1.sample(frac=1).reset_index(drop=True)  

test = pd.read_json(datapath2,typ = 'frame')
if jieba:
    textdict = dp.text_jiebaDictionary(jiebapath, mink)  ## mink 以上
    userdict = dp.Dictionary(userpath,usermink)
else:
    user = np.array(train["user"])
    text = np.array(train["text"])
    textdict = dp.textDictionary(text)
    #userdict = dp.Dictionary(user)
    userdict = dp.Dictionary(userpath,usermink)
#textdict = dp.textDictionary(text)



train_loader =  dp.TextClassDataLoader_class(train=train, userdict=userdict,textdict=textdict,flow = trainflow, batch_size=BATCH_SIZE)
if isval:
    val_loader = dp.TextClassDataLoader_class(val, userdict,textdict,flow = valflow,batch_size=BATCH_SIZE)

model = EDModel(userdict.__len__(),textdict.__len__()).cuda()
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
        #ll = ll.repeat(2,1).t()
        blabel= Variable(ll.float())
        # compute output
        label3= model(bsen,buser,btime, seq_lengths,num4,textnum)
        #print(label3.size())
        #print(blabel.size())
        loss = loss_pred(label3,blabel)
        pred =  label3.data
        pred[pred<0] = 0
        #pred = torch.sqrt(pred)
        pred = pred.round()
        accuracy += pred.eq(blabel.data.view_as(pred)).cpu().sum()/len(pred)
        #print("acc:",1-loss2.data/label3.size()[0])
        all_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%10==0:
            print("accu ",accuracy/10,'loss is ',all_loss.cpu().numpy().item()/10)
            all_loss = 0
            accuracy = 0
    if isval:
        print("validation result")
        model.eval()
        all_loss = 0
        accuracy = 0
        for i, (bsen, seq_lengths, buser, btime, ll, num4,textnum) in enumerate(val_loader,1):
            # measure data loading time
            # data_time.update(time.time() - end)
            bsen = Variable(bsen)
            buser = Variable(buser)
            btime = Variable(btime)
            num4 = Variable(num4)
            textnum = Variable(textnum)
            #ll = ll.repeat(2,1).t()
            blabel = Variable(ll.float())
            # compute output
            label3 = model(bsen, buser, btime, seq_lengths,num4,textnum)
            loss = loss_pred(label3, blabel)
            pred =  label3.data
            pred[pred<0] = 0
            #pred = torch.sqrt(pred)
            pred = pred.round()
            accuracy += pred.eq(blabel.data.view_as(pred)).cpu().sum()/len(pred)
            # print("acc:",1-loss2.data/label3.size()[0])
            # all_loss += 1-loss2.data/label3.size()[0]
            all_loss += loss.data
            if i %10 == 0:
                print("accu ",accuracy/10,"loss ",all_loss.cpu().numpy().item()/10)
                all_loss = 0
                accuracy = 0
        if ep == n_epoches-1:
            sen_result = []
            ll_result = []
            for i, (bsen, seq_lengths, buser, btime, ll, num4,textnum) in enumerate(val_loader,1):
                bsen = Variable(bsen)
                buser = Variable(buser)
                btime = Variable(btime)
                num4 = Variable(num4)
                textnum = Variable(textnum)
                label3 = model(bsen,buser,btime, seq_lengths,num4,textnum)#input_em :(len(input_em)-1)
                ll_result.append(ll.cpu().numpy())
                label3 = label3.data
                label3[label3<0] = 0
                #label3 = torch.sqrt(label3)
                label3 = label3.cpu().numpy()
                #label3 = np.around(label3)
                sen_result.append(label3)
            sen_result = np.concatenate(sen_result)
            ll_result = np.concatenate(ll_result)
            f = open(basepath + "phase1/validation_mlp_score.txt", "w+")
            i = 0
            for i in range(sen_result.shape[0]):
                f.write(str(ll_result[i])+"\t"+str(sen_result[i])+"\n")
            f.close()


save = True

if save:
    test_loader = dp.TestDataLoader_class(test,userdict,textdict,batch_size=BATCH_SIZE)
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
        #label3 = torch.max(label3,dim = 1)[0]
        label3 = label3.data.cpu().numpy()
        sen_result.append(label3)
        #if i == 2:
        #    break
    sen_result = np.concatenate(sen_result)
    f = open(basepath + "phase1/prediction_mlp_score.txt", "w+")
    i = 0
    for i in range(sen_result.shape[0]):
        f.write(str(sen_result[i])+"\n")
    f.close()

