
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

save = True
serve = False
BATCH_SIZE = 4096
n_epoches = 200
jieba = True
mink = 20
usermink = 5
TIMEDIM = 33
isval = False
ispre = True
file_num = 7
ffnum = [6,7]

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
userpath = basepath+"user_ana.txt"
if jieba:
    txtName = basepath + 'phase2/resultmlpmodel_file'+str(file_num)+'.txt'


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
                                nn.Dropout(0.3),nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(text_embedding_dim+enc_dim+17+5,final_dim),
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
        return label3

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

#train = pd.read_json(datapath)
#train = pd.read_json(datapath,typ = 'frame')
if isval:
    '''
    train0 = pd.read_json(basepath + 'train_jeiba7/train_jeiba3.json',typ = 'frame')
    train1 = pd.read_json(basepath + 'train_jeiba7/train_jeiba4.json',typ = 'frame')
    train2 = pd.read_json(basepath + 'train_jeiba7/train_jeiba5.json',typ = 'frame')
    train3 = pd.read_json(basepath + 'train_jeiba7/train_jeiba6.json',typ = 'frame')
    train4 = pd.read_json(basepath + 'train_jeiba7/train_jeiba7.json',typ = 'frame')
    origin_train = pd.concat([train0,train1,train2,train3,train4],axis = 0,ignore_index = True)
    '''
    #origin_train = train
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
    train = origin_train

#train.ix[1]
#df1 = train.iloc[:5]
#df1.sample(frac=1)
#df1.sample(frac=1).reset_index(drop=True)  

if jieba:
    textdict = dp.text_jiebaDictionary(jiebapath, mink)  ## mink 以上
    userdict = dp.Dictionary(userpath,usermink)
else:
    user = np.array(train["user"])
    text = np.array(train["text"])
    textdict = dp.textDictionary(text)
    #userdict = dp.Dictionary(user)
    userdict = dp.userDictionary(userpath,usermink)
#textdict = dp.textDictionary(text)




train_loader =  dp.TextClassDataLoader_notidentify(train, userdict,textdict, batch_size=BATCH_SIZE)
if isval:
    val_loader = dp.TextClassDataLoader_notidentify(val, userdict,textdict,batch_size=BATCH_SIZE)

model = EDModel(userdict.__len__(),textdict.__len__()).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.005)
initial_model = model.state_dict().copy()

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
            print("loss ",all_loss.cpu().numpy().item()/5,"a acc ", accuracy.cpu().numpy().item()/5)
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
            blabel = Variable(ll.float())
            # compute output
            label3 = model(bsen, buser, btime, seq_lengths,num4,textnum)
            loss, accu = loss_pred(label3, blabel)
            all_loss = all_loss+ loss.data
            accuracy += accu.data
        print("loss ",all_loss.cpu().numpy().item()/i,"a acc ", accuracy.cpu().numpy().item()/i)

        if ep == n_epoches-1 and accuracy.cpu().numpy().item()>0.3:
            torch.save(initial_model,basepath + 'train_jeiba7/'+str(file_num)+'mlpinitial.pkl')
'''
datapath2 = basepath + 'test_jeiba.json'
test = pd.read_json(datapath2,typ = 'frame')

with open('/home/syugroup/LishanYu/hw2/phase2/phase2_class.txt','r') as df:
    count = 0
    testid = []
    for line in df:
        kv = float(line)
        if kv == 0:
            testid.append(count)
        count += 1
    test = test.iloc[testid]
f = open(basepath + 'phase2/resultmlpmodel_file0.txt', "w+")
i = 0
for i in range(len(testid)):
    f.write(test["user"].iloc[i]+"\t"+test["weibo"].iloc[i]+"\t"+"0,0,0\n")
f.close()
'''

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
    test_loader = dp.TestDataLoader_notidentify(test,userdict,textdict,batch_size=BATCH_SIZE)
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