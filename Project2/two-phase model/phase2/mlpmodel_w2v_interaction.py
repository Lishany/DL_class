
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from gensim.models import Word2Vec
w2vmodel = Word2Vec.load('/home/syugroup/LishanYu/hw2/word2vec')

save = False
serve = False
BATCH_SIZE = 10240*2
n_epoches = 14
jieba = True
mink = 30
usermink = 5
TIMEDIM = 33
w2vfix = False
isval = True
ispre = True
file_num = 7
ffnum = [5,6,7]


if jieba:
    import outer_pythoncode.datapreprocess_jieba as dp
else:
    import outer_pythoncode.datapreprocess as dp
basepath = '/data/user45/hw2/'

jiebapath = basepath+"jieba_frequency.txt"
#userpath = basepath+"userID_stat.txt"
userpath = basepath+"user_ana.txt"
txtName = basepath + 'phase2/resultmlpmodel_file'+str(file_num)+'w2vinitial.txt'

import pandas as pd
import numpy as np

datapath = basepath + 'train_jeiba7/train_jeiba'+str(file_num)+'.json'

origin_train = pd.read_json(basepath + 'train_jeiba7/train_jeiba'+str(ffnum[0])+'.json',typ = 'frame')
i = 1
while i < len(ffnum):
    temp = pd.read_json(basepath + 'train_jeiba7/train_jeiba'+str(ffnum[i])+'.json',typ = 'frame')
    origin_train = origin_train.append(temp,ignore_index=True)
    i = i + 1
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


if jieba:
    textdict = dp.text_jiebaDictionary(jiebapath, mink)  ## mink 以上
    userdict = dp.userDictionary(userpath,usermink)
else:
    user = np.array(train["user"])
    text = np.array(train["text"])
    textdict = dp.textDictionary(text)
    #userdict = dp.Dictionary(user)
    userdict = dp.userDictionary(userpath,usermink)
#textdict = dp.textDictionary(text)

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
    def __init__(self, len_userdic,len_textdic,final_dim = 64,user_embedding_dim=50,enc_dim=40,text_embedding_dim = 50,time_embedding_dim = 10,num_layers = 1):
        super(EDModel, self).__init__()
        self.model_name = 'Encoder'
        self.user_embedding_dim = user_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        self.text_embeddings.weight = nn.Parameter(torch.FloatTensor(w2vdict))
        self.user_embeddings = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
        self.time_embeddings = nn.Linear(TIMEDIM,self.time_embedding_dim,bias = True)
        self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+17,enc_dim),
                                nn.Dropout(0.3),nn.ReLU())

        self.mlpuser = nn.Sequential(nn.Linear(enc_dim,3),
                                nn.Dropout(0.3),nn.ReLU())
        self.mlptext = nn.Sequential(nn.Linear(self.text_embedding_dim+self.time_embedding_dim+5,9),
                                nn.Dropout(0.3),nn.Sigmoid())
        self.mlpuser2 = nn.Sequential(nn.Linear(enc_dim,3),
                                nn.Dropout(0.3),nn.ReLU())
    def forward(self, sen,user,time,seq_lengths,num4,textnum):
        user_embeds = self.user_embeddings(user)
        time_embeds = self.time_embeddings(time)
        input_embeds = torch.cat((user_embeds,num4),1)
        enc_embeds = self.enc_mlp(input_embeds)###context important!

        text_embeds = self.text_embeddings(sen)
        sen_embeds = torch.sum(text_embeds, 1)#/seq_lengths
        #print(sen_embeds.size())
        #print(seq_lengths.size())
        #sen_embeds = torch.t(torch.div(torch.t(sen_embeds),seq_lengths.long()))
        #print(seq_lengths)
        sen_embeds = torch.t(torch.t(sen_embeds)/Variable(seq_lengths).float())
        ###  text 可以乘tf-idf
        sen_embeds = torch.cat((textnum,sen_embeds,time_embeds), 1)
        out_user = self.mlpuser(enc_embeds)
        out_user2 = self.mlpuser2(enc_embeds)
        out_text = self.mlptext(sen_embeds)
        out_user = out_user.unsqueeze(1)
        out_text = out_text.view((sen.size()[0],3,-1))
        all_out = out_text * out_user
        label3 = torch.sum(all_out,1)/3
        return label3

loss_pred = Loss_3()


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
#optimizer = optim.Adam(model.parameters(), lr=0.008)

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
        if i%50==0:
            print("loss ",all_loss.cpu().numpy().item()/50,"a acc ", accuracy.cpu().numpy().item() / 50)
            tllist.append(all_loss.cpu().numpy().item()/50)
            talist.append(accuracy.cpu().numpy().item()/50)
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
            # print("acc:",1-loss2.data/label3.size()[0])
            # all_loss += 1-loss2.data/label3.size()[0]
            all_loss += loss.data
            accuracy += accu.data
        vllist.append(all_loss.cpu().numpy().item()/i)
        valist.append(accuracy.cpu().numpy().item()/i)
        print("loss ", all_loss.cpu().numpy().item() / i,"a acc ", accuracy.cpu().numpy().item()/ i)
        all_loss = 0
        accuracy = 0
print(torch.Tensor([tllist,vllist,talist,valist]))



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