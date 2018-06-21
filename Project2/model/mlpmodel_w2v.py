
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
mink = 1000
usermink = 10
TIMEDIM = 33
w2vfix = False
isval = True
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
    txtName = basepath + "lishan/resultmlpmodel_w2v.txt"
else:
    txtName = basepath + "resultmlpmodelpartuser.txt"



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
    train = pd.read_json(datapath,typ = 'frame')

#train.ix[1]
#df1 = train.iloc[:5]
#df1.sample(frac=1)
#df1.sample(frac=1).reset_index(drop=True)  

test = pd.read_json(datapath2,typ = 'frame')
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

'''
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

        self.mlptext = nn.Sequential(nn.Linear(self.text_embedding_dim+self.time_embedding_dim+5+enc_dim,64),
                                nn.Dropout(0.3),nn.ReLU())
        self.final_mlp = nn.Sequential(nn.Linear(64,3),
                                nn.Dropout(0.3))
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
        sen_embeds = torch.cat((textnum,sen_embeds,time_embeds,enc_embeds), 1)
        sen_embeds = self.mlptext(sen_embeds)
        label3 = self.final_mlp(sen_embeds)
        return label3

'''
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
        self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+self.time_embedding_dim,enc_dim),
                                nn.Dropout(0.3),nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(text_embedding_dim+enc_dim+17+5,final_dim),
                                nn.Dropout(0.3),nn.ReLU())
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
        return label3

loss_pred = Loss_3()


model = EDModel(userdict.__len__(),textdict.__len__()).cuda()
print(model)

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
            if i %10 == 0:
                print("loss ", all_loss.cpu().numpy().item() / 10,"a acc ", accuracy.cpu().numpy().item()/ 10)
                vllist.append(all_loss.cpu().numpy().item()/10)
                valist.append(accuracy.cpu().numpy().item()/10)
                all_loss = 0
                accuracy = 0
print(torch.Tensor([tllist,vllist,talist,valist]))


#print(model)

if save:
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
        label3 = model(bsen,buser,btime, seq_lengths,num4,textnum)#input_em :(len(input_em)-1)
        label3 = label3[oid_in_perm]
        label3 = label3.data
        label3[label3<0]=0
        sen_result.append(label3)
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
