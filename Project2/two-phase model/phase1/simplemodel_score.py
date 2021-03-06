
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch import optim
import torch.nn.functional as F

save = True
serve = False
BATCH_SIZE = 1024*4
n_epoches = 8
jieba = True
mink = 30
usermink = 5
offset = 0.92
isval = False
TIMEDIM = 31
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
#userpath = basepath+"frequency_userID.txt"
#userpath = basepath+"userID_stat.txt"
userpath = basepath+"user_ana.txt"

if jieba:
    txtName = basepath + "phase1/class_result_simplemodel_score.txt"
else:
    txtName = basepath + "lishan/class_result_relu_simplemodel_.txt"

class Loss_2(nn.Module):
    def forward(self, input_tensor,label):
        #out = torch.abs(input_tensor-label**2)
        out = (input_tensor-label)**2
        #out = torch.max(out,1)[0]
        return(torch.sum(out)/label.size()[0])

class EDModel1(nn.Module):
    def __init__(self, len_userdic,len_textdic,hidden_dim2 = 80,final_dim = 64,user_embedding_dim=40,enc_dim=50,text_embedding_dim = 100,time_embedding_dim = 10,num_layers = 1):
        super(EDModel1, self).__init__()
        self.model_name = 'Encoder'
        self.user_embedding_dim = user_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.user_embeddings = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
        self.time_embeddings = nn.Linear(TIMEDIM,self.time_embedding_dim,bias = False)
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        #self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+self.time_embedding_dim,enc_dim),
        #                        nn.Dropout(0.3),nn.Sigmoid())
        self.enc_mlp = nn.Sequential(nn.Linear(17+5+self.user_embedding_dim+self.time_embedding_dim,enc_dim),
                                nn.Dropout(0.3),nn.Sigmoid())

        self.senpred_lstm = nn.LSTM(input_size=self.text_embedding_dim,
                                hidden_size=hidden_dim2, num_layers=num_layers, batch_first=True,
                                # num_layers = opt.num_layers,
                                # bias = True,
                                dropout = 0.3,
                                bidirectional=False
                                )
        #self.mlp = nn.Sequential(nn.Linear(hidden_dim2+enc_dim+17+5,final_dim),
        #                        nn.Dropout(0.3),nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(hidden_dim2+enc_dim,final_dim),
                                nn.Dropout(0.3),nn.Sigmoid())
        self.mlp1 = nn.Sequential(nn.Linear(final_dim, 3),
                                     nn.Dropout(0.3))
    def forward(self, sen,user,time,seq_lengths,num4,textnum):
        user_embeds = self.user_embeddings(user)
        time_embeds = self.time_embeddings(time)
        input_embeds = torch.cat((user_embeds,time_embeds,num4,textnum),1)
        enc_embeds = self.enc_mlp(input_embeds)###context important!!
        text_embeds = self.text_embeddings(sen)
        sen_len = seq_lengths.cpu().numpy()
        packed_input_sen = pack_padded_sequence(text_embeds, sen_len, batch_first=True)
        packeed_lstm_out2, _ = self.senpred_lstm(packed_input_sen, None)
        out_vec2, _ = pad_packed_sequence(packeed_lstm_out2, batch_first=True)
        out_vec2 = out_vec2.contiguous()
        out_vec2 = out_vec2.view(sen.size()[0] * sen.size()[1], -1)
        id2 = []
        id2.append(sen_len[0])
        for i in range(1, user.size()[0]):
            id2.append(sen_len[i] + i * sen.size()[1])
        id2 = torch.LongTensor(np.array(id2)).cuda()
        out_vec2 = out_vec2[id2]
        user_text = torch.cat((enc_embeds,out_vec2), 1)
        label3 = self.mlp(user_text)
        label3 = self.mlp1(label3)
        label3 = torch.sum(label3,1)
        return label3

class EDModel(nn.Module):
    def __init__(self, len_userdic,len_textdic,hidden_dim2 = 80,final_dim = 64,user_embedding_dim=40,enc_dim=50,text_embedding_dim = 100,time_embedding_dim = 10,num_layers = 1):
        super(EDModel, self).__init__()
        self.model_name = 'Encoder'
        self.user_embedding_dim = user_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.user_embeddings = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
        self.time_embeddings = nn.Linear(TIMEDIM,self.time_embedding_dim,bias = False)
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+self.time_embedding_dim,enc_dim),
                                nn.Dropout(0.3),nn.Sigmoid())

        self.senpred_lstm = nn.LSTM(input_size=self.text_embedding_dim,
                                hidden_size=hidden_dim2, num_layers=num_layers, batch_first=True,
                                # num_layers = opt.num_layers,
                                # bias = True,
                                dropout = 0.3,
                                bidirectional=False
                                )
        self.mlp = nn.Sequential(nn.Linear(hidden_dim2+enc_dim+17+5,final_dim),
                                nn.Dropout(0.3),nn.Sigmoid())
        self.mlp1 = nn.Sequential(nn.Linear(final_dim, 3),
                                     nn.Dropout(0.3))
    def forward(self, sen,user,time,seq_lengths,num4,textnum):
        user_embeds = self.user_embeddings(user)
        time_embeds = self.time_embeddings(time)
        input_embeds = torch.cat((user_embeds,time_embeds),1)
        enc_embeds = self.enc_mlp(input_embeds)###context important!!
        enc_embeds = torch.cat((enc_embeds,num4,textnum),1)
        text_embeds = self.text_embeddings(sen)
        sen_len = seq_lengths.cpu().numpy()
        packed_input_sen = pack_padded_sequence(text_embeds, sen_len, batch_first=True)
        packeed_lstm_out2, _ = self.senpred_lstm(packed_input_sen, None)
        out_vec2, _ = pad_packed_sequence(packeed_lstm_out2, batch_first=True)
        out_vec2 = out_vec2.contiguous()
        out_vec2 = out_vec2.view(sen.size()[0] * sen.size()[1], -1)
        id2 = []
        id2.append(sen_len[0])
        for i in range(1, user.size()[0]):
            id2.append(sen_len[i] + i * sen.size()[1])
        id2 = torch.LongTensor(np.array(id2)).cuda()
        out_vec2 = out_vec2[id2]
        user_text = torch.cat((enc_embeds,out_vec2), 1)
        label3 = self.mlp(user_text)
        label3 = self.mlp1(label3)
        label3 = torch.sum(label3,1)
        return label3


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


if jieba:
    textdict = dp.text_jiebaDictionary(jiebapath, mink)  ## mink 以上
    userdict = dp.Dictionary(userpath,usermink)
else:
    user = np.array(train["user"])
    text = np.array(train["text"])
    textdict = dp.textDictionary(text)
    userdict = dp.Dictionary(userpath,usermink)
#textdict = dp.textDictionary(text)
#train_loader =  dp.TextClassDataLoader(train, userdict,textdict, batch_size=BATCH_SIZE)
#if isval:
#    val_loader = dp.TextClassDataLoader(val, userdict,textdict, batch_size=BATCH_SIZE)

train_loader =  dp.TextClassDataLoader_class(train=train, userdict=userdict,textdict=textdict,flow = trainflow, batch_size=BATCH_SIZE)
if isval:
    val_loader = dp.TextClassDataLoader_class(val, userdict,textdict,flow = valflow,batch_size=BATCH_SIZE)


model = EDModel1(userdict.__len__(),textdict.__len__()).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.005)

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
        loss = loss_pred(label3,blabel)
        #print("acc:",1-loss2.data/label3.size()[0])
        pred = label3.data
        pred[pred<0]=0
        pred = pred.round()
        accuracy += pred.eq(blabel.data.view_as(pred)).cpu().sum()/len(pred)
        all_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%10==0:
            print("accu ",accuracy/10,"loss ",all_loss.cpu().numpy().item()/10)
            all_loss = 0
            accuracy = 0
        #if i>1:
        #    break
    if isval:
        print("validation result")
        all_loss = 0
        accuracy = 0
        model.eval()
        for i, (bsen, seq_lengths, buser, btime, ll,num4,textnum) in enumerate(val_loader,1):
            # measure data loading time
            # data_time.update(time.time() - end)
            bsen = Variable(bsen)
            buser = Variable(buser)
            btime = Variable(btime)
            blabel = Variable(ll.float())
            num4 = Variable(num4)
            textnum = Variable(textnum)
            # compute output
            label3 = model(bsen, buser, btime, seq_lengths,num4,textnum)
            loss = loss_pred(label3,blabel)
            #pred = label3.data.round()
            pred = label3.data
            pred[pred<0]=0
            pred = pred.round()
            accuracy += pred.eq(blabel.data.view_as(pred)).cpu().sum()/len(pred)
            all_loss += loss.data
            # print("acc:",1-loss2.data/label3.size()[0])
            # all_loss += 1-loss2.data/label3.size()[0]
            if i%10==0:
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
                #label3[label3<0]=0
                #label3 = label3.round()
                label3 = label3.cpu().numpy()
                #label3 = np.around(label3)
                sen_result.append(label3)
            sen_result = np.concatenate(sen_result)
            ll_result = np.concatenate(ll_result)
            f = open(basepath + "phase1/validation_simple_score.txt", "w+")
            i = 0
            for i in range(sen_result.shape[0]):
                f.write(str(ll_result[i])+"\t"+str(sen_result[i])+"\n")
            f.close()


#print(model)
save = True
if save:
    test_loader = dp.TestDataLoader_class(test,userdict,textdict,batch_size=BATCH_SIZE)
    sen_result = []
    model.eval()
    for i, (bsen, seq_lengths,buser,btime,perm_id,num4,textnum) in enumerate(test_loader):
        bsen = Variable(bsen)
        buser = Variable(buser)
        btime = Variable(btime)
        num4 = Variable(num4)
        textnum = Variable(textnum)
        oid,oid_in_perm = perm_id.sort(0, descending = False)
        label3 = model(bsen,buser,btime, seq_lengths,num4,textnum)#input_em :(len(input_em)-1)
        label3 = label3[oid_in_perm]
        pred = label3.data.round()
        pred = pred.cpu()
        sen_result.append(pred.numpy())
        #if i == 2:
        #    break
    sen_result = np.concatenate(sen_result)
    '''
    import math
    def savePred(result,test,txtName):
        f = open(txtName, "w+")
        i = 0
        for i in range(sen_result.shape[0]):
            f.write(test["user"][i]+"\t"+test["weibo"][i]+"\t"+str(sen_result[i])+"\n")
        f.close()
    savePred(sen_result,test,txtName)
    '''
    f = open(basepath + "phase1/prediction_simple_score.txt", "w+")
    i = 0
    for i in range(sen_result.shape[0]):
        f.write(str(sen_result[i])+"\n")
    f.close()
