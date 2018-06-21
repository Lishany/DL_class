
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch import optim
from gensim.models import Word2Vec
w2vmodel = Word2Vec.load('/home/syugroup/LishanYu/hw2/word2vec')

save = True
serve = False
BATCH_SIZE = 512
n_epoches = 18
jieba = True
mink = 30
usermink = 10
offset = 0.92
isval = True
TIMEDIM = 33
w2vfix = False
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
userpath = basepath+"user_ana.txt"

if jieba:
    txtName = basepath + "train_jeiba7/result_relu_simplemodel_jieba_userpart.txt"
else:
    txtName = basepath + "result_relu_simplemodel_userpart.txt"

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



loss_auto = nn.NLLLoss()
loss_pred = Loss_3()


import pandas as pd
import numpy as np
if jieba:
    datapath = basepath + 'train_jeiba.json'
    datapath2 = basepath + 'test_jeiba.json'
else:
    datapath = basepath + 'train.json'
    datapath2 = basepath+'test.json'


#train = pd.read_json(datapath)
if isval:
    origin_train = pd.read_json(datapath,typ = 'frame')
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
test = pd.read_json(datapath2,typ = 'frame')
if jieba:
    textdict = dp.text_jiebaDictionary(jiebapath, mink)  ## mink 以上
    userdict = dp.userDictionary(userpath,usermink)
else:
    user = np.array(train["user"])
    text = np.array(train["text"])
    textdict = dp.textDictionary(text)
    userdict = dp.Dictionary(user)
#textdict = dp.textDictionary(text)
#train_loader =  dp.TextClassDataLoader(train, userdict,textdict, batch_size=BATCH_SIZE)
#if isval:
#    val_loader = dp.TextClassDataLoader(val, userdict,textdict, batch_size=BATCH_SIZE)

w2vdict = []
for i in range(len(textdict.idx2word)):
    word = textdict.idx2word[i]
    w2vdict.append(w2vmodel.wv[word])
w2vdict = np.stack(w2vdict)

identifyuser = dp.Identifyuser(train, userdict,textdict, 7, isval)


train_loader =  dp.TextClassDataLoader(train, userdict,textdict,identifyuser, batch_size=BATCH_SIZE)
if isval:
    val_loader = dp.TextClassDataLoader(val, userdict,textdict, identifyuser,batch_size=BATCH_SIZE)


class EDModel(nn.Module):
    def __init__(self, len_userdic,len_textdic,hidden_dim2 = 120,final_dim = 80,user_embedding_dim=80,enc_dim=50,text_embedding_dim = 140,time_embedding_dim = 25,num_layers = 1):
        super(EDModel, self).__init__()
        self.model_name = 'Encoder'
        self.user_embedding_dim = user_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        self.text_embeddings.weight = nn.Parameter(torch.FloatTensor(w2vdict))
        self.user_embeddings = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
        #self.time_embeddings = nn.Embedding(67,self.time_embedding_dim,padding_idx=0)
        self.time_embeddings = nn.Linear(TIMEDIM,self.time_embedding_dim,bias = False)
        self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+self.time_embedding_dim,enc_dim),
                                nn.Dropout(0.3),nn.ReLU())

        self.senpred_lstm = nn.LSTM(input_size=self.text_embedding_dim,
                                hidden_size=hidden_dim2, num_layers=num_layers, batch_first=True,
                                # num_layers = opt.num_layers,
                                # bias = True,
                                dropout = 0.3,
                                bidirectional=False
                                )
        self.mlp = nn.Sequential(nn.Linear(hidden_dim2+enc_dim+4+5,final_dim),
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
        
        #_,packeed_lstm_out2 = self.senpred_lstm(packed_input_sen, None)
        #print('pack',packeed_lstm_out2.size())
        #out_vec2 = pad_packed_sequence(packeed_lstm_out2, batch_first=True)


        user_text = torch.cat((enc_embeds,out_vec2), 1)

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
        if i%150==0:
            print("loss ",all_loss.cpu().numpy().item()/150,"a acc ", accuracy.cpu().numpy().item() / 150)
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
            loss, accu = loss_pred(label3, blabel)
            # print("acc:",1-loss2.data/label3.size()[0])
            # all_loss += 1-loss2.data/label3.size()[0]
            all_loss += loss.data
            accuracy += accu.data
            if i % 90 == 0:
                print("loss ", all_loss.cpu().numpy().item() / 90,"a acc ", accuracy.cpu().numpy().item() / 90)
                all_loss = 0
                accuracy = 0


#print(model)

if save:
    test_loader = dp.TestDataLoader(test,userdict,textdict,identifyuser,batch_size=BATCH_SIZE)
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
