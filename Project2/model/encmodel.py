
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch import optim

n_epoches = 14
enc_epo = 2
BATCH_SIZE =10240
jieba = True
mink = 30
usermink = 5
offset = 0.92
isval = True
TIME_DIM = 33
save = False
if jieba:
    import datapreprocess_jieba as dp
else:
    import datapreprocess as dp
basepath = '/home/syugroup/LishanYu/hw2/'
if jieba:
    encmodelpath = basepath + 'lishan/encparams_jieba.pkl'
else:
    encmodelpath = basepath+'lishan/encparams.pkl'
jiebapath = basepath+"jieba_frequency.txt"
#userpath = basepath+"frequency_userID.txt"
userpath = basepath+"user_ana.txt"

if jieba:
    datapath = basepath + 'train_jeiba.json'
    datapath2 = basepath + 'test_jeiba.json'
else:
    datapath = basepath + 'train.json'
    datapath2 = basepath+'test.json'

class ENCModel(nn.Module):
    def __init__(self, len_userdic,len_textdic,hidden_dim = 40,user_embedding_dim=60,enc_dim=40,text_embedding_dim = 50,time_embedding_dim = 10,num_layers = 1):
        super(ENCModel, self).__init__()
        self.model_name = 'Encoder'
        self.user_embedding_dim = user_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.user_embeddings = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
        #self.time_embeddings = nn.Embedding(67,self.time_embedding_dim,padding_idx=0)
        self.time_embeddings = nn.Linear(TIME_DIM,self.time_embedding_dim,bias = False)
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+self.time_embedding_dim+17+5,enc_dim),
                                nn.Dropout(0.3),nn.ReLU())
        self.sen_lstm = nn.LSTM(input_size = self.text_embedding_dim+enc_dim,
                                hidden_size = hidden_dim,num_layers=num_layers,batch_first = True,
                                bidirectional = False
                                )
        self.dec_mlp = nn.Sequential(nn.Linear(hidden_dim, len_textdic),
                                     nn.Dropout(0.3), nn.ReLU())
        self.soft = nn.LogSoftmax()

    def forward(self, sen,user,time,seq_lengths,num4,textnum):
        user_embeds = self.user_embeddings(user)
        time_embeds = self.time_embeddings(time)
        input_embeds = torch.cat((user_embeds,time_embeds,num4,textnum),1)
        enc_embeds = self.enc_mlp(input_embeds)###context important!!
        text_embeds = self.text_embeddings(sen)


        c = Variable(torch.zeros((text_embeds.size()[0], text_embeds.size()[1], enc_embeds.size()[1])).cuda())
        for i in range(text_embeds.size()[1]):
            c[:, i, :] = enc_embeds
        sen_embeds = torch.cat((text_embeds,c),2)
        sen_len = seq_lengths.cpu().numpy()
        packed_input = pack_padded_sequence(sen_embeds,sen_len,batch_first=True)
        packeed_lstm_out, _ = self.sen_lstm(packed_input,None)
        out_vec, _ = pad_packed_sequence(packeed_lstm_out,batch_first=True)

        out_vec = out_vec.contiguous()
        out_vec = out_vec.view(sen.size()[0] * sen.size()[1],-1)

        id = []
        id.append(np.arange(0, sen_len[0]-1))
        for i in range(1, user.size()[0]):
            id.append(np.arange(0, sen_len[i] - 1) + i * sen.size()[1])
        id = np.concatenate(id)
        id = torch.LongTensor(id).cuda()
        out_vec = out_vec[id]

        out = self.dec_mlp(out_vec)
        out = self.soft(out)
        label = sen.view(-1)[id+1]
        return out,label

loss_auto = nn.NLLLoss()


import pandas as pd
import numpy as np


if isval:
    origin_train = pd.read_json(datapath)
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
    train = pd.read_json(datapath)
test = pd.read_json(datapath2)
if jieba:
    textdict = dp.text_jiebaDictionary(jiebapath, mink)  ## mink 以上
    userdict = dp.userDictionary(userpath,usermink)
else:
    user = np.array(train["user"])
    text = np.array(train["text"])
    textdict = dp.textDictionary(text)
    #userdict = dp.Dictionary(user)
    userdict = dp.userDictionary(userpath,usermink)


#train_loader =  dp.TextClassDataLoader(train, userdict,textdict, batch_size=BATCH_SIZE)
#if isval:
#    val_loader = dp.TextClassDataLoader(val, userdict,textdict, batch_size=BATCH_SIZE)
#identifyuser = dp.Identifyuser(train, userdict,textdict, 8, isval)
train_loader =  dp.TextClassDataLoader_notidentify(train, userdict,textdict, batch_size=BATCH_SIZE)
if isval:
    val_loader = dp.TextClassDataLoader_notidentify(val, userdict,textdict,batch_size=BATCH_SIZE)

train_test = pd.DataFrame(train,columns = ['user','time','weibo','text'])
if isval:
    train_test.append(val,ignore_index = True)
else:
    train_test.append(test,ignore_index = True)

train_test_loader =  dp.TestDataLoader_notidentify_train_test(train_test,userdict,textdict,batch_size=256)

val_traintest_loader =  dp.TestDataLoader_notidentify_train_test(val,userdict,textdict,batch_size=256)

model_enc = ENCModel(userdict.__len__(),textdict.__len__()).cuda()
optimizer = optim.Adam(model_enc.parameters(), lr=0.008)
print(model_enc)
for ep in range(enc_epo):
    print("epoch ",ep)
    all_loss = 0
    model_enc.train()
    for i, (bsen, seq_lengths, buser, btime,_, num4,textnum) in enumerate(train_test_loader,1):
        # measure data loading time
        #data_time.update(time.time() - end)
        bsen = Variable(bsen)
        buser = Variable(buser)
        btime = Variable(btime)
        num4 = Variable(num4)
        textnum = Variable(textnum)
        #blabel= Variable(ll.float())
        # compute output
        output,charlabel= model_enc(bsen,buser,btime, seq_lengths,num4,textnum)
        loss = loss_auto(output,charlabel)
        #print("acc:",1-loss2.data/label3.size()[0])
        #all_loss += 1-loss2.data/label3.size()[0]
        all_loss+=loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%40==0:
            print("loss ",all_loss.cpu().numpy().item()/40)
            all_loss = 0
    if isval:
        all_loss = 0
        print('validation enc')
        model_enc.eval()
        for j, (bsen, seq_lengths, buser, btime, _,num4,textnum) in enumerate(val_traintest_loader,1):
            # measure data loading time
            #data_time.update(time.time() - end)
            bsen = Variable(bsen)
            buser = Variable(buser)
            btime = Variable(btime)
            num4 = Variable(num4)
            textnum = Variable(textnum)
            #blabel= Variable(ll.float())
               # compute output
            output,charlabel= model_enc(bsen,buser,btime, seq_lengths,num4,textnum)
            loss = loss_auto(output,charlabel)
            #print("acc:",1-loss2.data/label3.size()[0])
            #all_loss += 1-loss2.data/label3.size()[0]
            all_loss+=loss.data
            if j%50==0:
                print("loss ",all_loss.cpu().numpy().item()/50)
                all_loss = 0

if not isval:
    torch.save(model_enc.state_dict(), encmodelpath)
else:
    print("validation_decoder: ")
    
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

    class EDDCModel(nn.Module):
        def __init__(self, len_userdic,len_textdic,hidden_dim = 40,hidden_dim2 = 40,final_dim = 64,
            user_embedding_dim=60,enc_dim=40,text_embedding_dim = 50,time_embedding_dim = 10,num_layers = 1):
            super(EDDCModel, self).__init__()
            self.model_name = 'Auto-Encoder'
            self.user_embedding_dim = user_embedding_dim
            self.time_embedding_dim = time_embedding_dim
            self.text_embedding_dim = text_embedding_dim
            self.user_embeddings = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
            self.time_embeddings = nn.Linear(TIME_DIM,self.time_embedding_dim,bias = False)
            self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
            self.enc_mlp = nn.Sequential(nn.Linear(self.user_embedding_dim+self.time_embedding_dim+5+17,enc_dim),
                                    nn.Dropout(0.3),nn.ReLU())

            self.senpred_lstm = nn.LSTM(input_size=self.text_embedding_dim,
                                    hidden_size=hidden_dim2, num_layers=num_layers, batch_first=True,
                                    bidirectional=False
                                    )
            self.mlp = nn.Sequential(nn.Linear(hidden_dim2+enc_dim,final_dim),
                                    nn.Dropout(0.3),nn.ReLU())
            self.mlp1 = nn.Sequential(nn.Linear(final_dim, 3),
                                         nn.Dropout(0.3))
            self.user_embeddings2 = nn.Embedding(len_userdic,self.user_embedding_dim,padding_idx=0)
            self.mlp2 = nn.Sequential(nn.Linear(self.user_embedding_dim, 3),
                                         nn.Dropout(0.3),nn.ReLU())


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
            user_embeds2 = self.user_embeddings2(user)
            label3 = label3+self.mlp2(user_embeds2)
            return label3

    modelpara = model_enc.state_dict()

    model = EDDCModel(userdict.__len__(),textdict.__len__()).cuda()
    model_dict = model.state_dict() # copy the model parameters into model_dict
    #print(model_dict)
    print(model)
    valist = []
    talist = []
    tllist=[]
    vllist=[]
    pretrained_dict = {k: v for k, v in modelpara.items() if k in model.state_dict()} # the pretrained parameter
    model_dict.update(pretrained_dict) ## update the corresponding para of model_dict as the same as pretrained para is
    model.load_state_dict(model_dict)  ## update the parameter of the model,
    count = 0
    for param in model.parameters():
        count += 1
        if count<6:
            param.requires_grad = False
        else:
            param.requires_grad = True
        #print(param.requires_grad)
    op_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(op_parameters, lr=0.01)

    loss_pred = Loss_3()
    for ep in range(n_epoches):
        print("epoch",ep)
        all_loss = 0
        accracy = 0
        model.train()
        for i, (bsen, seq_lengths, buser, btime, ll,num4,textnum) in enumerate(train_loader,1):
            # measure data loading time
            #data_time.update(time.time() - end)
            bsen = Variable(bsen)
            buser = Variable(buser)
            btime = Variable(btime)
            num4 = Variable(num4)
            textnum = Variable(textnum)
            blabel= Variable(ll.float(),requires_grad=False)
            # compute output
            label3= model(bsen,buser,btime, seq_lengths,num4,textnum)
            loss,accu = loss_pred(label3,blabel)
            #print("acc:",1-loss2.data/label3.size()[0])
            #all_loss += 1-loss2.data/label3.size()[0]
            all_loss += loss.data
            accracy += accu.data
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%50==0:
                #print("a acc ",all_loss.numpy().item()/1)
                print("a acc ", accracy.cpu().numpy().item() / 50,"loss ", all_loss.cpu().numpy().item() / 50)
                tllist.append(all_loss.cpu().numpy().item()/50)
                talist.append(accracy.cpu().numpy().item()/50)
                all_loss = 0
                accracy = 0
        if isval:
            print("validation result ")
            model.eval()
            all_loss = 0
            accracy = 0
            for i, (bsen, seq_lengths, buser, btime, ll,num4,textnum) in enumerate(val_loader,1):
                # measure data loading time
                # data_time.update(time.time() - end)
                bsen = Variable(bsen)
                buser = Variable(buser)
                btime = Variable(btime)
                blabel = Variable(ll.float(), requires_grad=False)
                num4 = Variable(num4)
                textnum = Variable(textnum)
                # compute output
                label3 = model(bsen, buser, btime, seq_lengths,num4,textnum)
                loss, accu = loss_pred(label3, blabel)
                # print("acc:",1-loss2.data/label3.size()[0])
                # all_loss += 1-loss2.data/label3.size()[0]
                all_loss += loss.data
                accracy += accu.data
                # print(loss)
            
            print("a acc ", accracy.cpu().numpy().item() / i,"loss ", all_loss.cpu().numpy().item() / i)
            vllist.append(all_loss.cpu().numpy().item()/i)
            valist.append(accracy.cpu().numpy().item()/i)
            all_loss = 0
            accracy = 0
    print(torch.Tensor([tllist,vllist,talist,valist]))
    if save:
        test_loader = dp.TestDataLoader_notidentify(test,userdict,textdict,batch_size=BATCH_SIZE)
        sen_result = []
        model.eval()
        for i, (bsen, seq_lengths,buser,btime,perm_id,_,_) in enumerate(test_loader,1):
            bsen = Variable(bsen)
            buser = Variable(buser)
            btime = Variable(btime)
            oid,oid_in_perm = perm_id.sort(0, descending = False)
            label3 = model(bsen,buser,btime, seq_lengths)#input_em :(len(input_em)-1)
            label3 = label3[oid_in_perm]
            label3 = label3.data
            label3[label3 < 0] = 0
            sen_result.append(label3.cpu().numpy())
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





