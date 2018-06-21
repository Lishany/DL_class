__author__ = 'lishanyu'

import pandas as pd
import numpy as np
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from sklearn.utils import shuffle
import math

class Dictionary(object):
    def __init__(self,path,mink = 20):
        self.word2idx = {}
        self.idx2word = []
        self.word2num = {}
        self.add_word("UNKNOWN")
        with open(path, 'r')as df:
            for line in df:
                tmp = line.split(" ")
                tmp[1] = int(tmp[1])
                if tmp[1] < mink:
                    break
                flow = float(tmp[2])+float(tmp[3])+float(tmp[4])
                if flow == 0:
                    if tmp[1] < 10:
                        continue
                self.add_word(tmp[0])
                #self.word2num[tmp[0]] = [math.log(tmp[1]),float(tmp[2]),float(tmp[3]),float(tmp[4])]
                self.word2num[tmp[0]] = [math.log(tmp[1]),float(tmp[2]),float(tmp[3]),
                float(tmp[4]),math.log(float(tmp[5])+1),math.log(float(tmp[6])+1),math.log(float(tmp[7])+1),
                math.log(float(tmp[8])+1),math.log(float(tmp[9])+1),math.log(float(tmp[10])+1),
                float(tmp[11]),float(tmp[12]),float(tmp[13]),
                math.log(float(tmp[14])+1),math.log(float(tmp[15])+1),math.log(float(tmp[16])+1),
                math.log(float(tmp[17])+1)]
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    def __len__(self):
        return len(self.idx2word)

class userDictionary(object):
    def __init__(self,path,mink = 20):
        self.word2idx = {}
        self.idx2word = []
        self.word2num = {}
        self.add_word("UNKNOWN")
        self.add_word("IMP")
        with open(path, 'r')as df:
            for line in df:
                tmp = line.split(" ")
                tmp[1] = int(tmp[1])
                if tmp[1] < mink:
                    break
                flow = float(tmp[2])+float(tmp[3])+float(tmp[4])
                if flow == 0:
                    if tmp[1] < 20:
                        continue
                self.add_word(tmp[0])
                #self.word2num[tmp[0]] = [math.log(tmp[1]),float(tmp[2]),float(tmp[3]),float(tmp[4])]
                self.word2num[tmp[0]] = [math.log(tmp[1]),float(tmp[2]),float(tmp[3]),
                float(tmp[4]),math.log(float(tmp[5])+1),math.log(float(tmp[6])+1),math.log(float(tmp[7])+1),
                math.log(float(tmp[8])+1),math.log(float(tmp[9])+1),math.log(float(tmp[10])+1),
                float(tmp[11]),float(tmp[12]),float(tmp[13]),
                math.log(float(tmp[14])+1),math.log(float(tmp[15])+1),math.log(float(tmp[16])+1),
                math.log(float(tmp[17])+1)]
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    def __len__(self):
        return len(self.idx2word)



class text_jiebaDictionary(object):
    def __init__(self,path,mink):
        self.word2idx = {}
        self.idx2word = []
        self.word2count = {}
        self.idx2word.append('\n')
        self.word2count['\n'] = 1
        self.word2idx['\n'] = len(self.idx2word) - 1
        #self.idx2word.append("UnknOwn")
        #self.word2count["UnknOwn"] = 1
        #self.word2idx["UnknOwn"] = len(self.idx2word) - 1
        with open(path, 'r')as df:
            for line in df:
                # 如果这行是换行符就跳过，这里用'\n'的长度来找空行
                tmp = line.split("\t")
                tmp[1] = int(tmp[1])
                if len(tmp[0])<2:
                    continue
                if tmp[1] < mink:
                    break
                if tmp[0] not in self.word2idx:
                    self.idx2word.append(tmp[0])
                    self.word2idx[tmp[0]] = len(self.idx2word) - 1
                    self.word2count[tmp[0]] = tmp[1]
    def __len__(self):
        return len(self.idx2word)

#textdict = text_jiebaDictionary("/Users/lishanyu/Desktop/hw2/jieba_frequency.txt",5)## mink 以上

#textdict.word2count.__len__()
#textdict.idx2word[3]


def time_convert(timetext):
    #temp = np.zeros(12 + 31 + 24)
    #temp[int(timetext[5:7])-1] = 1
    temp = np.zeros(31 + 24)
    #temp[int(timetext[5:7])-1] = 1
    #temp[int(timetext[8:10])+11] = 1
    #temp[int(timetext[11:13])+43] = 1
    temp[int(timetext[8:10])-1] = 1
    temp[int(timetext[11:13])+31] = 1
    return temp
monthday = [0,31,28,31,30,31,30,31,31,30,31,30,31]
def time_convert1(timetext):
    #temp = np.zeros(12 + 31 + 24)
    #temp[int(timetext[5:7])-1] = 1
    temp = np.zeros(7+2 + 24)
    day = int(timetext[8:10])
    month = int(timetext[5:7])
    if month == 2:
        temp[day%7]=1
    else:
        temp[(sum(monthday[2:month])+day)%7]=1
    if month ==2:#7 8
        if day in [11,12,13,14,18,19,20,21,22]:
            temp[7]=1
        temp[8] = 1
    elif month == 3:
        if day in [5,8,21]:
            temp[7] = 1
    elif month == 4:
        if day in [1,4,5,6]:
            temp[7] = 1
    elif month ==5:
        if day in [1,2,3,4,5,10,12]:
            temp[7] = 1
    elif month ==6:
        if day in [1,20,21,22]:
            temp[7] = 1
    elif month == 7:
        if day in [1,7,23]:
            temp[7] = 1
        temp[8] = 1
    elif month == 8:
        if day in [1,20,28]:
            temp[7] = 1
        temp[8] = 1
    temp[int(timetext[11:13])+9] = 1
    return temp

def time_convert2(timetext):
    temp = np.zeros(7 + 24)
    day = int(timetext[8:10])
    month = int(timetext[5:7])
    if month == 2:
        temp[day%7]=1
    else:
        temp[(sum(monthday[2:month])+day)%7]=1
    temp[int(timetext[11:13])+7] = 1
    return temp


def timebatch(timetext):
    return(np.array([time_convert1(item) for item in timetext]))

#datapath = '/Users/lishanyu/Desktop/HW2/train_jeiba.json'
#datapath = '/data/user45/hw2/train.json'
#train = pd.read_json(datapath)
#datapath2 = '/Users/lishanyu/Desktop/HW2/test.json'
#test = pd.read_json(datapath2)
#user = np.concatenate((np.array(train["user"]),np.array(test["user"])))
#user = np.array(train["user"])
#text = np.array(train["text"])
#time = np.array(train["time"])
#time_convert(time[0])
#try:
#    print(userdict.word2idx[test["user"][10]])
#except:
#    print(userdict.word2idx["UNKNOWN"])
#userdict = Dictionary(user)
#textdict = textDictionary(text)
#textdict = text_jiebaDictionary("/Users/lishanyu/Desktop/hw2/jieba_frequency.txt",5)## mink 以上
#print("total user:",userdict.word2idx.__len__())
#print("word count: ",textdict.word2idx.__len__())

class DataLoader(object):
    def __init__(self,text,target,textdict,batch_size = 128,shuf = False):
        textlist =[[textdict.word2idx[w] for w in sen.split(" ") if w in textdict.word2idx] for sen in text]
        self.textlist = textlist
        self.target = target
        lenlist = [len(sen) for sen in textlist]
        self.lenlist = lenlist
        self.shuf = shuf
        if shuf:
            self.shuffle_indices()
        else:
            self.index = 0
            self.batch_index = 0
        self.max_length = max(lenlist)
        self.n_batches = int(len(lenlist)/batch_size)
        self.batch_size = batch_size


    def shuffle_indices(self):
        self.indices = np.random.permutation(len(self.textlist))
        self.index = 0
        self.batch_index = 0

    def _create_batch(self):
        batch = []
        n = 0
        target = []
        lenbatch = []
        if self.shuf:
            while n < self.batch_size:
                _index = self.indices[self.index]
                batch.append(self.textlist[_index])
                target.append(self.target[_index])
                lenbatch.append(self.lenlist[_index])
                self.index += 1
                n += 1
        else:
            while n < self.batch_size:
                batch.append(self.textlist[self.index])
                target.append(self.target[self.index])
                lenbatch.append(self.lenlist[self.index])
                self.index += 1
                n += 1
        self.batch_index += 1
        lenbatch = torch.LongTensor(lenbatch)
        seq_tensor = torch.zeros((len(batch),self.max_length)).long().cuda()
        target = torch.FloatTensor(target).cuda()
        for idx, (seq, seqlen) in enumerate(zip(batch,lenbatch)):
            seq_tensor[idx,:seqlen] = torch.LongTensor(seq).cuda()
        return seq_tensor,target,lenbatch

    def __len__(self):
        return self.n_batches
    def __iter__(self):
        if self.shuf:
            self.shuffle_indices()
        else:
            self.index = 0
            self.batch_index = 0
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()


class mlpuser(nn.Module):
    def __init__(self, len_textdic,final_dim = 24,text_embedding_dim = 100):
        super(mlpuser, self).__init__()
        self.text_embedding_dim = text_embedding_dim
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(text_embedding_dim,final_dim),
                                nn.Dropout(0.5),nn.Tanh())
        self.outlayer = nn.Sequential(nn.Linear(final_dim, 1),nn.Dropout(0.5),nn.Sigmoid())
    def forward(self,sen,senlen):
        text_embeds = self.text_embeddings(sen)
        sen_embeds = torch.sum(text_embeds, 1)
        sen_embeds = torch.t(torch.t(sen_embeds)/Variable(senlen.cuda()).float())
        ###  text 可以乘tf-idf
        out = self.mlp(sen_embeds)
        out = self.outlayer(out)
        return out
'''
class mlpuser(nn.Module):
    def __init__(self, len_textdic,final_dim = 24,hidden_dim2 = 80,text_embedding_dim = 140):
        super(mlpuser, self).__init__()
        self.text_embedding_dim = text_embedding_dim
        self.text_embeddings = nn.Embedding(len_textdic, self.text_embedding_dim, padding_idx=0)
        self.senpred_lstm = nn.LSTM(input_size=self.text_embedding_dim,
                                hidden_size=hidden_dim2, num_layers=1, batch_first=True,
                                dropout = 0.2,
                                bidirectional=False
                                )
        self.mlp = nn.Sequential(nn.Linear(hidden_dim2,final_dim),
                                nn.Dropout(0.5),nn.Tanh())
        self.outlayer = nn.Sequential(nn.Linear(final_dim, 1),nn.Dropout(0.5),nn.Sigmoid())
    def forward(self,sen,senlen):
        text_embeds = self.text_embeddings(sen)
        _,(_,ht) = self.senpred_lstm(text_embeds)
        out = self.mlp(ht)
        out = self.outlayer(out)
        return out
'''
class Identifyuser(object):
    def __init__(self, train, userdict,textdict,n_epoches=3,isval = True):
        needval = isval
        imptarget = []
        flowtext = []
        print(len(train))
        for i in range(len(train)):
            record = train.iloc[i]
            if record["user"] not in userdict.word2idx:
                #print('not',i)
                flow = int(record["label1"])+int(record["label2"])+int(record["label3"])
                flowtext.append(record["text"])
                if flow <= 1:
                    imptarget.append(0)
                else:
                    imptarget.append(1)
        len_text = len(flowtext)
        if needval:
            offset = int(len_text*0.8)
            flowtext,imptarget = shuffle(flowtext,imptarget)
            traintext = flowtext[:offset]
            valtext = flowtext[offset:]
            traintarget = imptarget[:offset]
            valtarget = imptarget[offset:]
            val_loader = DataLoader(valtext,valtarget,textdict,shuf = False)
        else:
            traintext = flowtext
            traintarget = imptarget
        train_loader = DataLoader(traintext,traintarget,textdict,shuf = True)
        model_user = mlpuser(textdict.__len__()).cuda()
        optimizer_user = optim.Adam(model_user.parameters(), lr=0.008)
        loss_bi = nn.BCELoss()
        for ep in range(n_epoches):
            print("epoch ",ep)
            all_loss = 0
            model_user.train()
            for i, (bsen, target, lsen) in enumerate(train_loader,1):
                # measure data loading time
                #data_time.update(time.time() - end)
                bsen = Variable(bsen)
                target = Variable(target,volatile = True)
                label= model_user(bsen,lsen)
                loss= loss_bi(label,target)
                all_loss += loss.data
                optimizer_user.zero_grad()
                loss.backward()
                optimizer_user.step()
                if i%100==0:
                    print("loss ",all_loss.cpu().numpy().item()/100)
                    all_loss = 0
            if isval:
                print("validation result")
                model_user.eval()
                all_loss = 0
                for i, (bsen, target, lsen) in enumerate(val_loader,1):
                    # measure data loading time
                    # data_time.update(time.time() - end)
                    bsen = Variable(bsen)
                    target = Variable(target)
                    label= model_user(bsen,lsen)
                    loss= loss_bi(label,target)
                    all_loss += loss.data
                    if i%50==0:
                        print("loss ",all_loss.cpu().numpy().item()/50)
                        all_loss = 0
        self.model = model_user
        self.model.eval()
    def getuser_num(self,sen,lsen):
        tmp = self.model(Variable(torch.LongTensor(sen).unsqueeze(0).cuda()),torch.Tensor([lsen]))
        #Variable(torch.LongTensor(textlist[0]).unsqueeze(0).cuda()),torch.Tensor([lenlist[0]])
        if tmp.cpu().data.numpy() >0.5:
            return 1
        else:
            return 0

class TextClassDataLoader_notidentify(object):
    def __init__(self, train, userdict,textdict,batch_size=32):## text should be list ["sen","sen2","sen3"]
        user = np.array(train["user"])
        time = np.array(train["time"])
        text = np.array(train["text"])
        #print(text[0])
        self.userdict = userdict
        self.textdict = textdict
        self.batch_size = batch_size
        label1 = np.array(train["label1"])
        label2 = np.array(train["label2"])
        label3 = np.array(train["label3"])
        self.label = np.stack((label1,label2,label3),1)
        def trytext(w):
            try:
                a = textdict.word2idx[w]
            except:
                a = 1
            return a
        def tryuser4num(w):
            try:
                a = userdict.word2num[w]
            except:
                a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            return a
        def tryuser(w):
            try:
                a = userdict.word2idx[w]
            except:
                a = 0
            return a
        def trytextnum(sen):
            tmp = [0.0,0.0,0.0,0.0,0.0]
            tmp[0] = math.log(len(sen))
            if 'http://t.cn/' in sen:
                tmp[1] = 1.0
            if sen[0] == '#':
                tmp[2] = 1.0
            if '[' in sen and ']' in sen:
                tmp[3] = 1.0
            if '?' in sen:
                tmp[4] = 1.0
            return tmp

        textlist =[[trytext(w) for w in sen.split(" ") if w in textdict.word2idx] for sen in text]
        lenlist =[len(sen) for sen in textlist]
        userlist = [tryuser(w) for w in user]
        user4numlist = [tryuser4num(w) for w in user]
        textnum = [trytextnum(sen) for sen in text]
        timelist = [time_convert1(w) for w in time]
        self.sen = textlist
        self.user = userlist
        self.time = timelist
        self.lenlist = lenlist
        self.user4numlist = user4numlist
        self.textnum = textnum
        self.shuffle_indices()
        #print(self.batch_size)
        self.n_batches = int(len(self.user) / self.batch_size)
        self.last_size = len(self.user) % self.batch_size
        self.max_length = max(lenlist)
        self.report()
        #textlist[0]
    def shuffle_indices(self):
        self.indices = np.random.permutation(len(self.user))#句子的数量
        self.index = 0
        self.batch_index = 0

    def get_max_length(self):
        length = 0
        for sample in self.sen:
            length = max(length, len(sample))
        return length


    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] =  batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        tbatch = []
        ubatch = []
        lbatch=[]
        usernum = []
        textnum = []
        n = 0
        lenn_seq = []
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.sen[_index])
            tbatch.append(self.time[_index])
            ubatch.append(self.user[_index])
            lenn_seq.append(self.lenlist[_index])
            lbatch.append(self.label[_index])
            usernum.append(self.user4numlist[_index])
            textnum.append(self.textnum[_index])
            self.index += 1
            n += 1
        self.batch_index += 1
        string = batch#like [[1,2,3],[1,2],[3,6,2,8]]
        #label, string = tuple(zip(*batch))
        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        #print(seq_tensor.size())
        #print(seq_lengths.min())
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()
        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        user_tensor = ubatch[perm_idx]
        tbatch = torch.FloatTensor(tbatch).cuda()
        time_tensor = tbatch[perm_idx]
        usernum = torch.FloatTensor(usernum).cuda()
        usernum_tensor = usernum[perm_idx]
        textnum = torch.FloatTensor(textnum).cuda()
        textnum_tensor = textnum[perm_idx]
        lbatch = torch.from_numpy(np.array(lbatch)).float().cuda()
        # seq_tensor = seq_tensor.transpose(0, 1)
        return seq_tensor, seq_lengths, user_tensor, time_tensor, lbatch[perm_idx], usernum_tensor,textnum#seq_lengths is the resorted one

    def _create_batch_last(self):
        batch = []
        tbatch = []
        ubatch = []
        lbatch=[]
        usernum = []
        textnum = []
        n = 0
        lenn_seq = []
        while n < self.last_size:
            _index = self.indices[self.index]
            batch.append(self.sen[_index])
            tbatch.append(self.time[_index])
            ubatch.append(self.user[_index])
            lenn_seq.append(self.lenlist[_index])
            lbatch.append(self.label[_index])
            usernum.append(self.user4numlist[_index])
            textnum.append(self.textnum[_index])
            self.index += 1
            n += 1
        self.batch_index += 1
        string = batch#like [[1,2,3],[1,2],[3,6,2,8]]
        #label, string = tuple(zip(*batch))
        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        #print(seq_tensor.size())
        #print(seq_lengths.min())
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()
        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        user_tensor = ubatch[perm_idx]
        tbatch = torch.FloatTensor(tbatch).cuda()
        time_tensor = tbatch[perm_idx]
        usernum = torch.FloatTensor(usernum).cuda()
        usernum_tensor = usernum[perm_idx]
        textnum = torch.FloatTensor(textnum).cuda()
        textnum_tensor = textnum[perm_idx]
        lbatch = torch.from_numpy(np.array(lbatch)).float().cuda()
        # seq_tensor = seq_tensor.transpose(0, 1)
        return seq_tensor, seq_lengths, user_tensor, time_tensor, lbatch[perm_idx], usernum_tensor,textnum#seq_lengths is the resorted one

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.shuffle_indices()
        for i in range(self.n_batches):
            yield self._create_batch() # return tensor[] and corresponding length
        if self.batch_index == self.n_batches:#stop iteration
            if self.last_size>0:
                yield self._create_batch_last()
            raise StopIteration()

    def show_samples(self, n=2):
        for sample in self.sen[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.sen)))
        print('max len: {}'.format(self.max_length))
        print('text dict: {}'.format(self.textdict.word2idx.__len__()))
        print('user dict: {}'.format(self.userdict.word2idx.__len__()))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))


class TestDataLoader_notidentify_train_test(object):

    def __init__(self, test, userdict,textdict, batch_size=32):## text should be list ["sen","sen2","sen3"]
        user = np.array(test["user"])
        time = np.array(test["time"])
        text = np.array(test["text"])
        self.userdict = userdict
        self.textdict = textdict
        self.batch_size = batch_size
        self.index = 0
        self.batch_index = 0
        
        def trytext(w):
            try:
                a = textdict.word2idx[w]
            except:
                a = 1
            return a

        def tryuser(w):
            try:
                a = userdict.word2idx[w]
            except:
                a = 0
            return a
        def tryuser4num(w):
            try:
                a = userdict.word2num[w]
            except:
                a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            return a
        def trytextnum(sen):
            tmp = [0.0,0.0,0.0,0.0,0.0]
            tmp[0] = math.log(len(sen))
            if 'http://t.cn/' in sen:
                tmp[1] = 1.0
            if sen[0] == '#':
                tmp[2] = 1.0
            if '[' in sen and ']' in sen:
                tmp[3] = 1.0
            if '?' in sen:
                tmp[4] = 1.0
            return tmp
        textlist = [[trytext(w) for w in sen.split(" ") if w in self.textdict.word2idx] for sen in text]
        userlist = [tryuser(w) for w in user]
        lenlist = [len(sen) for sen in textlist]
        textnum = [trytextnum(sen) for sen in text]
        user4numlist = [tryuser4num(w) for w in user]
        timelist = [time_convert1(w) for w in time]
        self.sen = textlist
        self.user = userlist
        self.time = timelist
        self.lenlist = lenlist
        self.user4numlist = user4numlist
        self.textnum = textnum
        self.shuffle_indices()
        self.n_batches = int(len(self.user) / self.batch_size)
        self.last_size = len(self.user) % self.batch_size
        self.max_length = max(lenlist)
        self.report()


    def _create_batch(self):
        batch = []
        tbatch = []
        ubatch = []
        n = 0
        lenn_seq = []
        usernum = []
        textnum = []
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.sen[_index])
            tbatch.append(self.time[_index])
            ubatch.append(self.user[_index])
            lenn_seq.append(self.lenlist[_index])
            usernum.append(self.user4numlist[_index])
            textnum.append(self.textnum[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch  # like [[1,2,3],[1,2],[3,6,2,8]]
        # label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        tbatch = torch.FloatTensor(tbatch).cuda()
        usernum = torch.FloatTensor(usernum).cuda()
        textnum = torch.FloatTensor(textnum).cuda()
        user_tensor = ubatch[perm_idx]
        time_tensor = tbatch[perm_idx]
        usernum_tensor = usernum[perm_idx]
        textnum_tensor = textnum[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, perm_idx,usernum_tensor,textnum_tensor  # seq_lengths is the resorted one

    def shuffle_indices(self):
        self.indices = np.random.permutation(len(self.user))#句子的数量
        self.index = 0
        self.batch_index = 0

    def _create_batch_last(self):
        batch = []
        tbatch = []
        ubatch = []
        n = 0
        lenn_seq = []
        usernum = []
        textnum = []
        while n < self.last_size:
            _index = self.indices[self.index]
            batch.append(self.sen[_index])
            tbatch.append(self.time[_index])
            ubatch.append(self.user[_index])
            lenn_seq.append(self.lenlist[_index])
            usernum.append(self.user4numlist[_index])
            textnum.append(self.textnum[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch  # like [[1,2,3],[1,2],[3,6,2,8]]
        # label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        tbatch = torch.FloatTensor(tbatch).cuda()
        usernum = torch.FloatTensor(usernum).cuda()
        textnum = torch.FloatTensor(textnum).cuda()
        user_tensor = ubatch[perm_idx]
        time_tensor = tbatch[perm_idx]
        usernum_tensor = usernum[perm_idx]
        textnum_tensor = textnum[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, perm_idx,usernum_tensor,textnum_tensor  # seq_lengths is the resorted one

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.shuffle_indices()
        for i in range(self.n_batches):
            yield self._create_batch() # return tensor[] and corresponding length
        if self.batch_index == self.n_batches:#stop iteration
            if self.last_size > 0:
                yield self._create_batch_last()
            raise StopIteration()

    def show_samples(self, n=2):
        for sample in self.sen[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.sen)))
        print('max len: {}'.format(self.max_length))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))


class TestDataLoader_notidentify(object):

    def __init__(self, test, userdict,textdict, batch_size=32):## text should be list ["sen","sen2","sen3"]
        user = np.array(test["user"])
        time = np.array(test["time"])
        text = np.array(test["text"])
        self.userdict = userdict
        self.textdict = textdict
        self.batch_size = batch_size
        self.index = 0
        self.batch_index = 0
        
        def trytext(w):
            try:
                a = textdict.word2idx[w]
            except:
                a = 1
            return a

        def tryuser(w):
            try:
                a = userdict.word2idx[w]
            except:
                a = 0
            return a
        def tryuser4num(w):
            try:
                a = userdict.word2num[w]
            except:
                a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            return a
        def trytextnum(sen):
            tmp = [0.0,0.0,0.0,0.0,0.0]
            tmp[0] = math.log(len(sen))
            if 'http://t.cn/' in sen:
                tmp[1] = 1.0
            if sen[0] == '#':
                tmp[2] = 1.0
            if '[' in sen and ']' in sen:
                tmp[3] = 1.0
            if '?' in sen:
                tmp[4] = 1.0
            return tmp
        textlist = [[trytext(w) for w in sen.split(" ") if w in self.textdict.word2idx] for sen in text]
        userlist = [tryuser(w) for w in user]
        lenlist = [len(sen) for sen in textlist]
        textnum = [trytextnum(sen) for sen in text]
        user4numlist = [tryuser4num(w) for w in user]
        timelist = [time_convert1(w) for w in time]
        self.sen = textlist
        self.user = userlist
        self.time = timelist
        self.lenlist = lenlist
        self.user4numlist = user4numlist
        self.textnum = textnum
        self.n_batches = int(len(self.user) / self.batch_size)
        self.last_size = len(self.user) % self.batch_size
        self.max_length = max(lenlist)
        self.report()


    def _create_batch(self):
        batch = []
        tbatch = []
        ubatch = []
        n = 0
        lenn_seq = []
        usernum = []
        textnum = []
        while n < self.batch_size:
            batch.append(self.sen[self.index])
            tbatch.append(self.time[self.index])
            ubatch.append(self.user[self.index])
            lenn_seq.append(self.lenlist[self.index])
            usernum.append(self.user4numlist[self.index])
            textnum.append(self.textnum[self.index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch  # like [[1,2,3],[1,2],[3,6,2,8]]
        # label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        tbatch = torch.FloatTensor(tbatch).cuda()
        usernum = torch.FloatTensor(usernum).cuda()
        textnum = torch.FloatTensor(textnum).cuda()
        user_tensor = ubatch[perm_idx]
        time_tensor = tbatch[perm_idx]
        usernum_tensor = usernum[perm_idx]
        textnum_tensor = textnum[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, perm_idx,usernum_tensor,textnum_tensor  # seq_lengths is the resorted one


    def _create_batch_last(self):
        batch = []
        tbatch = []
        ubatch = []
        n = 0
        lenn_seq = []
        usernum = []
        textnum = []
        while n < self.last_size:
            batch.append(self.sen[self.index])
            tbatch.append(self.time[self.index])
            ubatch.append(self.user[self.index])
            lenn_seq.append(self.lenlist[self.index])
            usernum.append(self.user4numlist[self.index])
            textnum.append(self.textnum[self.index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch  # like [[1,2,3],[1,2],[3,6,2,8]]
        # label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        tbatch = torch.FloatTensor(tbatch).cuda()
        usernum = torch.FloatTensor(usernum).cuda()
        textnum = torch.FloatTensor(textnum).cuda()
        user_tensor = ubatch[perm_idx]
        time_tensor = tbatch[perm_idx]
        usernum_tensor = usernum[perm_idx]
        textnum_tensor = textnum[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, perm_idx,usernum_tensor,textnum_tensor  # seq_lengths is the resorted one

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for i in range(self.n_batches):
            yield self._create_batch() # return tensor[] and corresponding length
        if self.batch_index == self.n_batches:#stop iteration
            if self.last_size > 0:
                yield self._create_batch_last()
            raise StopIteration()

    def show_samples(self, n=2):
        for sample in self.sen[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.sen)))
        print('max len: {}'.format(self.max_length))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))



class TextClassDataLoader(object):
    def __init__(self, train, userdict,textdict, identifyuser,batch_size=32):## text should be list ["sen","sen2","sen3"]
        user = np.array(train["user"])
        time = np.array(train["time"])
        text = np.array(train["text"])
        self.userdict = userdict
        self.textdict = textdict
        self.batch_size = batch_size
        label1 = np.array(train["label1"])
        label2 = np.array(train["label2"])
        label3 = np.array(train["label3"])
        self.label = np.stack((label1,label2,label3),1)
        def trytext(w):
            try:
                a = textdict.word2idx[w]
            except:
                a = 1
            return a
        def tryuser4num(w):
            try:
                a = userdict.word2num[w]
            except:
                a = [1,0,0,0]
            return a
        def tryuser(w):
            try:
                a = userdict.word2idx[w]
            except:
                a = 0
            return a
        def trytextnum(sen):
            tmp = [0.0,0.0,0.0,0.0,0.0]
            tmp[0] = math.log(len(sen))
            if 'http://t.cn/' in sen:
                tmp[1] = 1.0
            if sen[0] == '#':
                tmp[2] = 1.0
            if '[' in sen and ']' in sen:
                tmp[3] = 1.0
            if '?' in sen:
                tmp[4] = 1.0
            return tmp

        textlist =[[trytext(w) for w in sen.split(" ") if w in textdict.word2idx] for sen in text]
        lenlist =[len(sen) for sen in textlist]
        userlist = [tryuser(w) for w in user]
        user4numlist = [tryuser4num(w) for w in user]
        textnum = [trytextnum(sen) for sen in text]
        for i in range(len(userlist)):
            if userlist[i] > 0:
                continue
            #print(textlist[i])
            #print(lenlist[i])
            userlist[i] = identifyuser.getuser_num(textlist[i],lenlist[i])
            if userlist[i] == 1:
                user4numlist[i] = [math.log(20),3,2,3]

        

        timelist = [time_convert1(w) for w in time]
        self.sen = textlist
        self.user = userlist
        self.time = timelist
        self.lenlist = lenlist
        self.user4numlist = user4numlist
        self.textnum = textnum
        self.shuffle_indices()
        #print(self.batch_size)
        self.n_batches = int(len(self.user) / self.batch_size)
        self.max_length = max(lenlist)
        self.report()
        #textlist[0]
    def shuffle_indices(self):
        self.indices = np.random.permutation(len(self.user))#句子的数量
        self.index = 0
        self.batch_index = 0

    def get_max_length(self):
        length = 0
        for sample in self.sen:
            length = max(length, len(sample))
        return length


    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] =  batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        tbatch = []
        ubatch = []
        lbatch=[]
        usernum = []
        textnum = []
        n = 0
        lenn_seq = []
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.sen[_index])
            tbatch.append(self.time[_index])
            ubatch.append(self.user[_index])
            lenn_seq.append(self.lenlist[_index])
            lbatch.append(self.label[_index])
            usernum.append(self.user4numlist[_index])
            textnum.append(self.textnum[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch#like [[1,2,3],[1,2],[3,6,2,8]]
        #label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        user_tensor = ubatch[perm_idx]
        tbatch = torch.FloatTensor(tbatch).cuda()
        time_tensor = tbatch[perm_idx]
        usernum = torch.FloatTensor(usernum).cuda()
        usernum_tensor = usernum[perm_idx]
        textnum = torch.FloatTensor(textnum).cuda()
        textnum_tensor = textnum[perm_idx]

        lbatch = torch.from_numpy(np.array(lbatch)).float().cuda()
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, lbatch[perm_idx], usernum_tensor,textnum#seq_lengths is the resorted one

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.shuffle_indices()
        for i in range(self.n_batches):
            yield self._create_batch() # return tensor[] and corresponding length
        if self.batch_index == self.n_batches:#stop iteration
            if self.last_size > 0:
                yield self._create_batch_last()
            raise StopIteration()
            
    def show_samples(self, n=2):
        for sample in self.sen[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.sen)))
        print('max len: {}'.format(self.max_length))
        print('text dict: {}'.format(self.textdict.word2idx.__len__()))
        print('user dict: {}'.format(self.userdict.word2idx.__len__()))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))


class TestDataLoader(object):

    def __init__(self, test, userdict,textdict, identifyuser,batch_size=32):## text should be list ["sen","sen2","sen3"]
        user = np.array(test["user"])
        time = np.array(test["time"])
        text = np.array(test["text"])
        self.userdict = userdict
        self.textdict = textdict
        self.batch_size = batch_size
        self.index = 0
        self.batch_index = 0
        def trytext(w):
            try:
                a = textdict.word2idx[w]
            except:
                a = 1
            return a

        def tryuser(w):
            try:
                a = userdict.word2idx[w]
            except:
                a = 0
            return a
        def tryuser4num(w):
            try:
                a = userdict.word2num[w]
            except:
                a = [1,0,0,0]
            return a
        def trytextnum(sen):
            tmp = [0.0,0.0,0.0,0.0,0.0]
            tmp[0] = math.log(len(sen))
            if 'http://t.cn/' in sen:
                tmp[1] = 1.0
            if sen[0] == '#':
                tmp[2] = 1.0
            if '[' in sen and ']' in sen:
                tmp[3] = 1.0
            if '?' in sen:
                tmp[4] = 1.0
            return tmp
        textlist = [[trytext(w) for w in sen.split(" ") if w in self.textdict.word2idx] for sen in text]
        userlist = [tryuser(w) for w in user]
        lenlist = [len(sen) for sen in textlist]
        textnum = [trytextnum(sen) for sen in text]
        user4numlist = [tryuser4num(w) for w in user]
        for i in range(len(userlist)):
            if userlist[i] > 0:
                continue
            userlist[i] = identifyuser.getuser_num(textlist[i],lenlist[i])
            if userlist[i] == 1:
                user4numlist[i] = [math.log(20),3,2,3]
        timelist = [time_convert1(w) for w in time]
        self.sen = textlist
        self.user = userlist
        self.time = timelist
        self.lenlist = lenlist
        self.user4numlist = user4numlist
        self.textnum = textnum
        self.n_batches = int(len(self.user) / self.batch_size)
        self.max_length = max(lenlist)
        self.report()


    def _create_batch(self):
        batch = []
        tbatch = []
        ubatch = []
        n = 0
        lenn_seq = []
        usernum = []
        textnum = []
        while n < self.batch_size:
            batch.append(self.sen[self.index])
            tbatch.append(self.time[self.index])
            ubatch.append(self.user[self.index])
            lenn_seq.append(self.lenlist[self.index])
            usernum.append(self.user4numlist[self.index])
            textnum.append(self.textnum[self.index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch  # like [[1,2,3],[1,2],[3,6,2,8]]
        # label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        tbatch = torch.FloatTensor(tbatch).cuda()
        usernum = torch.FloatTensor(usernum).cuda()
        textnum = torch.FloatTensor(textnum).cuda()
        user_tensor = ubatch[perm_idx]
        time_tensor = tbatch[perm_idx]
        usernum_tensor = usernum[perm_idx]
        textnum_tensor = textnum[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, perm_idx,usernum_tensor,textnum_tensor  # seq_lengths is the resorted one

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:#stop iteration
                raise StopIteration()
            yield self._create_batch() # return tensor[] and corresponding length

    def show_samples(self, n=2):
        for sample in self.sen[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.sen)))
        print('max len: {}'.format(self.max_length))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))


#train_loader = TextClassDataLoader(train, userdict,textdict, batch_size=32)

#test_loader = TestDataLoader(test,userdict,textdict,batch_size=1)


'''
for i, (bsen, seq_lengths,buser,btime,l1,l2,l3) in enumerate(train_loader):
    if i > 0:
        break

for i, (bsen, seq_lengths,buser,btime,pid) in enumerate(test_loader):
    print(i)
    if i == 0:
        break
'''



class TextClassDataLoadernotforrnn(object):
    def __init__(self, train, userdict,textdict, maxlen = 0,batch_size=32):## text should be list ["sen","sen2","sen3"]
        user = np.array(train["user"])
        time = np.array(train["time"])
        text = np.array(train["text"])
        self.userdict = userdict
        self.textdict = textdict
        self.batch_size = batch_size
        label1 = np.array(train["label1"])
        label2 = np.array(train["label2"])
        label3 = np.array(train["label3"])
        self.label = np.stack((label1,label2,label3),1)
        def trytext(w):
            try:
                a = textdict.word2idx[w]
            except:
                a = 1
            return a
        def tryuser4num(w):
            try:
                a = userdict.word2num[w]
            except:
                a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            return a
        def tryuser(w):
            try:
                a = userdict.word2idx[w]
            except:
                a = 0
            return a
        def trytextnum(sen):
            tmp = [0.0,0.0,0.0,0.0,0.0]
            tmp[0] = math.log(len(sen))
            if 'http://t.cn/' in sen:
                tmp[1] = 1.0
            if sen[0] == '#':
                tmp[2] = 1.0
            if '[' in sen and ']' in sen:
                tmp[3] = 1.0
            if '?' in sen:
                tmp[4] = 1.0
            return tmp

        textlist =[[trytext(w) for w in sen.split(" ") if w in textdict.word2idx] for sen in text]
        lenlist =[len(sen) for sen in textlist]
        userlist = [tryuser(w) for w in user]
        user4numlist = [tryuser4num(w) for w in user]
        textnum = [trytextnum(sen) for sen in text]

        timelist = [time_convert1(w) for w in time]
        self.sen = textlist
        self.user = userlist
        self.time = timelist
        self.lenlist = lenlist
        self.user4numlist = user4numlist
        self.textnum = textnum
        self.shuffle_indices()
        #print(self.batch_size)
        self.n_batches = int(len(self.user) / self.batch_size)
        self.last_size = int(len(self.user) % self.batch_size)
        self.max_length = max(lenlist)
        if maxlen > 0:
            self.max_length = maxlen
        self.report()
        #textlist[0]
    def shuffle_indices(self):
        self.indices = np.random.permutation(len(self.user))#句子的数量
        self.index = 0
        self.batch_index = 0

    def get_max_length(self):
        length = 0
        for sample in self.sen:
            length = max(length, len(sample))
        return length


    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] =  batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        tbatch = []
        ubatch = []
        lbatch=[]
        usernum = []
        textnum = []
        n = 0
        lenn_seq = []
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.sen[_index])
            tbatch.append(self.time[_index])
            ubatch.append(self.user[_index])
            lenn_seq.append(self.lenlist[_index])
            lbatch.append(self.label[_index])
            usernum.append(self.user4numlist[_index])
            textnum.append(self.textnum[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch#like [[1,2,3],[1,2],[3,6,2,8]]
        #label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), self.max_length)).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        user_tensor = ubatch[perm_idx]
        tbatch = torch.FloatTensor(tbatch).cuda()
        time_tensor = tbatch[perm_idx]
        usernum = torch.FloatTensor(usernum).cuda()
        usernum_tensor = usernum[perm_idx]
        textnum = torch.FloatTensor(textnum).cuda()
        textnum_tensor = textnum[perm_idx]

        lbatch = torch.from_numpy(np.array(lbatch)).float().cuda()
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, lbatch[perm_idx], usernum_tensor,textnum#seq_lengths is the resorted one

    def _create_batch_last(self):
        batch = []
        tbatch = []
        ubatch = []
        lbatch=[]
        usernum = []
        textnum = []
        n = 0
        lenn_seq = []
        while n < self.last_size:
            _index = self.indices[self.index]
            batch.append(self.sen[_index])
            tbatch.append(self.time[_index])
            ubatch.append(self.user[_index])
            lenn_seq.append(self.lenlist[_index])
            lbatch.append(self.label[_index])
            usernum.append(self.user4numlist[_index])
            textnum.append(self.textnum[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch#like [[1,2,3],[1,2],[3,6,2,8]]
        #label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), self.max_length)).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        user_tensor = ubatch[perm_idx]
        tbatch = torch.FloatTensor(tbatch).cuda()
        time_tensor = tbatch[perm_idx]
        usernum = torch.FloatTensor(usernum).cuda()
        usernum_tensor = usernum[perm_idx]
        textnum = torch.FloatTensor(textnum).cuda()
        textnum_tensor = textnum[perm_idx]

        lbatch = torch.from_numpy(np.array(lbatch)).float().cuda()
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, lbatch[perm_idx], usernum_tensor,textnum#seq_lengths is the resorted one

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.shuffle_indices()
        for i in range(self.n_batches):
            yield self._create_batch() # return tensor[] and corresponding length
        if self.batch_index == self.n_batches:#stop iteration
            if self.last_size > 0:
                yield self._create_batch_last()
            raise StopIteration()

    def show_samples(self, n=2):
        for sample in self.sen[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.sen)))
        print('max len: {}'.format(self.max_length))
        print('text dict: {}'.format(self.textdict.word2idx.__len__()))
        print('user dict: {}'.format(self.userdict.word2idx.__len__()))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))


class TestDataLoadernotforrnn(object):

    def __init__(self, lenmax,test, userdict,textdict,batch_size=32):## text should be list ["sen","sen2","sen3"]
        user = np.array(test["user"])
        time = np.array(test["time"])
        text = np.array(test["text"])
        self.userdict = userdict
        self.textdict = textdict
        self.batch_size = batch_size
        self.index = 0
        self.batch_index = 0
        self.max_length = lenmax
        def trytext(w):
            try:
                a = textdict.word2idx[w]
            except:
                a = 1
            return a

        def tryuser(w):
            try:
                a = userdict.word2idx[w]
            except:
                a = 0
            return a
        def tryuser4num(w):
            try:
                a = userdict.word2num[w]
            except:
                a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            return a
        def trytextnum(sen):
            tmp = [0.0,0.0,0.0,0.0,0.0]
            tmp[0] = math.log(len(sen))
            if 'http://t.cn/' in sen:
                tmp[1] = 1.0
            if sen[0] == '#':
                tmp[2] = 1.0
            if '[' in sen and ']' in sen:
                tmp[3] = 1.0
            if '?' in sen:
                tmp[4] = 1.0
            return tmp
        textlist = [[trytext(w) for w in sen.split(" ") if w in self.textdict.word2idx] for sen in text]
        userlist = [tryuser(w) for w in user]
        lenlist = [len(sen) for sen in textlist]
        textnum = [trytextnum(sen) for sen in text]
        user4numlist = [tryuser4num(w) for w in user]

        timelist = [time_convert1(w) for w in time]
        self.sen = textlist
        self.user = userlist
        self.time = timelist
        self.lenlist = lenlist
        self.user4numlist = user4numlist
        self.textnum = textnum
        self.n_batches = int(len(self.user) / self.batch_size)
        self.last_size = int(len(self.user) % self.batch_size)
        self.max_length = lenmax
        self.report()


    def _create_batch(self):
        batch = []
        tbatch = []
        ubatch = []
        n = 0
        lenn_seq = []
        usernum = []
        textnum = []
        while n < self.batch_size:
            batch.append(self.sen[self.index])
            tbatch.append(self.time[self.index])
            ubatch.append(self.user[self.index])
            lenn_seq.append(self.lenlist[self.index])
            usernum.append(self.user4numlist[self.index])
            textnum.append(self.textnum[self.index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch  # like [[1,2,3],[1,2],[3,6,2,8]]
        # label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), self.max_length)).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        tbatch = torch.FloatTensor(tbatch).cuda()
        usernum = torch.FloatTensor(usernum).cuda()
        textnum = torch.FloatTensor(textnum).cuda()
        user_tensor = ubatch[perm_idx]
        time_tensor = tbatch[perm_idx]
        usernum_tensor = usernum[perm_idx]
        textnum_tensor = textnum[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, perm_idx,usernum_tensor,textnum_tensor  # seq_lengths is the resorted one

    def _create_batch_last(self):
        batch = []
        tbatch = []
        ubatch = []
        n = 0
        lenn_seq = []
        usernum = []
        textnum = []
        while n < self.last_size:
            batch.append(self.sen[self.index])
            tbatch.append(self.time[self.index])
            ubatch.append(self.user[self.index])
            lenn_seq.append(self.lenlist[self.index])
            usernum.append(self.user4numlist[self.index])
            textnum.append(self.textnum[self.index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch  # like [[1,2,3],[1,2],[3,6,2,8]]
        # label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), self.max_length)).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        tbatch = torch.FloatTensor(tbatch).cuda()
        usernum = torch.FloatTensor(usernum).cuda()
        textnum = torch.FloatTensor(textnum).cuda()
        user_tensor = ubatch[perm_idx]
        time_tensor = tbatch[perm_idx]
        usernum_tensor = usernum[perm_idx]
        textnum_tensor = textnum[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, perm_idx,usernum_tensor,textnum_tensor  # seq_lengths is the resorted one

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for i in range(self.n_batches):
            yield self._create_batch() # return tensor[] and corresponding length
        if self.batch_index == self.n_batches:#stop iteration
            if self.last_size > 0:
                yield self._create_batch_last()
            raise StopIteration()

    def show_samples(self, n=2):
        for sample in self.sen[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.sen)))
        print('max len: {}'.format(self.max_length))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))




class TextClassDataLoader_class(object):
    def __init__(self, train, userdict,textdict,flow,batch_size=32):## text should be list ["sen","sen2","sen3"]
        user = np.array(train["user"])
        time = np.array(train["time"])
        text = np.array(train["text"])
        self.userdict = userdict
        self.textdict = textdict
        self.batch_size = batch_size
        self.label = flow
        def trytext(w):
            try:
                a = textdict.word2idx[w]
            except:
                a = 1
            return a
        def tryuser4num(w):
            try:
                a = userdict.word2num[w]
            except:
                #a = [1,0,0,0]
                a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            return a
        def tryuser(w):
            try:
                a = userdict.word2idx[w]
            except:
                a = 0
            return a
        def trytextnum(sen):
            tmp = [0.0,0.0,0.0,0.0,0.0]
            tmp[0] = math.log(len(sen))
            if 'http://t.cn/' in sen:
                tmp[1] = 1.0
            if sen[0] == '#':
                tmp[2] = 1.0
            if '[' in sen and ']' in sen:
                tmp[3] = 1.0
            if '?' in sen:
                tmp[4] = 1.0
            return tmp
        textlist =[[trytext(w) for w in sen.split(" ") if w in textdict.word2idx] for sen in text]
        lenlist =[len(sen) for sen in textlist]
        userlist = [tryuser(w) for w in user]
        user4numlist = [tryuser4num(w) for w in user]
        textnum = [trytextnum(sen) for sen in text]
        timelist = [time_convert2(w) for w in time]
        self.sen = textlist
        self.user = userlist
        self.time = timelist
        self.lenlist = lenlist
        self.user4numlist = user4numlist
        self.textnum = textnum
        self.shuffle_indices()
        #print(self.batch_size)
        self.n_batches = int(len(self.user) / self.batch_size)
        self.last_size = int(len(self.user) % self.batch_size)
        self.max_length = max(lenlist)
        self.report()
        #textlist[0]
    def shuffle_indices(self):
        self.indices = np.random.permutation(len(self.user))#句子的数量
        self.index = 0
        self.batch_index = 0
    def get_max_length(self):
        length = 0
        for sample in self.sen:
            length = max(length, len(sample))
        return length
    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] =  batch_x[i] + [0 for _ in range(missing)]
        return batch_x
    def _create_batch(self):
        batch = []
        tbatch = []
        ubatch = []
        lbatch=[]
        usernum = []
        textnum = []
        n = 0
        lenn_seq = []
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.sen[_index])
            tbatch.append(self.time[_index])
            ubatch.append(self.user[_index])
            lenn_seq.append(self.lenlist[_index])
            lbatch.append(self.label[_index])
            usernum.append(self.user4numlist[_index])
            textnum.append(self.textnum[_index])
            self.index += 1
            n += 1
        self.batch_index += 1
        string = batch#like [[1,2,3],[1,2],[3,6,2,8]]
        seq_lengths = torch.LongTensor(lenn_seq).cuda()
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()
        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        user_tensor = ubatch[perm_idx]
        tbatch = torch.FloatTensor(tbatch).cuda()
        time_tensor = tbatch[perm_idx]
        usernum = torch.FloatTensor(usernum).cuda()
        usernum_tensor = usernum[perm_idx]
        textnum = torch.FloatTensor(textnum).cuda()
        textnum_tensor = textnum[perm_idx]
        lbatch = torch.LongTensor(lbatch).cuda()
        # seq_tensor = seq_tensor.transpose(0, 1)
        return seq_tensor, seq_lengths, user_tensor, time_tensor, lbatch[perm_idx], usernum_tensor,textnum#seq_lengths is the resorted one
    def _create_batch_last(self):
        batch = []
        tbatch = []
        ubatch = []
        lbatch=[]
        usernum = []
        textnum = []
        n = 0
        lenn_seq = []
        while n < self.last_size:
            _index = self.indices[self.index]
            batch.append(self.sen[_index])
            tbatch.append(self.time[_index])
            ubatch.append(self.user[_index])
            lenn_seq.append(self.lenlist[_index])
            lbatch.append(self.label[_index])
            usernum.append(self.user4numlist[_index])
            textnum.append(self.textnum[_index])
            self.index += 1
            n += 1
        self.batch_index += 1
        string = batch#like [[1,2,3],[1,2],[3,6,2,8]]
        seq_lengths = torch.LongTensor(lenn_seq).cuda()
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()
        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        user_tensor = ubatch[perm_idx]
        tbatch = torch.FloatTensor(tbatch).cuda()
        time_tensor = tbatch[perm_idx]
        usernum = torch.FloatTensor(usernum).cuda()
        usernum_tensor = usernum[perm_idx]
        textnum = torch.FloatTensor(textnum).cuda()
        textnum_tensor = textnum[perm_idx]
        lbatch = torch.LongTensor(lbatch).cuda()
        # seq_tensor = seq_tensor.transpose(0, 1)
        return seq_tensor, seq_lengths, user_tensor, time_tensor, lbatch[perm_idx], usernum_tensor,textnum#seq_lengths is the resorted one


    def __len__(self):
        return self.n_batches
    def __iter__(self):
        self.shuffle_indices()
        for i in range(self.n_batches):
            yield self._create_batch() # return tensor[] and corresponding length
        if self.batch_index == self.n_batches:#stop iteration
            if self.last_size > 0:
                yield self._create_batch_last()
            raise StopIteration()
    def show_samples(self, n=2):
        for sample in self.sen[:n]:
            print(sample)
    def report(self):
        print('# samples: {}'.format(len(self.sen)))
        print('max len: {}'.format(self.max_length))
        print('text dict: {}'.format(self.textdict.word2idx.__len__()))
        print('user dict: {}'.format(self.userdict.word2idx.__len__()))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))


class TestDataLoader_class(object):

    def __init__(self, test, userdict,textdict, batch_size=32):## text should be list ["sen","sen2","sen3"]
        user = np.array(test["user"])
        time = np.array(test["time"])
        text = np.array(test["text"])
        self.userdict = userdict
        self.textdict = textdict
        self.batch_size = batch_size
        self.index = 0
        self.batch_index = 0
        def trytext(w):
            try:
                a = textdict.word2idx[w]
            except:
                a = 1
            return a

        def tryuser(w):
            try:
                a = userdict.word2idx[w]
            except:
                a = 0
            return a
        def tryuser4num(w):
            try:
                a = userdict.word2num[w]
            except:
                a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            return a
        def trytextnum(sen):
            tmp = [0.0,0.0,0.0,0.0,0.0]
            tmp[0] = math.log(len(sen))
            if 'http://t.cn/' in sen:
                tmp[1] = 1.0
            if sen[0] == '#':
                tmp[2] = 1.0
            if '[' in sen and ']' in sen:
                tmp[3] = 1.0
            if '?' in sen:
                tmp[4] = 1.0
            return tmp
        textlist = [[trytext(w) for w in sen.split(" ") if w in self.textdict.word2idx] for sen in text]
        userlist = [tryuser(w) for w in user]
        lenlist = [len(sen) for sen in textlist]
        textnum = [trytextnum(sen) for sen in text]
        user4numlist = [tryuser4num(w) for w in user]
        timelist = [time_convert2(w) for w in time]
        self.sen = textlist
        self.user = userlist
        self.time = timelist
        self.lenlist = lenlist
        self.user4numlist = user4numlist
        self.textnum = textnum
        self.n_batches = int(len(self.user) / self.batch_size)
        self.last_size = int(len(self.user) % self.batch_size)
        self.max_length = max(lenlist)
        self.report()


    def _create_batch(self):
        batch = []
        tbatch = []
        ubatch = []
        n = 0
        lenn_seq = []
        usernum = []
        textnum = []
        while n < self.batch_size:
            batch.append(self.sen[self.index])
            tbatch.append(self.time[self.index])
            ubatch.append(self.user[self.index])
            lenn_seq.append(self.lenlist[self.index])
            usernum.append(self.user4numlist[self.index])
            textnum.append(self.textnum[self.index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch  # like [[1,2,3],[1,2],[3,6,2,8]]
        # label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        tbatch = torch.FloatTensor(tbatch).cuda()
        usernum = torch.FloatTensor(usernum).cuda()
        textnum = torch.FloatTensor(textnum).cuda()
        user_tensor = ubatch[perm_idx]
        time_tensor = tbatch[perm_idx]
        usernum_tensor = usernum[perm_idx]
        textnum_tensor = textnum[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, perm_idx,usernum_tensor,textnum_tensor  # seq_lengths is the resorted one

    def _create_batch_last(self):
        batch = []
        tbatch = []
        ubatch = []
        n = 0
        lenn_seq = []
        usernum = []
        textnum = []
        while n < self.last_size:
            batch.append(self.sen[self.index])
            tbatch.append(self.time[self.index])
            ubatch.append(self.user[self.index])
            lenn_seq.append(self.lenlist[self.index])
            usernum.append(self.user4numlist[self.index])
            textnum.append(self.textnum[self.index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string = batch  # like [[1,2,3],[1,2],[3,6,2,8]]
        # label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(lenn_seq).cuda()

        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()

        # SORT YOUR TENSORS BY LENGTH!  seq_lengths is length of sen  perm_id is the original id
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        ubatch = torch.LongTensor(ubatch).cuda()
        tbatch = torch.FloatTensor(tbatch).cuda()
        usernum = torch.FloatTensor(usernum).cuda()
        textnum = torch.FloatTensor(textnum).cuda()
        user_tensor = ubatch[perm_idx]
        time_tensor = tbatch[perm_idx]
        usernum_tensor = usernum[perm_idx]
        textnum_tensor = textnum[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        return seq_tensor, seq_lengths, user_tensor, time_tensor, perm_idx,usernum_tensor,textnum_tensor  # seq_lengths is the resorted one



    def __len__(self):
        return self.n_batches

    def __iter__(self):
        '''
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:#stop iteration
                if self.last_size > 0:
                    yield self._create_batch_last()
                raise StopIteration()
            yield self._create_batch() # return tensor[] and corresponding length
        '''
        for i in range(self.n_batches):
            yield self._create_batch() # return tensor[] and corresponding length
        if self.batch_index == self.n_batches:#stop iteration
            if self.last_size > 0:
                yield self._create_batch_last()
            raise StopIteration()

    def show_samples(self, n=2):
        for sample in self.sen[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.sen)))
        print('max len: {}'.format(self.max_length))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))

