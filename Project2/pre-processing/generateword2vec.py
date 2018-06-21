import tensorflow as tf
from gensim.models import Word2Vec
import pandas as pd

basepath = '/home/syugroup/LishanYu/hw2/'
datapath = basepath + 'train_jeiba.json'
datapath2 = basepath + 'test_jeiba.json'

train = pd.read_json(datapath,typ='frame')
test = pd.read_json(datapath2,typ='frame')
text=pd.concat([train['text'],test['text']],0,ignore_index=True)
output_file_name = '/home/syugroup/LishanYu/hw2/word2vec'
'''
class generate_w2v():
    def __init__(self,usecomma = True):
        self.comma = ["，","。","：","“","”","、","；","\"","\n"]
        self.usecomma = usecomma
    def generate(self,text,embedding_size=140,niter=1):
        all_word = []
        if not self.usecomma:
            for line in text:
                temp = []
                for w in line.split(' '):
                    if w in self.comma:
                        temp.append(" ")
                    else:
                        temp.append(w)
                all_word.append(temp)
        else:
            for line in text:
                temp = []
                for w in line.split(' '):
                    temp.append(w)
                all_word.append(temp)
        model = Word2Vec(all_word,size = embedding_size,min_count=10,sg = 0,window = 5,hs = 1,iter = niter)
        #print(model.compute_loss)
        return(model)
w2vmodel = generate_w2v()
result = w2vmodel.generate(text)
wvv = result.wv
wvv.save(output_file_name)
wvv.most_similar(u"网红", topn=20)
'''

def generate_wv(text, embedding_size=50, niter=5):
    all_word = []
    for line in text:
        temp = []
        for w in line.split(' '):
            temp.append(w)
        all_word.append(temp)
    model = Word2Vec(all_word, size=embedding_size, min_count=8, sg=0, window=5, hs=1, iter=niter)
    # print(model.compute_loss)
    return (model)
result = generate_wv(text,niter = 8)
result.save(output_file_name)
#resultcopy = Word2Vec.load(output_file_name)
#resultcopy.wv[","]
#resultcopy.wv.index2word[0]

'''
result.most_similar(u"爸爸",topn=20)
result.save_word2vec_format(output_file_name)
resultcopy = KeyedVectors.load_word2vec_format(output_file_name)
resultcopu = Word2Vec.load(output_file_name)
model.save(fname)
model = Word2Vec.load(fname)
'''