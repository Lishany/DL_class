'''
import pandas as pd
basepath = '/home/syugroup/LishanYu/hw2/'
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
 
f = open(basepath + 'phase2/resultmlpmodel_file0plus.txt', "w+")
i = 0
for i in range(len(testid)):
    f.write(test["user"].iloc[i]+"\t"+test["weibo"].iloc[i]+"\t"+"0,0,0\n")
f.close()
'''
'''
import pandas as pd
basepath = '/home/syugroup/LishanYu/hw2/'
with open(basepath + 'phase2/resultmlpmodelall_order_pluscnn.txt','w+') as wf:
	ff = open(basepath + 'phase2/resultmlpmodel_file0plus.txt','r')
	ff1 = open(basepath + 'phase2/resultmlpmodel_file1pluscnn.txt','r')
	ff2 = open(basepath + 'phase2/resultmlpmodel_file2pluscnn.txt','r')
	ff3 = open(basepath + 'phase2/resultmlpmodel_file3pluscnn.txt','r')#
	with open('/home/syugroup/LishanYu/hw2/phase2/phase2_class.txt','r') as df:
		for line in df:
			kv = float(line)
			if kv == 0:
				wf.write(ff.readline())
			elif kv == 1:
				wf.write(ff1.readline())
			elif kv == 2:
				wf.write(ff2.readline())
			elif kv > 2:
				wf.write(ff3.readline())
	ff.close()
	ff1.close()
	ff2.close()
	ff3.close()

'''

import pandas as pd
basepath = '/home/syugroup/LishanYu/hw2/'
with open(basepath + 'phase2/resultmlpmodelall_order_pluscnn.txt','w+') as wf:
	ff = open(basepath + 'phase2/resultmlpmodel_file0plus.txt','r')
	ff1 = open(basepath + 'phase2/resultmlpmodel_file1pluscnn.txt','r')
	ff2 = open(basepath + 'phase2/resultmlpmodel_file2pluscnn.txt','r')
	ff3 = open(basepath + 'phase2/resultmlpmodel_file3pluscnn.txt','r')#
	ff4 = open(basepath + 'phase2/resultmlpmodel_file4pluscnn.txt','r')
	ff5 = open(basepath + 'phase2/resultmlpmodel_file5pluscnn.txt','r')
	ff6 = open(basepath + 'phase2/resultmlpmodel_file6pluscnn.txt','r')
	ff7 = open(basepath + 'phase2/resultmlpmodel_file7pluscnn.txt','r')
	with open('/home/syugroup/LishanYu/hw2/phase2/phase2_class.txt','r') as df:
		for line in df:
			kv = float(line)
			if kv == 0:
				wf.write(ff.readline())
			elif kv == 1:
				wf.write(ff1.readline())
			elif kv == 2:
				wf.write(ff2.readline())
			elif kv == 3:
				wf.write(ff3.readline())
			elif kv == 4:
				wf.write(ff4.readline())
			elif kv == 5:
				wf.write(ff5.readline())
			elif kv == 6:
				wf.write(ff6.readline())
			elif kv == 7:
				wf.write(ff7.readline())
	ff.close()
	ff1.close()
	ff2.close()
	ff3.close()
	ff4.close()
	ff5.close()
	ff6.close()
	ff7.close()

'''
with open('/home/syugroup/LishanYu/hw2/phase2/phase2_class.txt','r') as df:
    for line in df:
        kv = float(line)
        if kv == 0:
            testid.append(count)
        count += 1
'''

'''
datapath2 = basepath + 'test_jeiba.json'
test = pd.read_json(datapath2,typ = 'frame')#print(len(test['user'])
print(len(test['user']))
userpath = basepath+"user_ana.txt"

class Dictionary(object):
    def __init__(self,path):
        self.word2num = {}
        with open(path, 'r')as df:
            for line in df:
                tmp = line.split(" ")
                #self.word2num[tmp[0]] = [math.log(tmp[1]),float(tmp[2]),float(tmp[3]),float(tmp[4])]
                self.word2num[tmp[0]] = [tmp[5],tmp[6],tmp[7]]
    def __len__(self):
        return len(self.word2num)
user = Dictionary(userpath)

with open(basepath + 'phase2/maxresult_plus.txt','w+') as wf:
	count = 0
	for line in test['user']:
		try:
			temp = user.word2num[line]
		except:
			temp = ['0','0','0']
		wf.write(line+'\t'+test['weibo'].iloc[count]+'\t'+temp[0]+','+temp[1]+','+temp[2]+'\n')
		count = count + 1


basepath = '/home/syugroup/LishanYu/hw2/'
import math
with open(basepath + 'phase2/class_label.txt','w+') as wf:
	ff1 = open(basepath + 'phase1/prediction_mlp_class.txt','r')
	ff2 = open(basepath + 'phase1/prediction_mlp_score.txt','r')
	ff3 = open(basepath + 'phase1/prediction_simple_class.txt','r')
	#ff4 = open(basepath + 'phase1/prediction_simple_score.txt','r')#
	for line in ff1:
		kv1 = int(line[1])
		kv2 = round(float(ff2.readline()))
		kv3 = int(ff3.readline()[1])
		#kv4 = math.round(float(ff4.readline()))
		wf.write(str(max(kv1,kv2,kv3))+'\n')
	ff1.close()
	ff2.close()
	ff3.close()
	#ff4.close()
'''