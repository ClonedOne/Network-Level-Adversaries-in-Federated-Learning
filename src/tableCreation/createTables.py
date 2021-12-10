import pandas as pd
import sys

sys.path.append('../')
pd.set_option('display.max_rows', 500)
pd.options.display.float_format = '{:,.2f}'.format
from  collections import defaultdict
import os
import numpy as np
import pandas as pd
from pprint import pprint
import parse
from nlafl import common
from runParallel.job import Scheduler
from pprint import pprint




class Table:


    def __init__(self,dataset,version,base= None):

        if base is None:
            base = os.path.expanduser(common.npy_SaveDir['base'])
        self.version = version
        self.resultDir = os.path.expanduser(os.path.join(base,dataset,version))
        self.dataset = dataset
        self.getNpyFiles()
        self.getEntries()
        self.getDf()

    
    def getNpyFiles(self,):
        self.npyFiles = [r for r in os.listdir(self.resultDir) if r.endswith('.npy')]
        return self.npyFiles


    def getEntries(self,):
        self.entries = [Entry(self.resultDir,npyFile,self.dataset) for npyFile in self.npyFiles]
        return self.entries

    def getDf(self,):

        df_dict = defaultdict(list)

        for entry in self.entries:
            df_dict['target_class'].append( entry.target_class )
            df_dict['num_pop_clients'].append( entry.num_pop_clients )
            df_dict['remove_pop_clients'].append( entry.remove_pop_clients )
            df_dict['drop_count'].append( entry.drop_count )
            df_dict['drop_epoch'].append( entry.drop_epoch )
            df_dict['poison_count'].append( entry.poison_count )
            df_dict['trial_ind'].append( entry.trial_ind )
            df_dict['agg_fn'].append( entry.agg_fn )
            df_dict['boost_factor'].append( entry.boost_factor )
            df_dict['upsample_epoch'].append( entry.upsample_epoch )
            df_dict['upsample_ct'].append( entry.upsample_ct )
            df_dict['network_knowledge'].append( entry.network_knowledge )
            df_dict['server_knowledge'].append( entry.server_knowledge )
            df_dict['model_left_acc'].append( entry.model_left_acc )
            df_dict['model_right_acc'].append( entry.model_right_acc )
            df_dict['target_left_acc'].append( entry.target_left_acc )
            df_dict['target_right_acc'].append( entry.target_right_acc )
        self.df = pd.DataFrame(df_dict)
        return self.df
    def tableProducer(self,df,taskDict,mode,removeCount=False,dropCount=False,trialInd=False):   
        pivots = []
        #print(taskDict.keys())


        index = ["num_pop_clients","target_class",]
        if removeCount:
            index.append('remove_pop_clients')
        if dropCount:
            index.append('drop_count')
        if trialInd:
            index.append('trial_ind')
        for key,value in taskDict.items():
            
            valueDf =  pd.concat([self.filterDf(df,i) for i in value])
            if mode == "target":
                pivot =pd.pivot_table(valueDf,values=['target_left_acc','target_right_acc'],index=index,aggfunc=np.mean )
            elif mode == "model":
                pivot =pd.pivot_table(valueDf,values=['model_left_acc','model_right_acc'],index=index,aggfunc=np.mean ) 
            # print(key)
            # print(pivot)
            # print('-'*50)
            pivot[key.upper().replace("_","")] = pivot.apply(lambda x: ('% .2f' % x[0]) + "/"+ ('% .2f' % x[1])  ,axis=1)
            pivot = pivot[[key.upper().replace("_","")]]
            pivots.append(pivot)
        return pd.concat(pivots,axis=1)

    def writeTableToLatex(self,pivot,filename,caption):
        with open(filename,"w") as f:
            f.write(pivot.to_latex(caption=caption, position="h!",multirow=True))

    def filterDf(self,df,filter_v):
        return df.loc[(df[list(filter_v)] == pd.Series(filter_v)).all(axis=1)]

    def query(self,knowledgeType,accType,trialInd=False):

        scheduler = Scheduler(self.dataset,self.version)
        removeCount=False
        dropCount=False
        if knowledgeType == "c1" :
            _,taskDict = scheduler._do_each_attack_table9()
            title = f"EACH (C1) - {accType.upper()}"
            filename = f"Table_9_{accType.upper()}.txt"

        elif knowledgeType == "c2" :
            _,taskDict = scheduler._do_mean_attack_table10()
            title = f"MEAN (C2) - {accType.upper()}"
            filename = f"Table_10_{accType.upper()}.txt"

        elif knowledgeType == "c3" :
            _,taskDict = scheduler._do_c3_attack_table11()
            title = f"C3 - {accType.upper}"
            filename = f"Table_11_{accType.upper()}.txt"

        elif knowledgeType == "pk":
            _,taskDict = scheduler._do_perfect_knowledge_randomDropping_table2()
            title = f"PK and RD"
            filename = f"Table2.txt"
            removeCount = True

        
        elif knowledgeType == "identificationDrop":
            _,taskDict = scheduler._do_idenfication_drop_comparison()
            title = f"PK and RD"
            filename = f"Table7.txt"
            dropCount = True

        elif knowledgeType == "baseline":
            _,taskDict = scheduler._do_baseline()
            title = f"PK and RD"
            filename = f"Baseline.txt"
            removeCount = True
        

        else:
            raise NotImplementedError
        
        table = self.tableProducer(self.df,taskDict,accType,removeCount,dropCount,trialInd)
        self.writeTableToLatex(table,filename,title)
        print(table)
        return table
        
class Entry:
    EMNIST_LEFT = 50
    EMNIST_RIGHT = 100
    FASHION_LEFT = 150
    FASHION_RIGHT = 300
    DBPEDIA_LEFT = 150
    DBPEDIA_RIGHT = 300

    def __init__(self,resultDir,npyFile,dataset):
        self.path = os.path.expanduser(os.path.join(resultDir,npyFile))
        data  = np.load(os.path.expanduser(self.path), allow_pickle=True)[()]
        formatString = "results_upsample_multitarget_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npy"
        parsed = parse.parse(formatString, npyFile)
        self.target_class        =int(parsed[0])
        self.num_pop_clients     =int(parsed[1])
        self.remove_pop_clients  =int(parsed[2])
        self.drop_count          =int(parsed[3])
        self.drop_epoch          =int(parsed[4])
        self.poison_count        =int(parsed[5])
        self.trial_ind           =int(parsed[6])
        self.agg_fn              =parsed[7]
        self.boost_factor        =float(parsed[8])
        self.upsample_epoch      =int(parsed[9])
        self.upsample_ct         =int(parsed[10])
        self.network_knowledge   =parsed[11]
        self.server_knowledge    =parsed[12]
        
        if dataset == 'emnist':
            left = Entry.EMNIST_LEFT
            right = Entry.EMNIST_RIGHT
        elif dataset == 'fashionMnist':
            left = Entry.FASHION_LEFT
            right = Entry.FASHION_RIGHT
        elif dataset == 'dbpedia':
            left = Entry.DBPEDIA_LEFT
            right = Entry.DBPEDIA_RIGHT
        else:
            raise NotImplementedError
        
        self.model_left_acc = self.getAcc(data,'model',left)
        self.model_right_acc = self.getAcc(data,'model',right)
        self.target_left_acc = self.getAcc(data,'target',left)
        self.target_right_acc = self.getAcc(data,'target',right)

    def __str__(self,):


        return  str({
            "target_class" :self.target_class,
            "num_pop_clients" :self.num_pop_clients,
            "remove_pop_clients" :self.remove_pop_clients,
            "drop_count" :self.drop_count,
            "drop_epoch" :self.drop_epoch,
            "poison_count" :self.poison_count,
            "trial_ind" :self.trial_ind,
            "agg_fn" :self.agg_fn,
            "boost_factor" :self.boost_factor,
            "upsample_epoch" :self.upsample_epoch,
            "upsample_ct" :self.upsample_ct,
            "network_knowledge" :self.network_knowledge,
            "server_knowledge" :self.server_knowledge,
            
        })

    def getAcc(self,data,mode,round):
        if mode == 'model':
            return np.mean(np.array(data["accs"][round-1:round+4]    ),axis=0)[1]
        elif mode == 'target':
            return np.mean(np.array(data["pop_accs"][round-1:round+4]),axis=0)[1]
        else:
            raise NotImplementedError
