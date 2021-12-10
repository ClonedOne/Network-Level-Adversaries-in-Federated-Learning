import os
import numpy as np
from itertools import product
from collections import defaultdict
import pandas as pd
import json
from nlafl import common
class HeatMapValue:
    IsSet = False

    def set_dir_version(dir,version):
        HeatMapValue.dir = dir
        HeatMapValue.version = version
        HeatMapValue.IsSet = True


    def __init__(self,num_pop_client,remove_pop_client,poison_count,trial_ind,file_path,upsample_count=0):
        if not HeatMapValue.IsSet:
            raise ValueError("First set directory and version")

        self.num_pop_client = num_pop_client
        self.remove_pop_client  = remove_pop_client
        self.poison_count = poison_count
        self.trial_ind = trial_ind
        self.file_path = file_path
        self.upsample_count = upsample_count
    

        self.readValues()

    def readValues(self,):

        fullPath = os.path.expanduser(os.path.join(HeatMapValue.dir,HeatMapValue.version,self.file_path))

        try:

            data = np.load(fullPath,allow_pickle=True)[()]
            self.targetAcc_50 = np.mean([data["pop_accs"][49+i][1] for i in range(5)])
            self.targetAcc_100 = np.mean([data["pop_accs"][99+i][1] for i in range(5)])
            try:

                self.targetAcc_200 = np.mean([data["pop_accs"][199+i][1] for i in range(5)])
            except:
                self.targetAcc_200 = np.nan

 
        except FileNotFoundError :
            # print(fullPath)
            # print('notFound')
            self.targetAcc_50=np.nan
            self.targetAcc_100=np.nan
            self.targetAcc_200=np.nan






    def averageByTrialIndex(list_heatvalue):
        acc50 = defaultdict(list)
        acc100 = defaultdict(list)
        acc200 = defaultdict(list)

        for value in list_heatvalue:
            acc50[(value.num_pop_client,value.remove_pop_client,value.poison_count,value.upsample_count)].append(value.targetAcc_50)
            acc100[(value.num_pop_client,value.remove_pop_client,value.poison_count,value.upsample_count)].append(value.targetAcc_100)
            acc200[(value.num_pop_client,value.remove_pop_client,value.poison_count,value.upsample_count)].append(value.targetAcc_200)
        
        # acc100= {k:np.mean(v) for k,v in acc100}
        # acc200= {k:np.mean(v) for k,v in acc200}

        return acc50,acc100,acc200
    






def getDf(dir,version,mode,agg):
           

    dir  = os.path.join(dir,version)
    HeatMapValue.set_dir_version(dir,version)
    poisonRatios = [0,3,7]
    removePopClientRatios = [0,3,7]
    trialInds = [0,1,2,4,42]

    numPopClients = [15]
    if mode == 'c1':
        cmd = os.path.join(dir , "results_upsample_multitarget_0_{numPopClient}_0_{remove_pop_client}_15_{poison_count}_{trialInd}_{agg}_10.0_-1_0_each_each.npy")
    elif mode == 'c2': 
        cmd = os.path.join(dir , "results_upsample_multitarget_0_{numPopClient}_0_{remove_pop_client}_30_{poison_count}_{trialInd}_{agg}_10.0_-1_0_agg_each.npy" )
    elif mode == 'pk':
        cmd = os.path.join(dir ,  "results_upsample_multitarget_0_{numPopClient}_{remove_pop_client}_0_-1_{poison_count}_{trialInd}_{agg}_10.0_-1_0_each_each.npy")
    filesnames = list()
    for numPopClient in numPopClients:
        for poisonRatio, removePopClientRatio,trial_ind  in product(poisonRatios,removePopClientRatios,trialInds):
            poisonCount = poisonRatio
            removePopClientCount = removePopClientRatio
            filePath = cmd.format(numPopClient=numPopClient,trialInd= trial_ind,remove_pop_client =removePopClientCount, poison_count= poisonCount,agg=agg )
            filesnames.append(HeatMapValue(numPopClient,removePopClientCount,poisonCount,trial_ind,filePath))
    
    acc50,acc100,acc200 = HeatMapValue.averageByTrialIndex(filesnames)
    df = defaultdict(list)

    for k,v in acc100.items():
        num_pop_client,remove_pop_client,poison_count,upsample_count = k

        acc100_ = removeNanAndAverage(v)
        acc50_  = removeNanAndAverage(acc50[k]) 
        acc200_  = removeNanAndAverageSilent(acc200[k]) 


        # df["num_pop_client"].append(num_pop_client)
        df["remove_pop_client"].append(remove_pop_client)
        df["poison_count"].append(poison_count)
        df["acc50"].append(acc50_ )
        df["acc100"].append(acc100_)
        df["acc200"].append(acc200_ )

    df = pd.DataFrame(df)
    return df
        

def removeNanAndAverage(arr):

    # print(len(arr))
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    # print(len(arr)) 
    
    arr = arr[:4]
    # print(len(arr)) 
    # print('--')
    if len(arr) != 4:
        print('Fails')
    return round(np.mean(arr),2)

def removeNanAndAverageSilent(arr):

    # print(len(arr))
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    # print(len(arr)) 
    
    arr = arr[:4]
    # print(len(arr)) 
    # print('--')
    # if len(arr) != 4:
    #     print('Fails')
    return round(np.mean(arr),2)





if __name__ == "__main__":



    base = common.npy_SaveDir['base']
    emnist_dir = os.path.join(base,'emnist')
    emnist_version = common.version['emnist']




    emnist_pk = getDf(emnist_dir,emnist_version,'pk','mean')
    emnist_pk_clip = getDf(emnist_dir,emnist_version,'pk','clip')
    emnist_c1 = getDf(emnist_dir,emnist_version,'c1','mean')
    emnist_c1_clip = getDf(emnist_dir,emnist_version,'c1','clip')
    emnist_c2 = getDf(emnist_dir,emnist_version,'c2','mean')
    emnist_c2_clip = getDf(emnist_dir,emnist_version,'c2','clip')

    # emnist_pk.iloc[0] = {'remove_pop_client':0.0,'poison_count':0,'acc50':0.65,'acc100':0.80}
    print('pk')
    print(emnist_pk)
    print("-"*100)
    print('pk clip')
    print(emnist_pk_clip)

    print('c1')
    print(emnist_c1)
    print("-"*100)
    print('c1_clip')
    print(emnist_c1_clip)

    print('c2')
    print(emnist_c2)
    print("-"*100)
    print('c2 clip')
    print(emnist_c2_clip)


    exportDict  = {

        'emnist_pk':emnist_pk.to_json(),
        'emnist_pk_clip':emnist_pk_clip.to_json(),
        'emnist_c1':emnist_c1.to_json(),
        'emnist_c1_clip':emnist_c1_clip.to_json(),
        'emnist_c2':emnist_c2.to_json(),
        'emnist_c2_clip':emnist_c2_clip.to_json(),

    }

    with open("heatMap_data.json","w") as f:
        json.dump(exportDict,f)




