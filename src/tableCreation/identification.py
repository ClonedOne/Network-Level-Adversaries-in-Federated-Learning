from collections import defaultdict
import os
import numpy as np
import numpy as np
import pickle
import pprint



def readIdentificationInfo(paths,ct,mode="c2",):
    NUM_ROUNDS = 120
    NUM_USERS = 100
    rounds = [5,10,15,20,30,40,50,70]
    allTrials  = []

    for path in paths:
        trial = []
        try:
            data = np.load(path,allow_pickle=True)[()]
        except:
            # since we need are looking for clean training (dropping no clients at all), it does not matter to chose knowledge level random or each both same thing.
            print("look at equivalent at random")
            try:
                path = path.replace("each","random",1)
                data = np.load(path,allow_pickle=True)[()]
            except:
                print("failed")
                continue

        for r in rounds:
            
            pop_clients = list(range(ct))

            if mode == "c1":
                knowledge_dict = data["per_round_each_improvements_attacker"][r]
            elif mode == "c2":
                knowledge_dict = data["per_round_agg_improvements_attacker"][r]

            all_changes = [knowledge_dict.get(ind, np.inf) for ind in range(NUM_USERS)]
            all_means = [np.mean(change) for change in all_changes]
            worst_means = np.argsort(all_means)[:ct]

            # Compute overlap between dropped clients and actual target clients
            overlap = len([i for i in worst_means if i in pop_clients])
            trial.append(overlap)
        # print(trial)
        allTrials.append(trial)


    return np.array(allTrials).mean(axis=0),allTrials


def identification(base,version,dataset,mode=None):
    dir  = os.path.expanduser(os.path.join(base , dataset , version))
    print(dir)
    cmd = dir + "/results_upsample_multitarget_0_{numPopClient}_0_0_-1_0_{trialInd}_mean_10.0_-1_0_each_each.npy"
    filesnames = {}

    if dataset == 'emnist':
        numPopClients = [9,12,15]
    elif dataset == 'dbpedia':
        numPopClients = [15]
    elif dataset == 'fashionMnist':
        numPopClients = [15]
    for numPopClient in numPopClients:
        filesnames[numPopClient] =  [cmd.format(numPopClient=numPopClient,trialInd= i ) for i in [0,1,2,4]]
    
    results = defaultdict(int)
    if mode == 'c1': 
        for k,v in filesnames.items():

            result,allTrial  = readIdentificationInfo(v,k,mode="c1")
            # print(k,result)
            # print(allTrial)
            results[k] = result
            return results

        
    elif mode == 'c2':
        for k,v in filesnames.items():
    
            result,allTrial  = readIdentificationInfo(v,k,mode="c2")
            # print(k,result)
            # print(allTrial)
            results[k] = result
            return results
    else:
        print(dataset)
        print('Plain')
        print('-'* 50)
        for k,v in filesnames.items():

            result,allTrial  = readIdentificationInfo(v,k,mode="c1")
            results[k] = result
        pprint.pprint(results)
        print('Enc')
        print('-'* 50)
        for k,v in filesnames.items():
    
            result,allTrial  = readIdentificationInfo(v,k,mode="c2")
            results[k] = result
        pprint.pprint ( results)
