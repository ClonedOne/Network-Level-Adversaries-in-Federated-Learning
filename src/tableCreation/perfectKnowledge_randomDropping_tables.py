import os
import numpy as np
import numpy as np
import os
NUM_ROUNDS = 120
NUM_USERS = 100
def readIdentificationInfo(path):
    
    data = np.load(os.expanduser(path),allow_pickle=True)[()]
    

    
    identified = []
    for round in range(NUM_ROUNDS):
        per_round_agg_improvements = data["per_round_agg_improvements"][round]
        targetUsers = data["per_round_target_users"]
        newDict = {}
        for i in range(NUM_USERS):
            if i in per_round_agg_improvements:
                newDict[i] = np.mean(per_round_agg_improvements[i])
            else:
                newDict[i] =np.inf
        
        identified.append([k for k, v in sorted(newDict.items(), key=lambda item: item[1])])  
        print(identified[round][:12],targetUsers[round])
        print("-"*100)

        
        