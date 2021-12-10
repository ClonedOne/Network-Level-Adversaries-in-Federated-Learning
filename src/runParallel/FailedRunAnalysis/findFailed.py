import os
from pprint import pprint
import parse
from nlafl.common import npy_SaveDir,npy_LogDir

def readFiledAndSearch(path):

    with open(path,"r") as f:
        content = f.read()
    #print(content)
    return "nan" in content

def log2npy(log):
    


    formatString = "output_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt"
    parsed = parse.parse(formatString, log)

    target_class        =parsed[0]
    num_pop_clients     =parsed[1]
    remove_pop_clients  =parsed[2]
    drop_count          =parsed[4]
    drop_epoch          =parsed[3]
    poison_count        =parsed[5]
    trial_ind           =parsed[6]
    agg_fn              =parsed[7]
    boost_factor        =parsed[8]
    upsample_epoch      =parsed[9]
    upsample_ct         =parsed[10]
    network_knowledge   =parsed[11]
    server_knowledge    =parsed[12]

    result = f"results_upsample_multitarget_{target_class}_{num_pop_clients}_{remove_pop_clients}_{drop_count}_{drop_epoch}_{poison_count}_{trial_ind}_{agg_fn}_{boost_factor}_{upsample_epoch}_{upsample_ct}_{network_knowledge}_{server_knowledge}.npy"
    return result
    
def main():
    logPathBase = npy_LogDir['emnist']
    resultPathBase = npy_SaveDir['emnist']

    version = "v1"

    logPath = os.path.expanduser(os.path.join(logPathBase,version))
    resultPath = os.path.expanduser(os.path.join(resultPathBase,version))


    dirs = os.listdir( logPath )
    # print(dirs)
    filteredDir = [i for i in dirs if ".txt" in i]

    failedTxt = [i for i in filteredDir if readFiledAndSearch(os.path.expanduser(os.path.join(logPath,i)))]
    failedTxtWithDir = [ os.path.expanduser(os.path.join(logPath,i)) for i in failedTxt]

    failedNpy = [ log2npy(i)  for i in failedTxt]
    
    failedNpyWithDir = [ os.path.expanduser(os.path.join(resultPath,i)) for i in failedNpy]

    # FileNotFoundError 
    print(len(failedTxtWithDir))
    print(len(failedNpyWithDir))
    pprint(failedNpy)
    pprint(failedTxtWithDir)
    # for i in failedNpyWithDir:
    #     os.remove(i)
    for i in failedTxtWithDir:
        os.remove(i)


    
    # os.remove(i)
    
    

if __name__ == "__main__":
    main()
