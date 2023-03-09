import os
import sys
from itertools import product
from pprint import pprint
import time
from nlafl.common import npy_SaveDir,npy_LogDir,execfile,version

class Scheduler(object):
    

    def __init__(self,name,version):
        self.trials = [0,1,2,4]
        self.classes = list(range(10))
        self.projectDir = npy_SaveDir[name]
        self.resultsDir = os.path.expanduser(os.path.join(self.projectDir,version))
        self.projectLog = npy_LogDir[name]
        self.logsDir    = os.path.expanduser(os.path.join(self.projectLog,version))


        if name =="emnist":
            self.clientSizes = [9,12,15]
            self.ratioMultiplier = 2

           # self.clientSizes = list(range(9,16,3))
        elif name =="fashionMnist":
            self.clientSizes = [15]
            self.ratioMultiplier = 1

        elif name =="dbpedia":
            self.clientSizes = [15]
            self.ratioMultiplier = 1

        else:
            raise NotImplementedError
        self.execfile  =  execfile[name]
        self.name = name
        self.version = version


        
        self.doneJobs = self._getDoneTasks()
        self.runningJobs = self._getRunningJobs()
        self.allJobs = self._getAllJobs()
        self.waitingJobs = self._getWaitingJobs()

    def _getDoneTasks(self):
        tasks = [i for i in os.listdir(self.resultsDir) if ".npy" in i]
        return tasks

    def _getAllJobs(self):
        if self.name == "emnist":

            tasks = []
            # tasks += self._do_perfect_knowledge_randomDropping_table2()[0] # 1
            # tasks += self._do_each_attack_table9()[0] #1 
            # tasks +=self._do_mean_attack_table10()[0] # 0
            # tasks +=self._do_c3_attack_table11()[0] # 0
            # tasks +=self._do_baseline()[0] # 1
            # tasks +=self._do_idenfication_drop_comparison()[0] # 0
            # tasks += self._do_aggfunction_comparison()[0] #12
            tasks += self._do_heatmaps()[0] # 
            # tasks += self._do_dummy_experiment()[0]
        elif self.name == "fashionMnist":
            tasks = []

            tasks += self._do_perfect_knowledge_randomDropping_table2()[0] # 0
            tasks +=self._do_mean_attack_table10()[0] # 0
            tasks +=self._do_baseline()[0] # 0
            tasks +=self._do_c3_attack_table11()[0] # 0
            tasks += self._do_each_attack_table9()[0] #  0

            # tasks +=self._do_idenfication_drop_comparison()[0] # 0
            # tasks += self._do_heatmaps()[0] # 0


        elif self.name == "dbpedia":

            tasks = []

            # tasks += self._do_perfect_knowledge_randomDropping_table2()[0] # 0
            tasks += self._do_each_attack_table9()[0] #  0
            tasks +=self._do_mean_attack_table10()[0] # 0
            tasks +=self._do_c3_attack_table11()[0] # 0
            tasks +=self._do_baseline()[0] # 0
            # tasks +=self._do_idenfication_drop_comparison()[0] # 0
            # tasks += self._do_heatmaps()[0] 




        return tasks

    def _getRunningJobs(self):
        #TODO:
        tasks = []
        return tasks
    def getSpecificJobs(self,case):
        tasks = []

        if case == 'identification':
            tasks+= self._do_clean()[0]
        elif case =='big_plain' :
             tasks += self._do_each_attack_table9()[0]
        elif case =='big_enc' :
            tasks +=self._do_mean_attack_table10()[0] 
        elif case =='big_mpc' :
            tasks +=self._do_c3_attack_table11()[0] 
        elif case =='baseline' :
            tasks +=self._do_baseline()[0] 
        elif case =='targeted' :
            tasks +=self._do_idenfication_drop_comparison()[0]
        elif case == 'visibility':
            tasks+=self._do_visibility()[0]
        else:
            raise NotImplementedError
        
        result = []
        for i in tasks :
            result.append(self._turnRunCommandFormat(i))
        return result
    def _getWaitingJobs(self):
        setDoneJobs =  set(self.doneJobs)
        # print("set done jobs",setDoneJobs)
        result = []
        for i in self.allJobs :
            if( self._turnNpyFormat(i) not in setDoneJobs) and (self._turnRunCommandFormat(i) not in result):
                #print(self._turnNpyFormat(i))
                result.append(self._turnRunCommandFormat(i))

        #tasks=[i for i in self.allJobs if self._turnNpyFormat(i) not in setDoneJobs]
        

        return result
        #return list(map(self._turnRunCommandFormat,tasks))


    def _do_baseline(self,):
        experimentClasses = [0,1,9]
        trials = [0,1,2,4]
        perfectKnowledge= []

        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            perfectKnowledge.append(currentJob)

        randomDropping = []
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
           
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "random" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            randomDropping.append(currentJob)


        # tasks = perfectKnowledge  + randomDropping 
        tasks = perfectKnowledge 
        dict_tasks = {
            "perfectKnowledge":perfectKnowledge,
            # "randomDropping":randomDropping,
        }
        return tasks,dict_tasks

    def _do_idenfication_drop_comparison(self):
        experimentClasses = [0]
        trials = [0,1,2,4]


        # C1
        #__________________________________________________________________________________________________________________________________________________
        c1 = []
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            for ratio in (range(1,4)):
                currentJob = {}

                #varies between experiements
                currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
                currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
                
                currentJob["trial_ind"]         = trial #trial number
                currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

                currentJob["drop_epoch"]        =  15 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
                currentJob["drop_count"]        =  ratio*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

                # same for all iterations 
                currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
                currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
                currentJob["poison_count"]      =  0 #number of poisoned clients
                currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
                currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
                currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
                currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
                c1.append(currentJob)
        # Random Dropping

        # C2
        #__________________________________________________________________________________________________________________________________________________
        c2 = []
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            for ratio in (range(1,4)):
                currentJob = {}

                #varies between experiements
                currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
                currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
                
                currentJob["trial_ind"]         = trial #trial number
                currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

                currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
                currentJob["drop_count"]        =  ratio*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

                # same for all iterations 
                currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
                currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
                currentJob["poison_count"]      =  0 #number of poisoned clients
                currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
                currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
                currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
                currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
                c2.append(currentJob)

        tasks = c1 + c2 
        dict_tasks = {
            "c1":c1,
            "c2":c2,
        }
        return tasks,dict_tasks
    def _do_clean(self):
        experimentClasses = [0]
        trials = [0,1,2,4]


        clean = []
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            clean.append(currentJob)
        
        tasks = clean
        dict_tasks = {
            "clean":clean,
        }
        return tasks,dict_tasks

    def _do_visibility(self):
        experimentClasses = [0]
        trials = [0,1,2,4]
        visible_fracs = [0,0.2,0.4,0.8,1]
        visible_pop_fracs = [0.2, 0.4 ,  0.8]

        visibility = []
        for targetClass,targetClientSize,trial,visible_frac, visible_pop_frac in product(experimentClasses,self.clientSizes,trials,visible_fracs,visible_pop_fracs):
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            currentJob["visible_frac"]  =  visible_frac #how much knowledge does server have: mean of updates or update
            currentJob["visible_pop_frac"]  =  visible_pop_frac #how much knowledge does server have: mean of updates or update

            visibility.append(currentJob)
        
        tasks = visibility
        dict_tasks = {
            "visibility":visibility,
        }
        return tasks,dict_tasks

    def _do_perfect_knowledge_randomDropping_table2(self):
        experimentClasses = [0]
        # tests it for client sizes 10,12,15 as usual
        # repeat each experiment for 5 times as usual

        trials = [0,1,2,4]

        # Perfect Knowledge
        #__________________________________________________________________________________________________________________________________________________

        perfectKnowledge= []
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            for ratio in (range(4)):
                currentJob = {}

                #varies between experiements
                currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
                currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
                
                currentJob["trial_ind"]         = trial #trial number
                currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

                currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
                currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

                # same for all iterations 
                currentJob["remove_pop_clients"]=  ratio*(targetClientSize//3) #how many clients are dropped at round 0, used in PK or RD
                currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
                currentJob["poison_count"]      =  0 #number of poisoned clients
                currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
                currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
                currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
                currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
                perfectKnowledge.append(currentJob)
        # Random Dropping
        #__________________________________________________________________________________________________________________________________________________

        randomDropping = []
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            for ratio in (range(4)):
                currentJob = {}

                #varies between experiements
                currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
                currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
                
                currentJob["trial_ind"]         = trial #trial number
                currentJob["network_knowledge"] =  "random" #how much knowledge does the network have: random dropping, mean of updates, or update

                currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
                currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

                # same for all iterations 
                currentJob["remove_pop_clients"]=  ratio*(targetClientSize//3) #how many clients are dropped at round 0, used in PK or RD
                currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
                currentJob["poison_count"]      =  0 #number of poisoned clients
                currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
                currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
                currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
                currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
                randomDropping.append(currentJob)


        tasks = perfectKnowledge + randomDropping 
        dict_tasks = {
            "perfectKnowledge":perfectKnowledge,
            "randomDropping":randomDropping,
        }
        return tasks,dict_tasks

    # def _do_Identification_table3_4(self):
    #     experimentClasses = [0]
    #     # tests it for client sizes 10,12,15 as usual
    #     # repeat each experiment for 5 times as usual

    #     trials = list(range(5))
    #     # Identification Each
    #     #__________________________________________________________________________________________________________________________________________________

    #     each = []
    #     for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
    #         currentJob = {}

    #         #varies between experiements
    #         currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
    #         currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
    #         currentJob["trial_ind"]         = trial #trial number
    #         currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

    #         currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
    #         currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

    #         # same for all iterations 
    #         currentJob["remove_pop_clients"]=  0  #how many clients are dropped at round 0, used in PK or RD
    #         currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
    #         currentJob["poison_count"]      =  0 #number of poisoned clients
    #         currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
    #         currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
    #         currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
    #         currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
    #         each.append(currentJob)


    #     # Identification Mean
    #     #__________________________________________________________________________________________________________________________________________________

    #     mean = []
    #     for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
    #         currentJob = {}

    #         #varies between experiements
    #         currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
    #         currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
    #         currentJob["trial_ind"]         = trial #trial number
    #         currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

    #         currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
    #         currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

    #         # same for all iterations 
    #         currentJob["remove_pop_clients"]=  0  #how many clients are dropped at round 0, used in PK or RD
    #         currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
    #         currentJob["poison_count"]      =  0 #number of poisoned clients
    #         currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
    #         currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
    #         currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
    #         currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
    #         mean.append(currentJob)
            
    #     tasks = each + mean 
    #     dict_tasks = {
    #         "each":each,
    #         "mean":mean,
    #     }
    #     return tasks,dict_tasks

    def _do_aggfunction_comparison(self):
        experimentClasses = [0]
        trials = [0,1,2,4]
        poisonClientRatios = list(range(3))
        boostFactors = [1.0,100.0]

        #mean
        #__________________________________________________________________________________________________________________________________________________
        mean = []
        for targetClass,targetClientSize,trial,boostFactor,poisonRatio in product(experimentClasses,self.clientSizes,trials,boostFactors,poisonClientRatios):

            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  poisonRatio*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  boostFactor #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            mean.append(currentJob)
        

        #clip 
        #__________________________________________________________________________________________________________________________________________________
        clip = []

        for targetClass,targetClientSize,trial,boostFactor,poisonRatio in product(experimentClasses,self.clientSizes,trials,boostFactors,poisonClientRatios):

            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  0 #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  poisonRatio*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  boostFactor #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            clip.append(currentJob)


        tasks = mean + clip
        dict_tasks = {
            "mean":mean,
            "clip":clip,
        }
        return tasks,dict_tasks

    def _do_heatmaps(self):
        experimentClasses = [0]
        trials = [0,1,2,4,42]
        poisonCounts = [0,3,7]
        dropCounts = [0,3,7]

        clientSizes =[15]
        #pk 
        #__________________________________________________________________________________________________________________________________________________
        pk = []

        for targetClass,targetClientSize,trial,poisonCount,dropCount in product(experimentClasses,clientSizes,trials,poisonCounts,dropCounts):
            if poisonCount == 0 and dropCount ==0:
                continue

            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =   0 #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  dropCount #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  poisonCount #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            pk.append(currentJob)

        #pk_clip 
        #__________________________________________________________________________________________________________________________________________________
        pk_clip = []
        for targetClass,targetClientSize,trial,poisonCount,dropCount in product(experimentClasses,clientSizes,trials,poisonCounts,dropCounts):

            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  -1 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =   0 #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  dropCount #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  poisonCount #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            pk_clip.append(currentJob)

     
        #c1 
        #__________________________________________________________________________________________________________________________________________________
        c1 = []

        for targetClass,targetClientSize,trial,poisonCount,dropCount in product(experimentClasses,clientSizes,trials,poisonCounts,dropCounts):

            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  15 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =   dropCount #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  poisonCount #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            c1.append(currentJob)
        #c1_clip
        #__________________________________________________________________________________________________________________________________________________
        c1_clip = []

        for targetClass,targetClientSize,trial,poisonCount,dropCount in product(experimentClasses,clientSizes,trials,poisonCounts,dropCounts):

            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  15 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =   dropCount #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  poisonCount #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            c1_clip.append(currentJob)

        #c2
        #__________________________________________________________________________________________________________________________________________________
        c2 = []

        for targetClass,targetClientSize,trial,poisonCount,dropCount in product(experimentClasses,clientSizes,trials,poisonCounts,dropCounts):

            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =   dropCount #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  poisonCount #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            c2.append(currentJob)
        #c1_clip
        #__________________________________________________________________________________________________________________________________________________
        c2_clip = []

        for targetClass,targetClientSize,trial,poisonCount,dropCount in product(experimentClasses,clientSizes,trials,poisonCounts,dropCounts):

            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]   = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =   dropCount #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]=  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  poisonCount #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            c2_clip.append(currentJob)
        # tasks = c1 + c2 + pk + upsample
        tasks = c1 + c1_clip + c2 + c2_clip#+ upsample_clip
        dict_tasks = {
            "c1":c1,
            "c1_clip":c1_clip,
            "c2":c2,
            "c2_clip":c2_clip,
            #"c2":c2,
            # "pk":pk,
            # "pk_clip":pk_clip,
         #   'upsample_clip':upsample_clip
        }
        return tasks,dict_tasks


        




    def _do_each_attack_table9(self):
        #only considers classes 0,1,9 for decreasing computation cost
        experimentClasses = [0,1,9]
        # tests it for client sizes 10,12,15 as usual
        # repeat each experiment for 5 times as usual
 

        trials = [0,1,2,4]


       
        # Drop
        drop = []
        #__________________________________________________________________________________________________________________________________________________
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  15 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop.append(currentJob)

        #__________________________________________________________________________________________________________________________________________________
        
        
        drop_upsample = []
        #Drop + Upsample
        #__________________________________________________________________________________________________________________________________________________
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  15 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  15 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  2*(targetClientSize//3) #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_upsample.append(currentJob)

        #__________________________________________________________________________________________________________________________________________________
        drop_poison_fedavg = []
        # Drop + Poison + FedAvg
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  15 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_poison_fedavg.append(currentJob)
        #__________________________________________________________________________________________________________________________________________________

        #__________________________________________________________________________________________________________________________________________________
        drop_poison_clip = []
        # Drop + Poison + Clip
        #__________________________________________________________________________________________________________________________________________________
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  15 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_poison_clip.append(currentJob)

        #__________________________________________________________________________________________________________________________________________________

        drop_poison_fedavg_upsample = []
        # Drop + Poison + FedAvg + Upsample
        #__________________________________________________________________________________________________________________________________________________

        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  15 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  15 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  2*(targetClientSize//3) #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_poison_fedavg_upsample.append(currentJob)
        #__________________________________________________________________________________________________________________________________________________

        drop_poison_clip_upsample = []
        # Drop + Poison + Clip + Upsample
        #__________________________________________________________________________________________________________________________________________________

        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "each" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  15 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  15 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  2*(targetClientSize//3) #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_poison_clip_upsample.append(currentJob)        
        #__________________________________________________________________________________________________________________________________________________




        tasks = drop + drop_upsample + drop_poison_fedavg + drop_poison_clip + drop_poison_fedavg_upsample + drop_poison_clip_upsample
        # tasks = drop + drop_upsample + drop_poison_fedavg  + drop_poison_fedavg_upsample 

        dict_tasks = {
            "drop":drop,
            "drop_upsample":drop_upsample,
            "drop_poison_fedavg":drop_poison_fedavg,
            "drop_poison_clip":drop_poison_clip,
            "drop_poison_fedavg_upsample":drop_poison_fedavg_upsample,
            "drop_poison_clip_upsample":drop_poison_clip_upsample
        }
        return tasks,dict_tasks

    def _do_mean_attack_table10(self):
        #only considers classes 0,1,9 for decreasing computation cost
        experimentClasses = [0,1,9]
        # experimentClasses = [9]
        # tests it for client sizes 10,12,15 as usual
        # repeat each experiment for 5 times as usual

        trials = [0,1,2,4]


       
        # Drop
        drop = []
        #__________________________________________________________________________________________________________________________________________________
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop.append(currentJob)

        #__________________________________________________________________________________________________________________________________________________
        
        
        drop_upsample = []
        #Drop + Upsample
        #__________________________________________________________________________________________________________________________________________________
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update


            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  15 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  2*(targetClientSize//3) #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_upsample.append(currentJob)

        #__________________________________________________________________________________________________________________________________________________
        drop_poison_fedavg = []
        # Drop + Poison + FedAvg
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update


            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_poison_fedavg.append(currentJob)
        #__________________________________________________________________________________________________________________________________________________

        #__________________________________________________________________________________________________________________________________________________
        drop_poison_clip = []
        # Drop + Poison + Clip
        #__________________________________________________________________________________________________________________________________________________
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_poison_clip.append(currentJob)

        #__________________________________________________________________________________________________________________________________________________

        drop_poison_fedavg_upsample = []
        # Drop + Poison + FedAvg + Upsample
        #__________________________________________________________________________________________________________________________________________________

        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update


            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  15 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  2*(targetClientSize//3) #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_poison_fedavg_upsample.append(currentJob)
        #__________________________________________________________________________________________________________________________________________________

        drop_poison_clip_upsample = []
        # Drop + Poison + Clip + Upsample
        #__________________________________________________________________________________________________________________________________________________

        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  15 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  2*(targetClientSize//3) #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "each" #how much knowledge does server have: mean of updates or update
            drop_poison_clip_upsample.append(currentJob)        
        #__________________________________________________________________________________________________________________________________________________


        # tasks = drop + drop_upsample + drop_poison_fedavg  + drop_poison_fedavg_upsample 

        tasks = drop + drop_upsample + drop_poison_fedavg + drop_poison_clip + drop_poison_fedavg_upsample + drop_poison_clip_upsample
        # tasks =  drop_poison_clip_upsample
        dict_tasks = {
            "drop":drop,
            "drop_upsample":drop_upsample,
            "drop_poison_fedavg":drop_poison_fedavg,
            "drop_poison_clip":drop_poison_clip,
            "drop_poison_fedavg_upsample":drop_poison_fedavg_upsample,
            "drop_poison_clip_upsample":drop_poison_clip_upsample
        }
        return tasks,dict_tasks
        
    def _do_c3_attack_table11(self):
        #only considers classes 0,1,9 for decreasing computation cost
        experimentClasses = [0,1,9]
        # tests it for client sizes 10,12,15 as usual
        # repeat each experiment for 5 times as usual

        trials = [0,1,2,4]


       
        # Drop
        drop = []
        #__________________________________________________________________________________________________________________________________________________
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "agg" #how much knowledge does server have: mean of updates or update
            drop.append(currentJob)

        #__________________________________________________________________________________________________________________________________________________
        
        
        drop_upsample = []
        #Drop + Upsample
        #__________________________________________________________________________________________________________________________________________________
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update


            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  0 #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  30 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  2*(targetClientSize//3) #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "agg" #how much knowledge does server have: mean of updates or update
            drop_upsample.append(currentJob)

        #__________________________________________________________________________________________________________________________________________________
        drop_poison_fedavg = []
        # Drop + Poison + FedAvg
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update


            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "agg" #how much knowledge does server have: mean of updates or update
            drop_poison_fedavg.append(currentJob)
        #__________________________________________________________________________________________________________________________________________________

        #__________________________________________________________________________________________________________________________________________________
        drop_poison_clip = []
        # Drop + Poison + Clip
        #__________________________________________________________________________________________________________________________________________________
        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  -1 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  0 #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "agg" #how much knowledge does server have: mean of updates or update
            drop_poison_clip.append(currentJob)

        #__________________________________________________________________________________________________________________________________________________

        drop_poison_fedavg_upsample = []
        # Drop + Poison + FedAvg + Upsample
        #__________________________________________________________________________________________________________________________________________________

        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update


            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "mean" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  30 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  2*(targetClientSize//3) #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "agg" #how much knowledge does server have: mean of updates or update
            drop_poison_fedavg_upsample.append(currentJob)
        #__________________________________________________________________________________________________________________________________________________

        drop_poison_clip_upsample = []
        # Drop + Poison + Clip + Upsample
        #__________________________________________________________________________________________________________________________________________________

        for targetClass,targetClientSize,trial in product(experimentClasses,self.clientSizes,trials):
            
            currentJob = {}

            #varies between experiements
            currentJob["target_class"]      = targetClass # targetClass attacked , class 0 ,1 ...10
            currentJob["num_pop_clients"]         = targetClientSize # how many client have target class points
            
            currentJob["trial_ind"]         = trial #trial number
            currentJob["network_knowledge"] =  "agg" #how much knowledge does the network have: random dropping, mean of updates, or update

            currentJob["drop_epoch"]        =  30 #client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
            currentJob["drop_count"]        =  self.ratioMultiplier*(targetClientSize//3) #client id + dropping drop count k_N fixed to 10 for less computation

            # same for all iterations 
            currentJob["remove_pop_clients"]      =  0 #how many clients are dropped at round 0, used in PK or RD
            currentJob["agg_fn"]            =  "clip" #aggregate with fedavg or clipped fedavg?
            currentJob["poison_count"]      =  self.ratioMultiplier*(targetClientSize//3) #number of poisoned clients
            currentJob["boost_factor"]      =  10.0 #model poisoning boost factor when poison count is 0, meaningless
            currentJob["upsample_epoch"]    =  30 #defensive upsampling epoch T_S, set to -1 to deactivate
            currentJob["upsample_ct"]       =  2*(targetClientSize//3) #defensive upsampling client count k_S
            currentJob["server_knowledge"]  =  "agg" #how much knowledge does server have: mean of updates or update
            drop_poison_clip_upsample.append(currentJob)        
        #__________________________________________________________________________________________________________________________________________________


        # tasks = drop + drop_upsample + drop_poison_fedavg  + drop_poison_fedavg_upsample 

        tasks = drop + drop_upsample + drop_poison_fedavg + drop_poison_clip + drop_poison_fedavg_upsample + drop_poison_clip_upsample
        dict_tasks = {
            "drop":drop,
            "drop_upsample":drop_upsample,
            "drop_poison_fedavg":drop_poison_fedavg,
            "drop_poison_clip":drop_poison_clip,
            "drop_poison_fedavg_upsample":drop_poison_fedavg_upsample,
            "drop_poison_clip_upsample":drop_poison_clip_upsample
        }
        return tasks,dict_tasks



    def _turnNpyFormat(self,exp_dict):
        commonPart =f"results_upsample_multitarget_"+\
            f"{exp_dict['target_class']}_"+\
            f"{exp_dict['num_pop_clients']}_"+\
            f"{exp_dict['remove_pop_clients']}_"+\
            f"{exp_dict['drop_count']}_"+\
            f"{exp_dict['drop_epoch']}_"+\
            f"{exp_dict['poison_count']}_"+\
            f"{exp_dict['trial_ind']}_"+\
            f"{exp_dict['agg_fn']}_"+\
            f"{exp_dict['boost_factor']}_"+\
            f"{exp_dict['upsample_epoch']}_"+\
            f"{exp_dict['upsample_ct']}_"+\
            f"{exp_dict['network_knowledge']}_"+\
            f"{exp_dict['server_knowledge']}"

        if exp_dict.get('visible_frac') != None and exp_dict.get('visible_pop_frac') != None:
            commonPart = f"{commonPart}_{exp_dict['visible_frac']}_{exp_dict['visible_pop_frac']}"  
        
        return f"{commonPart}.npy"

    def _turnRunCommandFormat(self,exp_dict):
        firstPart = self.execfile +\
        f"{exp_dict['target_class']} "+\
        f"{exp_dict['num_pop_clients']} "+\
        f"{exp_dict['remove_pop_clients']} "+\
        f"{exp_dict['drop_epoch']} "+\
        f"{exp_dict['drop_count']} "+\
        f"{exp_dict['poison_count']} "+\
        f"{exp_dict['trial_ind']} "+\
        f"{exp_dict['agg_fn']} "+\
        f"{exp_dict['boost_factor']} "+\
        f"{exp_dict['upsample_epoch']} "+\
        f"{exp_dict['upsample_ct']} "+\
        f"{exp_dict['network_knowledge']} "+\
        f"{exp_dict['server_knowledge']} "

        if exp_dict.get('visible_frac') != None and exp_dict.get('visible_pop_frac') != None:
            firstPart = f"{firstPart} {exp_dict['visible_frac']} {exp_dict['visible_pop_frac']}"


        secondPart = f" > " + self.logsDir + "/output_"+\
        f"{exp_dict['target_class']}_"+\
        f"{exp_dict['num_pop_clients']}_"+\
        f"{exp_dict['remove_pop_clients']}_"+\
        f"{exp_dict['drop_epoch']}_"+\
        f"{exp_dict['drop_count']}_"+\
        f"{exp_dict['poison_count']}_"+\
        f"{exp_dict['trial_ind']}_"+\
        f"{exp_dict['agg_fn']}_"+\
        f"{exp_dict['boost_factor']}_"+\
        f"{exp_dict['upsample_epoch']}_"+\
        f"{exp_dict['upsample_ct']}_"+\
        f"{exp_dict['network_knowledge']}_"+\
        f"{exp_dict['server_knowledge']}"

        if exp_dict.get('visible_frac') != None and exp_dict.get('visible_pop_frac') != None:
            secondPart = f"{secondPart}_{exp_dict['visible_frac']}_{exp_dict['visible_pop_frac']}"

      

        return  f"{firstPart}{secondPart}.txt 2>&1"


        

    


    def getJobs(self):
        return self.waitingJobs



if __name__ == '__main__':
    from runParallel.tasks import runCommand
    
    scheduler = Scheduler("emnist","v8")

    # data = scheduler._doRandomDrop()
    # pData = list(map(scheduler._turnNpyFormat, data))
    # for i in pData:
    #     print(i)
    print(f"Done  {len(scheduler.doneJobs)} jobs...")
    print(f"Waiting {len(scheduler.waitingJobs)} jobs...")
    print("Scheduling ...")
    # for i in scheduler.waitingJobs:
    #     # print(i)
    #     # break
    #     runCommand.delay(i)
    #     time.sleep(0.1)  # give some time
    # print("Done!")



   
