# Network Level Adversaries in Federated Learning

![Abstract art generated with VQGan + Clip](res/nlafl.png)

This repository contains code for the paper Network Level Adversaries in Federated Learning (`nlafl`) currently under submission.


## Get started

### Environment setup

#### Add your ssh key to agent
``` bash
ssh-agent bash
ssh-add
ssh 'remoteMachine' 
# you need to use ssh agents because project involves ssh to your account. 
# Password authentication is not supported.
```

#### Create a virtualenv
``` bash
virtualenv env
source env/bin/activate
```

#### Clone the repo
```bash
git lfs clone 'REPO'
cd 'REPO'
```

If there are issues with cloning with `git lfs` related to exceeding the lfs quota, the content of the `data/` folder can be downloaded from [this link.](https://drive.google.com/file/d/1E_rd1X8D2HNRcUYe7_wmoQBbAWDQiyXB/view?usp=share_link)

#### Check your python version>=3.8.10
``` bash
python --version
```

#### Install dependencies
```bash
pip install -e .
```

### Set common.py
Modify the variables in `common.py` to adapt to your setup.

```bash
In src/nlafl/commons.py set your repo dir and env dir. Both must end with /(slash). i.e : '~/env/'
```

## How to Use

### Command line tool
- Using emnist,dbpedia or fashionMnist commands you can run a single experiment. You need to provide additional position arguments for details look section below.
- Using docker command you can set the cluster it is required for the run command.
- If you already run the experiments you can create tables using the table command/
- You can run experiments using the run command.

``` shell
usage: nlafl [-h] {emnist,dbpedia,fashionMnist,docker,table,run} ...

positional arguments:
  {emnist,dbpedia,fashionMnist,docker,table,run}
                        Choose one of the following
    emnist              Run single emnist experiment
    dbpedia             Run single dbpedia experiment
    fashionMnist        Run single fashionMnist experiment
    docker              Start/Stop cluster to support many experiments
    table               Produce tables if you have already run the tables
    run                 Run experiments for a table

optional arguments:
  -h, --help            show this help message and exit

```

### Run single experiment

``` shell
- Using emnist,dbpedia or fashionMnist commands you can run a single experiment.
You need to provide positional arguments below to run the experiment.

usage: nlafl emnist/dbpedia/fashionMnist [-h] target_class num_pop_clients remove_pop_clients drop_epoch drop_count poison_count trial_ind {clip,mean} boost_factor upsample_epoch upsample_ct {random,agg,each} {agg,each}

positional arguments:
  target_class        which class is only present in subpopulation and it will be attacked ?
  num_pop_clients     how many clients have target class points?
  remove_pop_clients  how many clients are dropped at round 0, perfect knowledge
  drop_epoch          client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack
  drop_count          client id + dropping drop count k_N
  poison_count        number of poisoned clients
  trial_ind           trial number
  {clip,mean}         aggregate with fedavg or clipped fedavg?
  boost_factor        model poisoning boost factor
  upsample_epoch      defensive upsampling epoch T_S, set to -1 to deactivate
  upsample_ct         defensive upsampling client count k_S
  {random,agg,each}   how much knowledge does the network have: random dropping, mean of updates, or update
  {agg,each}          how much knowledge does server have: mean of updates or update

optional arguments:
  -h, --help          show this help message and exit
```

### Run docker
To run the experiments on mulitple machines, we used `celery` for orchestration. This requires `docker` to be installed on the system.
This step is required to use the `nlafl run` command.

Use `nlafl docker start` to start the `celery` instance and `stop` to kill celery and running docker images:
``` shell
usage: nlafl docker [-h] {start,stop}

positional arguments:
  {start,stop}  start stop docker
```

### Run experiment tables
To reproduce the tables presented in the paper:

- Run `nlafl docker start` first.
- There are 5 different tables and 3 different datasets.
  - Using the `identification` command you can run required experiments table 4 in paper.
  - Using the `baseline` command you can run required experiments clean target acc
  - Using the `targeted` command you can run required experiments table 2 and table 3.
  - Using the `big_plain` command you can run required experiments part of table 6 (plain)
  - Using the `big_enc` command you can run required experiments part of table 6 (enc)
  - Using the `big_mpc` command you can run required experiments a similar figure to 6 under mpc setting.

```shell
usage: nlafl run [-h] [--base BASE] [--version VERSION] {identification,baseline,targeted,big_plain,big_enc,big_mpc} {emnist,fashionMnist,dbpedia}

positional arguments:
  {identification,baseline,targeted,big_plain,big_enc,big_mpc}
                        which table will be run
  {emnist,fashionMnist,dbpedia}
                        which dataset will be used for running

optional arguments:
  -h, --help            show this help message and exit
  --base BASE           set different base, ie : "/path/to/base"
  --version VERSION     set different version, ie : "v2"
```
## Produce experiment tables
-  With table command if you complete running experiments you can produce tables with same commands you used for the running table.

``` shell
usage: nlafl table [-h] [--base BASE] [--version VERSION] {identification,baseline,targeted,big_plain,big_enc,big_mpc} {emnist,fashionMnist,dbpedia}

positional arguments:
  {identification,baseline,targeted,big_plain,big_enc,big_mpc}
  {emnist,fashionMnist,dbpedia}

optional arguments:
  -h, --help            show this help message and exit
  --base BASE
  --version VERSION
```

## Project Structure
```bash

.                                                                                                                       
├── data                                               #   where training data is stored-                                                                                                        
├── experimentLogs                                     #                                                                                                   
└── src                                                #                                          
    ├── main.py                                        #   command line tool                                                                                        
    ├── nlafl                                          #   implementation of the paper                                             
    │   ├── common.py                                  #   hard coded parameters are here.                                                     
    │   ├── dbpedia_models.py                          #   dbpedia model                                                             
    │   ├── dbpedia_sample.py                          #   dbpedia sample                                                             
    │   ├── emnist_models.py                           #   emnist model                                                            
    │   ├── emnist_sample.py                           #   emnist sample                                                            
    │   ├── fashionMnist_models.py                     #   fashionMnist model                                                                  
    │   ├── fashionMnist_sample.py                     #   fashionMnist sample                                                                  
    │   ├── main_dbpedia_upsample_multitarget.py       #   dbpedia Implementation                                                                                
    │   ├── main_emnist_upsample_multitarget.py        #   emnist  Implementation                     
    │   ├── main_fashionMnist_upsample_multitarget.py  #   fashionMnist Implementation                                     
    │   ├── make_dbpedia.py                            #   create dbpedia from raw            
    │   ├── make_emnist.py                             #   create emnist from raw           
    │   └── make_fashionMnist.py                       #   create fashionMnist from raw                         
    ├── runParallel                                    #   module for running it in parallel   
    │   ├── connection.py                              #   makes connections from multiple machines                
    │   ├── job.py                                     #   creates required jobs for the tables  
    │   ├── myrabbitmq.conf                            #   increases max run time in celery              
    │   ├── shutdown.py                                #   removes multiple celery instances       
    │   └── tasks.py                                   #   creates a celery task that runs command line instruction   
    └── tableCreation                                  #   module for creating tables     
        ├── createTables.py                            #   creates the tables
        ├── heatmap.py                                 #   heatmap raw data      
        ├── heatMapVisualization.py                    #   read heatmap raw data                   
        └── identification.py                          #   created identification results raw                    

```


## Experiments

**Note:** in order to run the `make_dbpedia.py` script necessary to create the data files, you will need to create a different environment with Tensorflow 2.7. This is necessary to create the embedding matrix successfully. With Tensorflow 2.2, the generated embedding matrix will contain only 0 entries.


Similarly to the other datasets, first generate the data files using `make_dbpedia.py`, then run a single experiment using `main_dbpedia_upsample_multitarget.py`.

