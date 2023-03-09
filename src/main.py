import os
from runParallel.job import Scheduler
from tableCreation.createTables import Table
from tableCreation.identification import identification
from runParallel.connection import connect
from runParallel.shutdown import shutdown
from runParallel.tasks import runCommand
from nlafl import common

import time
def main():
    if common.REPO_DIR == '' or common.ENV_DIR == '':
        print('Please REPO_DIR AND ENV_DIR in src/nlafl/common.py')
        return
    if common.REPO_DIR[-1] != '/' or common.ENV_DIR[-1] != '/':
        print('REPO_DIR AND ENV_DIR must end with /(Slash) in the end')
        return
    parser = setup_argparse()
    args = parser.parse_args()
    if args.command == 'emnist':
        os.system(f'cd {common.network["repoDir"]} && python src/nlafl/main_emnist_upsample_multitarget.py  {args.target_class} {args.num_pop_clients} {args.remove_pop_clients} {args.drop_epoch} {args.drop_count} {args.poison_count} {args.trial_ind} {args.agg_fn} {args.boost_factor} {args.upsample_epoch} {args.upsample_ct} {args.network_knowledge} {args.server_knowledge}')
    elif args.command == 'dbpedia':
        os.system(f'cd {common.network["repoDir"]} &&  python src/nlafl/main_dbpedia_upsample_multitarget.py  {args.target_class} {args.num_pop_clients} {args.remove_pop_clients} {args.drop_epoch} {args.drop_count} {args.poison_count} {args.trial_ind} {args.agg_fn} {args.boost_factor} {args.upsample_epoch} {args.upsample_ct} {args.network_knowledge} {args.server_knowledge}')
    elif args.command == 'fashionMnist':
        os.system(f'cd {common.network["repoDir"]} &&  python src/nlafl/main_fashionMnistMnist_upsample_multitarget.py  {args.target_class} {args.num_pop_clients} {args.remove_pop_clients} {args.drop_epoch} {args.drop_count} {args.poison_count} {args.trial_ind} {args.agg_fn} {args.boost_factor} {args.upsample_epoch} {args.upsample_ct} {args.network_knowledge} {args.server_knowledge}')
    elif args.command == 'docker':
        if args.action == 'start':
            runDocker()
        if args.action == 'stop':
            stopDocker()
    elif args.command == 'table':
        if args.table_type == 'identification':
            identification(args.base,args.version,args.dataset)
        elif args.table_type == 'big_plain':
            table = Table(args.dataset,args.version,args.base)
            table.query('c1','target',)
        elif args.table_type == 'big_enc':
            table = Table(args.dataset,args.version,args.base)
            table.query('c2','target',)
        elif args.table_type == 'big_mpc':
            table = Table(args.dataset,args.version,args.base)
            table.query('c3','target',)
        elif args.table_type == 'baseline':
            table = Table(args.dataset,args.version,args.base)
            table.query('baseline','target',)
        elif args.table_type == 'targeted':
            table = Table(args.dataset,args.version,args.base)
            table.query('identificationDrop','target',)
        else:
            raise NotImplementedError
    elif args.command == 'run':
        scheduler = Scheduler(args.dataset,args.version)
        experiments = scheduler.getSpecificJobs(args.table_type)
        for i in experiments:
            print(i)
            runCommand.delay(i)
            time.sleep(0.1)  # give some time

def runDocker():
    os.system(f'docker run -d  -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 -v {common.network["workDir"]}/runParallel/myrabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq:3.9-management')
    os.system('docker run  -d -it --rm --name flower   -p 5555:5555 mher/flower:0.9.5 --broker="amqp://guest:guest@`hostname`:5672" --broker_api="http://guest:guest@`hostname`:15672/api/vhost" --address=0.0.0.0')
    connect()

    
def stopDocker():
    shutdown()
    os.system('docker stop flower rabbitmq')


def setup_argparse():
    import argparse
    def addArguments(parser):
        parser.add_argument('target_class', type=int,
                        help='which class is only present in subpopulation and it will be attacked ?')
        parser.add_argument('num_pop_clients', type=int,
                            help='how many clients have target class points?')
        parser.add_argument('remove_pop_clients', type=int,
                            help='how many clients are dropped at round 0, perfect knowledge')
        parser.add_argument("drop_epoch", type=int,
                            help='client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack')
        parser.add_argument("drop_count", type=int,
                            help='client id + dropping drop count k_N')
        parser.add_argument("poison_count", type=int,
                            help='number of poisoned clients')
        parser.add_argument("trial_ind", type=int, help='trial number')
        parser.add_argument('agg_fn', choices=[
                            'clip', 'mean'], help='aggregate with fedavg or clipped fedavg?')
        parser.add_argument('boost_factor', type=float,
                            help='model poisoning boost factor')
        parser.add_argument('upsample_epoch', type=int,
                            help='defensive upsampling epoch T_S, set to -1 to deactivate')
        parser.add_argument('upsample_ct', type=int,
                            help='defensive upsampling client count k_S')
        parser.add_argument('network_knowledge', choices=[
                            'random', 'agg', 'each'], help='how much knowledge does the network have: random dropping, mean of updates, or update')
        parser.add_argument('server_knowledge', choices=['agg', 'each'], help='how much knowledge does server have: mean of updates or update')

 
    parser = argparse.ArgumentParser('nlafl')
    subparsers = parser.add_subparsers(help='Choose one of the following',dest='command')

    parser_emnist = subparsers.add_parser('emnist', help='Run single emnist experiment')
    addArguments(parser_emnist)
    parser_dbpedia = subparsers.add_parser('dbpedia', help='Run single dbpedia experiment')
    addArguments(parser_dbpedia)
    parser_fashion = subparsers.add_parser('fashionMnist', help='Run single fashionMnist experiment')
    addArguments(parser_fashion)
    parser_docker= subparsers.add_parser('docker', help='Start/Stop cluster to support many experiments')
    parser_docker.add_argument('action', choices=['start', 'stop'],help='start stop docker')

    parser_table= subparsers.add_parser('table', help='Produce tables if you already run the tables')
    parser_table.add_argument('table_type', choices=['identification', 'baseline','targeted','big_plain','big_enc','big_mpc'])
    parser_table.add_argument('dataset',choices=['emnist', 'fashionMnist','dbpedia'])
    parser_table.add_argument('--base',type=str,default=common.npy_SaveDir['base'])
    parser_table.add_argument('--version',type=str,default=common.version['base'])

    parser_run= subparsers.add_parser('run', help='Run experiments for a table')
    parser_run.add_argument('table_type', choices=['identification', 'baseline','targeted','big_plain','big_enc','big_mpc','visibility'], help= 'which table will be run')
    parser_run.add_argument('dataset',choices=['emnist', 'fashionMnist','dbpedia'], help= 'which dataset will be used for running')
    parser_run.add_argument('--base',type=str,default=common.npy_SaveDir['base'],help ='set different base, ie : "/path/to/base"' )
    parser_run.add_argument('--version',type=str,default=common.version['base'], help ='set different version, ie : "v2"')








 
    return parser

