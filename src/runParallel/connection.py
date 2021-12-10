from fabric import Connection
from fabric import SerialGroup as Group
from nlafl.common import network
from os import getenv




#command = "hostname"
def task(c):
    
    command = "celery --app runParallel.tasks worker --detach --loglevel INFO -O fair --prefetch-multiplier 1 --concurrency 1 --logfile celery_`hostname`.txt"
    with c.cd(network['workDir']):
        with c.prefix(network['environmentPath']):         
           result= c.run(command)
           print(result)
            



def connect():
    group = Group(*network['group'])
    for connection in group:
        task(connection)




