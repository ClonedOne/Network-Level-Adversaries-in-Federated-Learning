from fabric import Connection
from fabric import ThreadingGroup as Group
from os import getenv
from nlafl.common import network
from fabric.exceptions import GroupException
def shutdown():
    try:
        group = Group(*network['group'])
        result = group.run("kill $(pgrep -f celery)")


    except:
        print('Succesfully killed celery')