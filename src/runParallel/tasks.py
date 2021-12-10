from celery import Celery
import os
app = Celery('runParallel',broker='amqp://guest:guest@0.0.0.0:5672',include=['runParallel.tasks'])

app.conf.update(
    BROKER_TRANSPORT_OPTIONS = {"manager_port": 15672}
)


@app.task
def add(x,y):
    return x+y

@app.task
def runCommand(command):
    return os.system(command)