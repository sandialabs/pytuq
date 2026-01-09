#!/usr/bin/env python
import os,sys
import numpy as np
from multiprocessing import Process, Lock, Pool, Manager



# sys.path.append(os.environ['LIBDISTDIR']+"/pytools")
# import uqmrun as mr

def f(fcnpars,output):
    #l.acquire()
    task=fcnpars[0]
    id=fcnpars[1]
    dir = 'run_' + str(id)
    if not os.path.exists(dir):
        os.mkdir(dir, 0o755)
    prev = os.path.abspath(os.getcwd())
    os.chdir(dir)

    output.append([id, os.system(task)])
    os.chdir(prev)
#l.release()

def mrun(list_tasks,npieces):

    lock = Lock()


    output = Manager().list()
    jobs=[]
    for i in range(npieces):
        p=Process(target=f, args=([list_tasks[i],i],output))
        jobs.append(p)
        p.start()
        #p.join()

        #for proc in jobs:
    #proc.start()

    for proc in jobs:
        proc.join()

    #return xx_chunks,output

    print(output)

    soutput=sorted(output)
    npout=np.hstack([a[1] for a in soutput])
    return npout

##########################################################################################

def g(args):
    task, dirname, id=args
    #dir = 'run_' + str(id)
    #dir = 'task_' + str(id+1)
    if not os.path.exists(dirname):
        os.mkdir(dirname, 0o755)
    prev = os.path.abspath(os.getcwd())
    os.chdir(dirname)
    output=os.system(task)
    os.chdir(prev)

    return output

def mpool(list_tasks, dirnames, npieces, nproc):


    p = Pool(nproc)
    arglist=[[list_tasks[id],dirnames[id], id] for id in range(npieces)]
    res=p.map(g,arglist)

    npout=np.hstack(res)

    return npout
    #p.apply_async(worker)
#p.join()


##########################################################################################
def myfcn(xx,pars):

    np.savetxt('out',xx)
    sum=np.sum(xx,axis=1)

    #if (xx.shape[0]==5):
    #time.sleep(3)

    return sum

def myfcn2(xx,script):
    os.system(script+' '+str())
    np.savetxt('out',xx)
    sum=np.sum(xx,axis=1)

    #if (xx.shape[0]==5):
    #time.sleep(3)

    return sum

if __name__ == '__main__':
    # xx=np.random.rand(13,5)
    # #print(xx)
    # np.savetxt('par.ref',xx)
    # print(mr.mrun(xx,[myfcn,[],3]))
    # #print(mr.mpool(xx,[myfcn2,[],3]))
    dirs_list_tasks=open('tasks').read().splitlines()
    dirnames = []
    list_tasks = []
    for s in dirs_list_tasks:
        a, b = s.split(" ", 1)
        dirnames.append(a)
        list_tasks.append(b)
    print(dirnames, list_tasks)
    #mrun(list_tasks, 96) #commenting out since this overworks the computer
    nproc = int(sys.argv[1])
    ntasks = len(list_tasks)
    mpool(list_tasks, dirnames, ntasks, nproc)



