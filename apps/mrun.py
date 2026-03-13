#!/usr/bin/env python
"""Utilities for parallel execution of shell tasks via multiprocessing.

This module provides helper functions for running batches of shell
commands in parallel, each in its own working directory.  Two strategies
are offered:

* :func:`mrun` — spawns one ``multiprocessing.Process`` per task.
* :func:`mpool` — uses a ``multiprocessing.Pool`` with a fixed number of
  workers.

When executed as a script, it reads a ``tasks`` file where each line has
the format ``<directory> <command>`` and dispatches all tasks using
:func:`mpool`.

Example::

    python mrun.py 4          # run tasks from ./tasks using 4 workers
"""

import os,sys
import numpy as np
from multiprocessing import Process, Lock, Pool, Manager



# sys.path.append(os.environ['LIBDISTDIR']+"/pytools")
# import uqmrun as mr

def f(fcnpars,output):
    """Execute a shell task in a subdirectory, appending the result to a shared list.

    Args:
        fcnpars (list): A 2-element list ``[task, id]`` where ``task`` is the shell command string and ``id`` is the integer task identifier.
        output (list): Shared list to which ``[id, return_code]`` is appended.
    """
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
    """Run tasks in parallel using separate processes.

    Each task is executed in its own subprocess via ``multiprocessing.Process``.

    Args:
        list_tasks (list[str]): List of shell command strings.
        npieces (int): Number of tasks to run.

    Returns:
        np.ndarray: 1d array of return codes sorted by task index.
    """

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
    """Execute a shell task in a named directory.

    Args:
        args (list): A 3-element list ``[task, dirname, id]`` where ``task`` is the shell command,
            ``dirname`` is the working directory, and ``id`` is the task identifier.

    Returns:
        int: The return code of the shell command.
    """
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
    """Run tasks in parallel using a process pool.

    Args:
        list_tasks (list[str]): List of shell command strings.
        dirnames (list[str]): List of directory names, one per task.
        npieces (int): Number of tasks to run.
        nproc (int): Number of worker processes in the pool.

    Returns:
        np.ndarray: 1d array of return codes.
    """

    p = Pool(nproc)
    arglist=[[list_tasks[id],dirnames[id], id] for id in range(npieces)]
    res=p.map(g,arglist)

    npout=np.hstack(res)

    return npout
    #p.apply_async(worker)
#p.join()


##########################################################################################
def myfcn(xx,pars):
    """Example function that saves input and returns row sums.

    Args:
        xx (np.ndarray): 2d input array.
        pars: Unused auxiliary parameters.

    Returns:
        np.ndarray: 1d array of row sums.
    """

    np.savetxt('out',xx)
    sum=np.sum(xx,axis=1)

    #if (xx.shape[0]==5):
    #time.sleep(3)

    return sum

def myfcn2(xx,script):
    """Example function that runs a script, saves input, and returns row sums.

    Args:
        xx (np.ndarray): 2d input array.
        script (str): Path to a shell script to execute.

    Returns:
        np.ndarray: 1d array of row sums.
    """
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



