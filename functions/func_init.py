import numpy as np
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

def initconstants(dt,tau,N,N_models):

    constants = {}

    constants['tau_e']      = tf.constant(tau, dtype=tf.float32)
    constants['tau_i']      = tf.constant(tau/2, dtype=tf.float32)
    constants['tau_n']      = tf.constant(tau, dtype=tf.float32)
    constants['dt']         = tf.constant(dt, dtype=tf.float32)

    constants['eps1']       = tf.constant((1.0-dt/tau), dtype=tf.float32)
    constants['eps2']       = tf.constant(np.sqrt(2.0*dt/tau), dtype=tf.float32)

    constants['N_models']   = N_models
    constants['N_exc']      = int(0.8*N)
    constants['N_inh']      = int(0.2*N)
    constants['N']          = N

    Tinvdiag                = np.ones([N])/tau
    Tinvdiag[int(0.8*N):]   = 2/tau
    constants['Tinv']       = tf.constant(np.diag(Tinvdiag), dtype=tf.float32)

    mask                    = np.ones([N,N])
    mask[:,int(0.8*N):]     = -1
    constants['mask']       = tf.constant(mask, dtype=tf.float32)

    return constants

@tf.function
def compute_dueta(u,eta,W,bias,hinput,constants,activation):

    du  = constants['Tinv']@(-u + activation(W@u + bias + 0.1*eta + hinput))*constants['dt']
    eta = constants['eps1']*eta + constants['eps2']*tf.random.normal(tf.shape(u))

    return du,eta

@tf.function
def compute_uall(u,eta,W,bias,hinput,constants,skip,ratio,timesteps,activation):

    du,eta = compute_dueta(u,eta,W,bias,hinput,constants,activation)

    align = (skip<100)
    skipsteps = skip%100

    if align:
        uskip = tf.identity(u) + skipsteps*du
    else:
        uskip = tf.identity(u)
    
    uall  = tf.TensorArray(dtype=tf.float32, size=timesteps)

    for t in tf.range(timesteps):

        du,eta = compute_dueta(u,eta,W,bias,hinput,constants,activation)
        u = u + du

        if skipsteps>1 and t%skipsteps==(skipsteps-1):
            u = ratio*u + (1-ratio)*uskip
            if align:
                uskip = tf.identity(u) + skipsteps*du
            else:
                uskip = tf.identity(u)
        
        uall = uall.write(t,u)

    return u,eta,uall.stack()



def distance(taskname,uall1,dt1,uall2,dt2): #[time,trials,models,units,1]

    cost = 0.

    tasksA = ['go','anti','rtgo','rtanti','dm1','dm2','ctxdm1','ctxdm2','multsendm']
    tasksB = ['dlygo','dlyanti']
    tasksC = ['dlydm1','dlydm2','ctxdlydm1','ctxdlydm2','multsendlydm','dms','dnms','dmc','dnmc']

    inter1 = int(500/dt1)
    inter2 = int(500/dt2)
    sbsmp = int(dt1/dt2)

    timeA = [0,1,2,3]
    timeB = [0,1,2,4,5]
    timeC = [0,1,2,4,5,6]

    if taskname in tasksA:
        times = timeA
    elif taskname in tasksB:
        times = timeB
    elif taskname in tasksC:
        times = timeC

    if dt1 == 150 or dt1 == 200:
        uall1 = uall1[1:]

    for e in range(np.size(times)-1):

        timestart = times[e]
        timeend = times[e+1]
        timetaken = timeend - timestart

        u1 = np.zeros([timetaken*inter1,200,80,50,1])
        u2 = np.zeros([timetaken*inter1,200,80,50,1])

        u1[:] = uall1[timestart*inter1:timeend*inter1]
        u2[:] = uall2[timestart*inter2:timeend*inter2:sbsmp][-timetaken*inter1:]
        
        cost += np.mean((u1-u2)**2)/(np.size(times)-1) 

    return cost