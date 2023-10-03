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