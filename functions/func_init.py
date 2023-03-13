import numpy as np
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

def initconstants(dt,tau,N,task):

    constants = {}

    constants['tau_e']      = tf.constant(tau, dtype=tf.float32)
    constants['tau_i']      = tf.constant(tau/2, dtype=tf.float32)
    constants['tau_n']      = tf.constant(tau, dtype=tf.float32)
    constants['dt']         = tf.constant(dt, dtype=tf.float32)

    constants['eps1']       = tf.constant((1.0-dt/tau), dtype=tf.float32)
    constants['eps2']       = tf.constant(np.sqrt(2.0*dt/tau), dtype=tf.float32)

    constants['N_exc']      = int(0.8*N)
    constants['N_inh']      = int(0.2*N)
    constants['N']          = N

    Tinvdiag                = np.ones([N])/tau
    Tinvdiag[int(0.8*N):]   = 2/tau
    constants['Tinv']       = tf.constant(np.diag(Tinvdiag), dtype=tf.float32)

    mask                    = np.ones([N,N])
    mask[:,int(0.8*N):]     = -1
    constants['mask']       = tf.constant(mask, dtype=tf.float32)

    if task == 'workingmemory_delayed-match-to-sample':

        constants['N_stim'] = 2
        constants['T_stim'] = int(0.5/dt)
        constants['T_wait'] = int(0.1/dt)
        constants['T_min']  = int(0.2/dt)
        constants['T_max']  = int(0.8/dt)

    return constants

def initvariables(task,constants,N_models):
    
    Wraw = tf.Variable(tf.random.normal(shape=[N_models,constants['N'],constants['N']])/constants['N'], dtype=tf.float32, name='Wraw') #Raw weight matrix before Dale's Law and other constraints
    bias = tf.Variable(tf.random.normal(shape=[N_models,constants['N'],1]), dtype=tf.float32, name='bias') #Input bias

    if task == 'workingmemory_delayed-match-to-sample':

        Win  = tf.Variable(tf.random.normal(shape=[N_models,constants['N_exc'],constants['N_stim']])/(constants['N_exc']), dtype=tf.float32, name='Win')
        Wout = tf.Variable(tf.random.normal(shape=[N_models,2,constants['N_exc']])/(constants['N_exc']), dtype=tf.float32, name='Wout')

        tv = [Wraw,bias,Win,Wout]

        return tv