from functions.func_init import *

# -------------------------------------------------------
# Simulate dynamics
# -------------------------------------------------------

@tf.function
def compute_dueta(u,eta,W,bias,hinput,constants,activation):

    du  = constants['Tinv']@(-u + activation(W@u + bias + 0.1*eta + hinput))*constants['dt']
    eta = constants['eps1']*eta + constants['eps2']*tf.random.normal(tf.shape(u))

    return du,eta

@tf.function
def compute_uall(u,eta,W,bias,hinput,constants,skip,ratio,timesteps,activation):

    du,eta = compute_dueta(u,eta,W,bias,hinput,constants,activation)
    uskip = tf.identity(u) + skip*du
    uall  = tf.TensorArray(dtype=tf.float32, size=timesteps)

    for t in tf.range(timesteps):

        du,eta = compute_dueta(u,eta,W,bias,hinput,constants,activation)
        u = u + du

        if skip>0 and t%skip==(skip-1):
            u = ratio*u + (1-ratio)*uskip
            uskip = tf.identity(u) + skip*du
        
        uall = uall.write(t,u)

    return u,eta,uall.stack()

# -------------------------------------------------------
# Task-specific training functions
# FIXED GRAPHS only (i.e. accelerated by @tf.function)
# -------------------------------------------------------

@tf.function
def train_wmdms(constants,tv,activation,opt,skip,ratio,input0,input1,input2,target,N_models):

    activitypenalty = 1e-4
    Wraw = tv[0]
    bias = tv[1]
    Win  = tv[2]
    Wout = tv[3]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    upenalty = 0.
    u = tf.zeros([constants['N_stim']**2,N_models,constants['N'],1])
    eta = tf.random.normal(tf.shape(u))
    
    W = tf.abs(Wraw)*constants['mask']
    input0 = tf.pad(Win@input0,((0,0),(0,0),(0,constants['N_inh']),(0,0)))
    u,eta,_ = compute_uall(u,eta,W,bias,input0,constants,skip,ratio,5*constants['T_stim'],activation)

    with tf.GradientTape() as tape:

        input1 = tf.pad(Win@input1,((0,0),(0,0),(0,constants['N_inh']),(0,0)))
        input2 = tf.pad(Win@input2,((0,0),(0,0),(0,constants['N_inh']),(0,0)))
        W = tf.abs(Wraw)*constants['mask']
        W = Wraw

        # Sample stimulus
        #timesteps = tf.random.uniform(shape=(),minval=constants['T_min'], maxval=constants['T_max'], dtype=tf.int32)
        timesteps = constants['T_stim']
        u,eta,uall = compute_uall(u,eta,W,bias,input1,constants,skip,ratio,timesteps,activation)
        upenalty += activitypenalty*tf.reduce_mean(uall**2)

        # Delay period
        #timesteps = tf.random.uniform(shape=(),minval=2*constants['T_min'], maxval=2*constants['T_max'], dtype=tf.int32)
        timesteps = constants['T_stim']
        u,eta,uall = compute_uall(u,eta,W,bias,input0,constants,skip,ratio,timesteps,activation)
        upenalty += activitypenalty*tf.reduce_mean(uall**2)            

        # Test period
        #timesteps = tf.random.uniform(shape=(),minval=constants['T_min'], maxval=constants['T_max'], dtype=tf.int32)
        timesteps = constants['T_stim']
        u,eta,uall = compute_uall(u,eta,W,bias,input2,constants,skip,ratio,timesteps,activation)
        upenalty += activitypenalty*tf.reduce_mean(uall**2)

        # Response period
        #timesteps = tf.random.uniform(shape=(),minval=constants['T_min'], maxval=constants['T_max'], dtype=tf.int32)
        timesteps = constants['T_stim']
        u,eta,uall = compute_uall(u,eta,W,bias,input0,constants,skip,ratio,timesteps,activation)
        upenalty += activitypenalty*tf.reduce_mean(uall**2)

        timebroadcast = tf.ones([timesteps,1,1])
        pred = tf.nn.softmax(Wout@uall[:,:,:,:constants['N_exc']],axis=3)[:,:,:,:,0]
        cost = tf.reduce_mean(loss(target*timebroadcast,pred),axis=(0,1))
        grads= tape.gradient(tf.reduce_mean(cost) + upenalty, tv)

    opt.apply_gradients(zip(grads, tv))

    return cost,upenalty

# -------------------------------------------------------
# Final training function
# NO TRACING HERE
# -------------------------------------------------------
def train(constants,tv,task,activation,opt,skip,ratio,iterations,N_models):

    if task == 'workingmemory_delayed-match-to-sample':

        input0 = np.zeros([constants['N_stim']**2,N_models,constants['N_stim'],1])
        input1 = np.zeros([constants['N_stim']**2,N_models,constants['N_stim'],1])
        input2 = np.zeros([constants['N_stim']**2,N_models,constants['N_stim'],1])
        target = np.zeros([constants['N_stim']**2,N_models])
        for i in range(constants['N_stim']):
            input1[i*constants['N_stim']:(i+1)*constants['N_stim'],:,i] = 1
            input2[i::constants['N_stim'],:,i] = 1   
            target[i*constants['N_stim']+i] = 1

        input0 = tf.constant(input0,dtype=tf.float32)
        input1 = tf.constant(input1,dtype=tf.float32)
        input2 = tf.constant(input2,dtype=tf.float32)
        target = tf.constant(target,dtype=tf.float32)

        costs = []
        penalties = []

        for iter in tf.range(iterations):
            cost,penalty = train_wmdms(constants,tv,activation,opt,skip,ratio,input0,input1,input2,target,N_models)
            costs.append(cost.numpy())
            penalties.append(penalty.numpy())

            if iter%100 == 0:
                tf.print(skip,iter,cost,penalty)

        return costs,penalties

