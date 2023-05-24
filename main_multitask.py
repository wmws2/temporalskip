from functions.func_tasks import *

taskname = 'multitask'
actvfunc = 'relu'

N         = 500
dt        = 0.02
tau       = 5*dt
N_models  = 8
constants = initconstants(dt,tau,N,N_models)
task      = yangtasks(taskname,constants)

iterations = 10
if actvfunc == 'relu':
    activation = tf.nn.relu
if actvfunc == 'softplus':
    activation = tf.math.softplus

skip = 0

allcosts = np.zeros([0,N_models])
allperfs = np.zeros([0,2,20,N_models])
alltimes = []
task.opt = tf.optimizers.Adam(learning_rate=1e-1/constants['N'])
for epoch in tf.range(11):
    
    ratio = 0.1*epoch.numpy()

    for cp in tf.range(100):

        start = time.time()
        costs = task.trainmodel(constants,iterations,activation,skip,ratio)
        timetaken = time.time() - start
        performanceres,performance,uinf = task.eval(constants,activation)
        tf.print('Epoch',epoch,cp,'| Performance:',tf.reduce_mean(performance),'Task:',tf.reduce_mean(performanceres),'Activity:',tf.reduce_mean(uinf),'| Time taken:',timetaken)
        for ta in tf.range(20):
            tf.print('Task',ta,'Full Performance:',tf.reduce_mean(performance[ta]),'Task Performance:',tf.reduce_mean(performanceres[ta]))
        allcosts = np.concatenate([allcosts,costs],axis=0)
        perfs = np.array([performanceres,performance])[np.newaxis]
        allperfs = np.concatenate([allperfs,perfs],axis=0)
        alltimes.append(timetaken)
    
        np.save('./parameters/' + taskname + '_' + actvfunc + '_' + str(skip) + '_Wraw5.npy',task.Wraw.numpy())
        np.save('./parameters/' + taskname + '_' + actvfunc + '_' + str(skip) + '_bias5.npy',task.bias.numpy())
        np.save('./parameters/' + taskname + '_' + actvfunc + '_' + str(skip) + '_Winp5.npy',task.Winp.numpy())
        np.save('./parameters/' + taskname + '_' + actvfunc + '_' + str(skip) + '_Wout5.npy',task.Wout.numpy())
        np.save('./results/' + taskname + '_' + actvfunc + '_' + str(skip) + '_costs5.npy',allcosts)
        np.save('./results/' + taskname + '_' + actvfunc + '_' + str(skip) + '_perfs5.npy',allperfs)
        np.save('./results/' + taskname + '_' + actvfunc + '_' + str(skip) + '_times5.npy',alltimes)
    
    task.opt = tf.optimizers.Adam(learning_rate=1e-2/constants['N'])
    
