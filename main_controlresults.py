from functions.func_init import *
from functions.func_tasks import *

# activation = tf.nn.relu
# dale = 0
# basictasks = ['go','rtgo','dlygo','anti','rtanti','dlyanti','dm1','ctxdm1','multsendm','dlydm1','ctxdlydm1','multsendlydm','dms','dnms','dmc','dnmc']
# skips = [1,5,10,20,30,40]
# steps = [1,2,4,5,10]
# ts = [5,25,50,100,150,200]

# N         = 50
# tau       = 0.1
# N_models  = 100

# perfs = np.zeros([16,6,2])

# for tno in tf.range(16):
#     tf.print(tno)
#     taskname = basictasks[tno]
#     basedt = 0.005
#     baseconstants = initconstants(basedt,tau,N,N_models)
#     basetask = yangtasks(taskname,baseconstants)

#     for i in tf.range(6):
#         modeldt = ts[i]
#         dt = float(modeldt)/1000
#         constants = initconstants(dt,tau,N,N_models)
#         task      = yangtasks(taskname,constants)
#         task.Wraw = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Wraw.npy')
#         task.bias = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_bias.npy')
#         task.Winp = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Winp.npy')
#         task.Wout = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Wout.npy')

#         performanceres,performance,uinf,_ = task.eval(constants,activation,dale)
#         chosenmodels = np.array(np.argsort(performance)[int(0.1*N_models):int(0.9*N_models)])
#         currentperf = tf.reduce_mean(tf.gather(performance,chosenmodels))
#         perfs[tno,i,0] = currentperf.numpy()

#         basetask.Wraw = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Wraw.npy')
#         basetask.bias = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_bias.npy')
#         basetask.Winp = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Winp.npy')
#         basetask.Wout = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Wout.npy')

#         performanceres,performance,uinf,_ = basetask.eval(baseconstants,activation,dale)
#         currentperf = tf.reduce_mean(tf.gather(performance,chosenmodels))
#         perfs[tno,i,1] = currentperf.numpy()

# np.save('./analysis/coarse_perfs.npy',perfs)
#------------------------------------------------------------------------------------------------

# activation = tf.nn.relu
# dale = 0
# basictasks = ['go','rtgo','dlygo','anti','rtanti','dlyanti','dm1','ctxdm1','multsendm','dlydm1','ctxdlydm1','multsendlydm','dms','dnms','dmc','dnmc']
# skips = [5,10,20,30,40,105,110,120,130,140]

# N         = 50
# tau       = 0.1
# N_models  = 100
# dt        = 0.005
# constants = initconstants(dt,tau,N,N_models)
# ratio = 0.5

# perfs = np.zeros([16,10,2])

# for tno in tf.range(16):
#     tf.print(tno)
#     taskname = basictasks[tno]
#     task = yangtasks(taskname,constants)
    
#     for i in tf.range(10):
#         skip = skips[i]

#         task.Wraw = np.load('./controlresults2/' + taskname + '_' + str(i.numpy()) + '_Wraw.npy')
#         task.bias = np.load('./controlresults2/' + taskname + '_' + str(i.numpy()) + '_bias.npy')
#         task.Winp = np.load('./controlresults2/' + taskname + '_' + str(i.numpy()) + '_Winp.npy')
#         task.Wout = np.load('./controlresults2/' + taskname + '_' + str(i.numpy()) + '_Wout.npy')

#         performanceres,performance,uinf,_ = task.evalskip(constants,activation,dale,skip,ratio)
#         chosenmodels = np.array(np.argsort(performance)[int(0.1*N_models):int(0.9*N_models)])
#         currentperf = tf.reduce_mean(tf.gather(performance,chosenmodels))
#         perfs[tno,i,0] = currentperf.numpy()

#         performanceres,performance,uinf,_ = task.eval(constants,activation,dale)
#         currentperf = tf.reduce_mean(tf.gather(performance,chosenmodels))
#         perfs[tno,i,1] = currentperf.numpy()

# np.save('./analysis/skip_perfs.npy',perfs)

#------------------------------------------------------------------------------------------------
activation = tf.nn.relu
dale = 0
basictasks = ['go','rtgo','dlygo','anti','rtanti','dlyanti','dm1','ctxdm1','multsendm','dlydm1','ctxdlydm1','multsendlydm','dms','dnms','dmc','dnmc']
skips = [1,5,10,20,30,40]
steps = [1,2,4,5,10]
ts = [5,25,50,100,150,200]

N         = 50
tau       = 0.1
N_models  = 100

mses = np.zeros([16,6])

for tno in tf.range(16):
    tf.print(tno)
    taskname = basictasks[tno]
    basedt = 0.005
    baseconstants = initconstants(basedt,tau,N,N_models)
    basetask = yangtasks(taskname,baseconstants)

    for i in tf.range(6):

        modeldt = ts[i]
        dt = float(modeldt)/1000
        constants = initconstants(dt,tau,N,N_models)
        task      = yangtasks(taskname,constants)
        task.Wraw = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Wraw.npy')
        task.bias = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_bias.npy')
        task.Winp = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Winp.npy')
        task.Wout = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Wout.npy')

        performanceres,performance,uinf,_ = task.eval(constants,activation,dale)
        chosenmodels = np.array(np.argsort(performance)[int(0.1*N_models):int(0.9*N_models)])

        trial = task.construct_trial(trials=task.evaltrials)
        uall1 = task.uall(constants,activation,dale,trial,0,1).numpy()[:,:,chosenmodels]

        basetask.Wraw = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Wraw.npy')
        basetask.bias = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_bias.npy')
        basetask.Winp = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Winp.npy')
        basetask.Wout = np.load('./controlresults/' + taskname + '_coarse_' + str(modeldt) + 'ms_Wout.npy')

        uall2 = basetask.uall(baseconstants,activation,dale,trial,0,1).numpy()[:,:,chosenmodels]

        mses[tno,i] = distance(taskname,uall1,modeldt,uall2,5)

np.save('./analysis/coarse_mses.npy',mses)

#------------------------------------------------------------------------------------------------

# activation = tf.nn.relu
# dale = 0
# basictasks = ['go','rtgo','dlygo','anti','rtanti','dlyanti','dm1','ctxdm1','multsendm','dlydm1','ctxdlydm1','multsendlydm','dms','dnms','dmc','dnmc']
# skips = [5,10,20,30,40,105,110,120,130,140]

# N         = 50
# tau       = 0.1
# N_models  = 100
# dt        = 0.005
# constants = initconstants(dt,tau,N,N_models)
# ratio = 0.5

# mses = np.zeros([16,10])

# for tno in tf.range(16):
#     tf.print(tno)
#     taskname = basictasks[tno]
#     task = yangtasks(taskname,constants)
    
#     for i in tf.range(10):
#         skip = skips[i]

#         task.Wraw = np.load('./controlresults2/' + taskname + '_' + str(i.numpy()) + '_Wraw.npy')
#         task.bias = np.load('./controlresults2/' + taskname + '_' + str(i.numpy()) + '_bias.npy')
#         task.Winp = np.load('./controlresults2/' + taskname + '_' + str(i.numpy()) + '_Winp.npy')
#         task.Wout = np.load('./controlresults2/' + taskname + '_' + str(i.numpy()) + '_Wout.npy')

#         performanceres,performance,uinf,_ = task.evalskip(constants,activation,dale,skip,ratio)
#         chosenmodels = np.array(np.argsort(performance)[int(0.1*N_models):int(0.9*N_models)])

#         trial = task.construct_trial(trials=task.evaltrials)
#         uall1 = task.uall(constants,activation,dale,trial,skip,ratio).numpy()[:,:,chosenmodels]
#         uall2 = task.uall(constants,activation,dale,trial,0,1).numpy()[:,:,chosenmodels]

#         mses[tno,i] = distance(taskname,uall1,5,uall2,5)

# np.save('./analysis/skip_mses.npy',mses)