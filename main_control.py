from functions.func_tasks import *

a = 'relu'
d = 0
taskno = int(sys.argv[1])
basictasks = ['go','rtgo','dlygo','anti','rtanti','dlyanti','dm1','ctxdm1','multsendm','dlydm1','ctxdlydm1','multsendlydm','dms','dnms','dmc','dnmc']
taskname = basictasks[taskno]

if a == 'relu':
    activation = tf.nn.relu
elif a == 'tanh':
    activation = tf.math.tanh
elif a == 'sftp':
    activation = tf.math.softplus
if d == 0:
    dale = False
else:
    dale = True

N         = 50
dt        = float(sys.argv[2])/1000
tau       = 0.1
N_models  = 100
constants = initconstants(dt,tau,N,N_models)
task      = yangtasks(taskname,constants)


iterations = 10

allcosts = np.zeros([0,N_models])
allperfs = np.zeros([0,2,N_models])
alltimes = []
counters = []

totalcounter = 0
currentperf = 0.

while totalcounter<400:

    start = time.time()
    costs = task.trainmodel(constants,iterations,activation,dale,0,0)
    timetaken = time.time() - start
    performanceres,performance,uinf,_ = task.eval(constants,activation,dale)
    allcosts = np.concatenate([allcosts,costs],axis=0)
    perfs = np.array([performanceres,performance])[np.newaxis]
    allperfs = np.concatenate([allperfs,perfs],axis=0)
    alltimes.append(timetaken)

    totalcounter += 1

    currentperfsort = np.sort(performance)
    currentperf = np.mean(currentperfsort[int(0.1*N_models):int(0.9*N_models)])
    
    tf.print(str(int(dt*1000)),totalcounter,'| Current Performance:',currentperf,'Task:',tf.reduce_mean(performanceres),'Activity:',tf.reduce_mean(uinf),'| Time taken:',timetaken)

counters.append(totalcounter)

np.save('./controlresults/' + taskname + '_coarse_' + str(int(dt*1000)) + 'ms_Wraw.npy',task.Wraw.numpy())
np.save('./controlresults/' + taskname + '_coarse_' + str(int(dt*1000)) + 'ms_bias.npy',task.bias.numpy())
np.save('./controlresults/' + taskname + '_coarse_' + str(int(dt*1000)) + 'ms_Winp.npy',task.Winp.numpy())
np.save('./controlresults/' + taskname + '_coarse_' + str(int(dt*1000)) + 'ms_Wout.npy',task.Wout.numpy())
np.save('./controlresults/' + taskname + '_coarse_' + str(int(dt*1000)) + 'ms_costs.npy',allcosts)
np.save('./controlresults/' + taskname + '_coarse_' + str(int(dt*1000)) + 'ms_perfs.npy',allperfs)
np.save('./controlresults/' + taskname + '_coarse_' + str(int(dt*1000)) + 'ms_times.npy',alltimes)
np.save('./controlresults/' + taskname + '_coarse_' + str(int(dt*1000)) + 'ms_steps.npy',counters)
