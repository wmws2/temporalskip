from functions.func_tasks import *

activation  = tf.nn.relu
dale        = False
basictasks  = ['go','rtgo','dlygo','anti','rtanti','dlyanti','dm1','ctxdm1','multsendm','dlydm1','ctxdlydm1','multsendlydm','dms','dnms','dmc','dnmc']
steps       = [2,6,11]
init        = 10

taskno      = int(sys.argv[1])
stepno      = int(sys.argv[2])

step        = steps[stepno]
taskname    = basictasks[taskno]

N           = 50
tau         = 0.1
N_models    = 100
basedt      = 0.005
baseconstants = initconstants(basedt,tau,N,N_models)
basetask    = yangtasks(taskname,baseconstants)
iterations  = 10

allcosts    = np.zeros([0,N_models])
allperfs    = np.zeros([0,2,N_models])
alltimes    = []
counters    = []
totalcounter= 0
currentperf = 0.

for epoch in tf.range(step):
    
    alpha = init/(step-1)*(step-1-epoch.numpy())
    if alpha == 0:
        alpha = 1

    dt = 0.005*alpha
    constants = initconstants(dt,tau,N,N_models)
    task = yangtasks(taskname,constants)
    task.Wraw.assign(basetask.Wraw)
    task.bias.assign(basetask.bias)
    task.Winp.assign(basetask.Winp)
    task.Wout.assign(basetask.Wout)
    task.opt.from_config(basetask.opt.get_config())
    counter = 0

    while currentperf < 0.99 and counter<int(2000/step):

        tf.print('slowdown')
        start = time.time()
        costs = task.trainmodel(constants,iterations,activation,dale,1,1)
        timetaken = time.time() - start
        tf.print('slowdown')
        basetask.Wraw.assign(task.Wraw)
        basetask.bias.assign(task.bias)
        basetask.Winp.assign(task.Winp)
        basetask.Wout.assign(task.Wout)
        basetask.opt.from_config(task.opt.get_config())

        performanceres,performance,uinf,_ = basetask.eval(baseconstants,activation,dale)
        allcosts = np.concatenate([allcosts,costs],axis=0)
        perfs = np.array([performanceres,performance])[np.newaxis]
        allperfs = np.concatenate([allperfs,perfs],axis=0)
        alltimes.append(timetaken)

        counter += 1
        totalcounter += 1

        currentperfsort = np.sort(performance)
        currentperf = np.mean(currentperfsort[int(0.1*N_models):int(0.9*N_models)])
        
        tf.print(taskno,taskname,'Epoch',epoch,counter,'| Current Performance:',currentperf,'Task:',tf.reduce_mean(performanceres),'Activity:',tf.reduce_mean(uinf),'| Time taken:',timetaken)
    
    counters.append(totalcounter)

epoch = step
counter = 0
while currentperf < 0.99 and totalcounter<4000:
    start = time.time()
    costs = basetask.trainmodel(baseconstants,iterations,activation,dale,1,1)
    timetaken = time.time() - start
    performanceres,performance,uinf,_ = basetask.eval(baseconstants,activation,dale)
    allcosts = np.concatenate([allcosts,costs],axis=0)
    perfs = np.array([performanceres,performance])[np.newaxis]
    allperfs = np.concatenate([allperfs,perfs],axis=0)
    alltimes.append(timetaken)

    counter += 1
    totalcounter += 1

    currentperfsort = np.sort(performance)
    currentperf = np.mean(currentperfsort[int(0.1*N_models):int(0.9*N_models)])
    
    tf.print(taskno,taskname + '_coarse',str(stepno),'Epoch',epoch,counter,'| Current Performance:',currentperf,'Task:',tf.reduce_mean(performanceres),'Activity:',tf.reduce_mean(uinf),'| Time taken:',timetaken)

counters.append(totalcounter)
