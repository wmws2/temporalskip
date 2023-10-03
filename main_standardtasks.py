from functions.func_standardtasks import *

activation  = tf.nn.relu
dale        = False
basictasks  = ['go','rtgo','dlygo','anti','rtanti','dlyanti','dm1','ctxdm1','multsendm','dlydm1','ctxdlydm1','multsendlydm','dms','dnms','dmc','dnmc']
skips       = [1,10,20,40,110,120,140]
steps       = [1,2,6,11]
inits       = [0,0.2,0.5,0.8]

taskno      = int(sys.argv[1])
skipno      = int(sys.argv[2])
stepno      = int(sys.argv[3])
initno      = int(sys.argv[4])

step        = steps[stepno]
skip        = skips[skipno]
init        = inits[initno]
taskname    = basictasks[taskno]

N           = 50
dt          = 0.005
tau         = 0.1
N_models    = 100
constants   = initconstants(dt,tau,N,N_models)
task        = yangtasks(taskname,constants)
iterations  = 10 

allcosts    = np.zeros([0,N_models])
allperfs    = np.zeros([0,2,N_models])
alltimes    = []
counters    = []
totalcounter= 0
currentperf = 0.

for epoch in tf.range(step):
    
    if step>1:
        ratio = tf.constant(init + (1-init)/(step-1)*epoch.numpy(),dtype=tf.float32)
    else:
        ratio = tf.constant(1.,dtype=tf.float32)

    counter = 0

    while currentperf < 0.99 and totalcounter<2000 and counter<int(2000/step):

        start = time.time()
        costs = task.trainmodel(constants,iterations,activation,dale,skip,ratio)
        timetaken = time.time() - start
        performanceres,performance,uinf,_ = task.eval(constants,activation,dale)
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
    costs = task.trainmodel(constants,iterations,activation,dale,1,1)
    timetaken = time.time() - start
    performanceres,performance,uinf,_ = task.eval(constants,activation,dale)
    allcosts = np.concatenate([allcosts,costs],axis=0)
    perfs = np.array([performanceres,performance])[np.newaxis]
    allperfs = np.concatenate([allperfs,perfs],axis=0)
    alltimes.append(timetaken)

    counter += 1
    totalcounter += 1

    currentperfsort = np.sort(performance)
    currentperf = np.mean(currentperfsort[int(0.1*N_models):int(0.9*N_models)])
    
    tf.print(taskno,taskname,str(skipno) + str(stepno) + str(initno),'Epoch',epoch,counter,'| Current Performance:',currentperf,'Task:',tf.reduce_mean(performanceres),'Activity:',tf.reduce_mean(uinf),'| Time taken:',timetaken)

counters.append(totalcounter)
