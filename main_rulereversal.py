from functions.func_rulerev import *

typeno = int(sys.argv[1])
simno = int(sys.argv[2])

skips = [1,10,20,40,110,120,140]
steps = [1,2,6,11]
inits = [0,0.2,0.5,0.8]
a = 'relu'
d = 0
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
dt        = 0.005
tau       = 0.1
N_models  = 10
constants = initconstants(dt,tau,N,N_models)
task      = rulereversal(constants)

if typeno == 0:
    skip = skips[0]
    step = steps[0]
    init = inits[0]
    task.type = 0
if typeno == 1:
    skip = skips[6]
    step = steps[3]
    init = inits[3]
    task.type = 1
if typeno == 2:
    skip = skips[3]
    step = steps[1]
    init = inits[2]
    task.type = 2
if typeno == 3:
    skip = skips[6]
    step = steps[3]
    init = inits[3]
    task.type = 3  
if typeno == 4:
    skip = skips[3]
    step = steps[1]
    init = inits[2]
    task.type = 3  
if typeno == 7:
    skip = skips[0]
    step = steps[0]
    init = inits[0]
    task.type = 3  


iterations = 10
allcosts = np.zeros([0,N_models])
allperfs = np.zeros([0,N_models])
alltimes = []
totalcounter = 0

for epoch in tf.range(step):
    
    if step>1:
        ratio = tf.constant(init + (1-init)/(step-1)*epoch.numpy(),dtype=tf.float32)
    else:
        ratio = tf.constant(1.,dtype=tf.float32)

    counter = 0
    while counter<int(500/step):
        start = time.time()
        costs = task.trainmodel(constants,iterations,activation,dale,skip,ratio)
        timetaken = time.time() - start
        trial = task.construct_trial()
        perfs = task.eval(trial,constants,activation,dale)[np.newaxis]
        allcosts = np.concatenate([allcosts,costs],axis=0)
        allperfs = np.concatenate([allperfs,perfs],axis=0)
        alltimes.append(timetaken)
        tf.print(epoch,counter,'cost',np.mean(costs),'performance',np.mean(perfs),'bestperformance',np.amax(perfs),timetaken)
        counter += 1
        totalcounter += 1

epoch = step
counter = 0
while totalcounter<1000:
    start = time.time()
    costs = task.trainmodel(constants,iterations,activation,dale,skip,ratio)
    timetaken = time.time() - start
    trial = task.construct_trial()
    perfs = task.eval(trial,constants,activation,dale)[np.newaxis]
    allcosts = np.concatenate([allcosts,costs],axis=0)
    allperfs = np.concatenate([allperfs,perfs],axis=0)
    alltimes.append(timetaken)
    tf.print(epoch,counter,'cost',np.mean(costs),'performance',np.mean(perfs),'bestperformance',np.amax(perfs),timetaken)
    counter += 1
    totalcounter += 1

np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_Wraw.npy',task.Wraw.numpy())
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_bias.npy',task.bias.numpy())
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_Winp.npy',task.Winp.numpy())
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_Wout.npy',task.Wout.numpy())
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_costs.npy',allcosts)
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_perfs.npy',allperfs)
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_times.npy',alltimes)
