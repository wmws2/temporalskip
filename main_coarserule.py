from functions.func_rulerev import *

typeno = int(sys.argv[1])
simno = int(sys.argv[2])
step = 2
init = 10
skip = 1
ratio = 1

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

N         = 20
tau       = 0.1
N_models  = 10
basedt = 0.005
baseconstants = initconstants(basedt,tau,N,N_models)
basetask = rulereversal(baseconstants)

iterations = 10

allcosts = np.zeros([0,N_models])
allperfs = np.zeros([0,N_models])
alltimes = []

totalcounter = 0

dt = 0.005*10
constants = initconstants(dt,tau,N,N_models)
task = rulereversal(constants)
if typeno==6:
    task.type=3

counter = 0
while counter<int(500/step):

    start = time.time()
    costs = task.trainmodel(constants,iterations,activation,dale,1,1)
    timetaken = time.time() - start
    basetask.Wraw.assign(task.Wraw)
    basetask.bias.assign(task.bias)
    basetask.Winp.assign(task.Winp)
    basetask.Wout.assign(task.Wout)
    basetask.opt.from_config(task.opt.get_config())

    trial = basetask.construct_trial()
    perfs = basetask.eval(trial,baseconstants,activation,dale)[np.newaxis]
    allcosts = np.concatenate([allcosts,costs],axis=0)
    allperfs = np.concatenate([allperfs,perfs],axis=0)
    alltimes.append(timetaken)
    tf.print(counter,'cost',np.mean(costs),'performance',np.mean(perfs),'bestperformance',np.amax(perfs),timetaken)
    counter += 1
    totalcounter += 1

counter = 0
while totalcounter<1000:
    start = time.time()
    costs = basetask.trainmodel(baseconstants,iterations,activation,dale,skip,ratio)
    timetaken = time.time() - start
    trial = basetask.construct_trial()
    perfs = basetask.eval(trial,baseconstants,activation,dale)[np.newaxis]
    allcosts = np.concatenate([allcosts,costs],axis=0)
    allperfs = np.concatenate([allperfs,perfs],axis=0)
    alltimes.append(timetaken)
    tf.print(counter,'cost',np.mean(costs),'performance',np.mean(perfs),'bestperformance',np.amax(perfs),timetaken)
    counter += 1
    totalcounter += 1

np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_Wraw.npy',task.Wraw.numpy())
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_bias.npy',task.bias.numpy())
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_Winp.npy',task.Winp.numpy())
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_Wout.npy',task.Wout.numpy())
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_costs.npy',allcosts)
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_perfs.npy',allperfs)
np.save('./resultsrule2/' + str(typeno) + '_' + str(simno) + '_times.npy',alltimes)
