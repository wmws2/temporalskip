from functions.func_train import *

N_models = 1000
task = 'workingmemory_delayed-match-to-sample'
dt = 0.01
tau = 20*dt
N = 50

constants = initconstants(dt,tau,N,task)
tv        = initvariables(task,constants,N_models)
opt       = tf.optimizers.Adam(learning_rate=1e-3)
activation= tf.math.tanh

#Training with skip connections
skip = 10
allcosts = []
allpens = []

for i in range(10):
    ratio = 0.99-0.11*i
    iterations = 5000
    costs,penalties = train(constants,tv,task,activation,opt,skip,ratio,iterations,N_models)
    allcosts.append(costs)
    allpens.append(penalties)

allcosts = np.concatenate(allcosts,axis=0)
allpens = np.concatenate(allpens,axis=0)

np.save('./results/' + str(N_models) + 'models_' + str(int(1000*dt)) + 'ms_' + str(N) + 'N_' + str(skip) + 'skip_costs.npy',allcosts)
np.save('./results/' + str(N_models) + 'models_' + str(int(1000*dt)) + 'ms_' + str(N) + 'N_' + str(skip) + 'skip_penalties.npy',allpens)