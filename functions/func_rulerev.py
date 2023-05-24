from functions.func_init import *

class rulereversal:

    def __init__(self,constants):


        self.N_models = constants['N_models']
        self.type = 0

        self.tasktrials  = 5
        self.trials      = 40

        self.IO_inp   = 4 # 2 stim, right, wrong
        self.IO_out   = 4 # 2 response, 2 rule
        self.N_input  = int(constants['N'])
        self.N_output = int(constants['N'])

        self.T_test   = int(0.5/constants['dt'])
        self.T_short  = int(0.5/constants['dt'])
        self.T_long   = int(1.0/constants['dt'])
        self.T_jitter = int(0.1/constants['dt'])
        # self.T_jitter = 0
        self.T_wait   = int(0.1/constants['dt'])

        self.Wraw = tf.Variable(tf.random.normal(shape=[self.N_models,constants['N'],constants['N']])/constants['N'], dtype=tf.float32, name='Wraw')
        self.bias = tf.Variable(tf.random.normal(shape=[self.N_models,constants['N'],1]), dtype=tf.float32, name='bias')
        self.Winp = tf.Variable(tf.random.normal(shape=[self.N_models,constants['N'],self.IO_inp]), dtype=tf.float32, name='Winp')
        self.Wout = tf.Variable(tf.random.normal(shape=[self.N_models,self.IO_out,constants['N']]), dtype=tf.float32, name='Wout')
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.opt  = tf.optimizers.Adam(learning_rate=1e-1/constants['N'])
        self.tv   = [self.Winp,self.Wout,self.Wraw,self.bias]

        self.ureg = 1e-5
        self.wreg = 1e-5        

    def construct_trial(self):

        # rule 1: returns [1,0] when input is [1,0], and vice versa
        # rule 2: returns [0,1] when input is [1,0], and vice versa

        rules = np.zeros([self.tasktrials,self.trials])
        # trial 0: 00000
        # trial 1: 01100
        # trial 2: 00011
        # trial 3: 01111
        rules[1:3,5:10] = 1
        rules[3:5,10:15] = 1
        rules[1:5,15:20] = 1
        rules[1:3,25:30] = 1
        rules[3:5,30:35] = 1
        rules[1:5,35:40] = 1
        rules[0,20:40] = 1

        input  = np.zeros([self.tasktrials,self.trials,self.N_models,self.IO_inp,1])
        target = np.zeros([self.tasktrials,self.trials,self.N_models])
        ideal  = np.zeros([self.tasktrials,self.trials,self.N_models])

        for i in range(self.tasktrials):
            for j in range(self.trials):
            
                stim = np.random.randint(low=0,high=2)
                input[i,j,:,stim] = 1

                if rules[i,j] == 0:
                    target[i,j,:] = np.copy(input[i,j,:,0,0])
                else:
                    target[i,j,:] = np.copy(input[i,j,:,1,0])

                ideal[0,j,:] = np.copy(input[0,j,:,0,0])
                if i>0:
                    if rules[i-1,j] == 0:
                        ideal[i,j,:] = np.copy(input[i,j,:,0,0])
                    else:
                        ideal[i,j,:] = np.copy(input[i,j,:,1,0])

        trial = [input,target,rules,ideal]

        return [tf.constant(i,dtype=tf.float32) for i in trial]

    @tf.function
    def train(self,trial,constants,activation,dale,skip,ratio):

        input  = trial[0]
        target = trial[1]
        ideal  = trial[3]

        u = tf.zeros([self.trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.zeros([constants['N'],1])
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        with tf.GradientTape() as tape:

            if dale:
                Wtop = self.Wraw[:int(self.N_models/2)]
                Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
                W = tf.concat([Wtop,Wbot],axis=0)
            else:
                W = self.Wraw
            
            input = self.Winp@input
            right = tf.constant(np.array([[0],[0],[1],[0]]),dtype=tf.float32)
            right = self.Winp@right
            wrong = tf.constant(np.array([[0],[0],[0],[1]]),dtype=tf.float32)
            wrong = self.Winp@wrong
            
            cost = tf.zeros([self.N_models])

            for tr in tf.range(self.tasktrials):

                uskip = tf.identity(u)

                # sample
                if self.T_jitter>0:
                    timesteps1 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
                else:
                    timesteps1 = self.T_short
                u,eta,_ = compute_uall(u,eta,W,self.bias,input[tr],constants,skip,ratio,timesteps1,activation)

                # response
                if self.T_jitter>0:
                    timesteps2 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
                else:
                    timesteps2 = self.T_short
                u,eta,uall = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,timesteps2,activation)
                readout = tf.nn.softmax((self.Wout@uall[self.T_wait:])[:,:,:,:2],axis=3)[:,:,:,:,0] #[time,trials,models,units,1]
                timebroadcast = tf.ones([timesteps2-self.T_wait,1,1])
                cost += tf.reduce_mean(self.loss(ideal[tr] * timebroadcast, readout), axis=(0,1))/(self.tasktrials)
                cost += tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4)) * self.ureg /(self.tasktrials)
                cost += tf.reduce_mean(tf.abs(self.Wraw),axis=(1,2)) * self.wreg /(self.tasktrials)

                # reward
                if self.T_jitter>0:
                    timesteps3 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
                else:
                    timesteps3 = self.T_short

                response = tf.math.argmax(readout[-1],axis=2) #[trials,models,units] -> [trials,models]
                feedback = tf.cast(response == tf.cast(target[tr],tf.int64),tf.int32)
                inputreward = tf.TensorArray(dtype=tf.float32,size=self.trials)
                for i in tf.range(self.trials):
                    trialreward = tf.TensorArray(dtype=tf.float32,size=self.N_models)
                    for j in tf.range(self.N_models):
                        if feedback[i,j] == 0:
                            trialreward = trialreward.write(j,wrong[j])
                        else:
                            trialreward = trialreward.write(j,right[j])
                    trialreward = trialreward.stack()
                    inputreward = inputreward.write(i,trialreward)
                inputreward = inputreward.stack()
                u,eta,_ = compute_uall(u,eta,W,self.bias,inputreward,constants,skip,ratio,timesteps3,activation)

                # waiting
                if self.T_jitter>0:
                    timesteps4 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
                else:
                    timesteps4 = self.T_short
                u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,timesteps4,activation)

                if self.type==3:
                    u = ratio*u + (1-ratio)*uskip

            grads = tape.gradient(cost, self.tv)
        
        self.opt.apply_gradients(zip(grads, self.tv))

        return cost

    def eval(self,trial,constants,activation,dale):

        skip = 0
        ratio = 1
        input  = trial[0]
        target = trial[1]
        ideal = trial[3]

        u = tf.zeros([self.trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.zeros([constants['N'],1])
        input = self.Winp@input
        right = tf.constant(np.array([[0],[0],[1],[0]]),dtype=tf.float32)
        right = self.Winp@right
        wrong = tf.constant(np.array([[0],[0],[0],[1]]),dtype=tf.float32)
        wrong = self.Winp@wrong

        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)
        performance = tf.zeros([self.N_models], dtype=tf.float32)

        for tr in tf.range(self.tasktrials):

            # sample
            timesteps1 = self.T_short
            u,eta,_ = compute_uall(u,eta,W,self.bias,input[tr],constants,skip,ratio,timesteps1,activation)

            # response
            timesteps2 = self.T_short
            u,eta,uall = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,timesteps2,activation)
            readout = tf.nn.softmax((self.Wout@uall[self.T_wait:])[:,:,:,:2],axis=3)[:,:,:,:,0] #[time,trials,models,units,1]

            # reward
            timesteps3 = self.T_short
            response = tf.math.argmax(readout[-1],axis=2) #[trials,models,units] -> [trials,models]
            longresponse = tf.math.argmax(readout,axis=3) #[trials,models,units] -> [time,trials,models]
            timebroadcast = tf.ones([timesteps2-self.T_wait,1,1])
            idealresponse = tf.cast(longresponse == tf.cast(timebroadcast*ideal[tr],tf.int64),tf.int32)
            performance += tf.reduce_mean(tf.cast(idealresponse,dtype=tf.float32),axis=(0,1))/(self.tasktrials)
            feedback = tf.cast(response == tf.cast(target[tr],tf.int64),tf.int32)
            inputreward = tf.TensorArray(dtype=tf.float32,size=self.trials)
            for i in tf.range(self.trials):
                trialreward = tf.TensorArray(dtype=tf.float32,size=self.N_models)
                for j in tf.range(self.N_models):
                    if feedback[i,j] == 0:
                        trialreward = trialreward.write(j,wrong[j])
                    else:
                        trialreward = trialreward.write(j,right[j])
                trialreward = trialreward.stack()
                inputreward = inputreward.write(i,trialreward)
            inputreward = inputreward.stack()
            u,eta,_ = compute_uall(u,eta,W,self.bias,inputreward,constants,skip,ratio,timesteps3,activation)

            # waiting
            timesteps4 = self.T_short
            u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,timesteps4,activation)

        return performance.numpy()

    def trainmodel(self,constants,iterations,activation,dale,skip,ratio):
        costs = []
        for iter in tf.range(iterations):
            trial = self.construct_trial()
            cost = self.train(trial,constants,activation,dale,skip,ratio)
            costs.append(cost.numpy())
    
        return np.array(costs)