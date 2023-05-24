from functions.func_init import *

class yangtasks:

    def __init__(self,task,constants):

        self.tasks  = ['go','rtgo','dlygo','anti','rtanti','dlyanti','dm1','dm2','ctxdm1','ctxdm2','multsendm','dlydm1','dlydm2','ctxdlydm1','ctxdlydm2','multsendlydm','dms','dnms','dmc','dnmc']
        self.tasksA = ['go','anti','rtgo','rtanti','dm1','dm2','ctxdm1','ctxdm2','multsendm']
        self.tasksB = ['dlygo','dlyanti']
        self.tasksC = ['dlydm1','dlydm2','ctxdlydm1','ctxdlydm2','multsendlydm','dms','dnms','dmc','dnmc']
        self.task     = task
        self.N_models = constants['N_models']

        self.trials      = 20
        self.evaltrials  = 200

        self.IO_inp   = 85
        self.IO_out   = 33
        self.N_input  = int(0.4*constants['N'])
        self.N_output = int(0.4*constants['N'])
        self.N_pad    = int(0.6*constants['N'])

        self.T_test   = int(0.5/constants['dt'])
        self.T_short  = int(0.5/constants['dt'])
        self.T_long   = int(1.0/constants['dt'])
        self.T_jitter = int(0.1/constants['dt'])
        self.T_wait   = int(0.2/constants['dt'])

        self.Wraw = tf.Variable(tf.random.normal(shape=[self.N_models,constants['N'],constants['N']])/constants['N'], dtype=tf.float32, name='Wraw')
        self.bias = tf.Variable(tf.random.normal(shape=[self.N_models,constants['N'],1]), dtype=tf.float32, name='bias')
        self.Winp = tf.Variable(tf.random.normal(shape=[self.N_models,self.N_input,self.IO_inp]), dtype=tf.float32, name='Winp')
        self.Wout = tf.Variable(tf.random.normal(shape=[self.N_models,self.IO_out,self.N_output]), dtype=tf.float32, name='Wout')
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.opt  = tf.optimizers.Adam(learning_rate=1e-1/constants['N'])
        self.tv   = [self.Winp,self.Wout,self.Wraw,self.bias]

        # if task in self.tasks:

        #     ureg = np.zeros([self.N_models])
        #     ureg[int(0/4*self.N_models):int(1/4*self.N_models)] = 1e-5
        #     ureg[int(1/4*self.N_models):int(2/4*self.N_models)] = 1e-3
        #     ureg[int(2/4*self.N_models):int(3/4*self.N_models)] = 1e-5
        #     ureg[int(3/4*self.N_models):int(4/4*self.N_models)] = 1e-3
        #     self.ureg = tf.constant(ureg, dtype=tf.float32)

        #     wreg = np.zeros([self.N_models])
        #     for i in range(4):
        #         base = int(i/4*self.N_models)
        #         wreg[base+int(0/8*self.N_models):base+int(1/8*self.N_models)] = 1e-5
        #         wreg[base+int(1/8*self.N_models):base+int(2/8*self.N_models)] = 1e-3
        #     self.wreg = tf.constant(wreg, dtype=tf.float32)

        # elif task in ['multitask']:

        self.ureg = 1e-5
        self.wreg = 1e-5        


    def construct_ringinput(self,modality,angle):

        preferred = np.linspace(0,2*np.pi,32,endpoint=False)
        difference = np.mod(preferred-angle,2*np.pi)
        difference = np.minimum(difference,2*np.pi-difference)
        singlering = 0.8*np.exp(-0.5*(8*difference/np.pi)**2) + 0.05

        if modality == 0:
            ringinput = np.zeros([32,1])
            ringinput[:,0] = singlering
        if modality == 1:
            ringinput = np.zeros([64,1])
            ringinput[:32,0] = singlering - 0.05
        elif modality == 2:
            ringinput = np.zeros([64,1])
            ringinput[32:,0] = singlering - 0.05
        
        return ringinput

    @tf.function
    def construct_ringangle(self,singlering):
        preferred = tf.cast(np.linspace(0,2*np.pi,32,endpoint=False),dtype=tf.float32)
        popsin = tf.reduce_sum(singlering*tf.math.sin(preferred),axis=-1)/(tf.reduce_sum(singlering,axis=-1)+1e-5)
        popcos = tf.reduce_sum(singlering*tf.math.cos(preferred),axis=-1)/(tf.reduce_sum(singlering,axis=-1)+1e-5)
        return tf.math.mod(tf.math.atan2(popsin+1e-5,popcos+1e-5),2*np.pi)

    def construct_trialA(self,task,trials):

        input0 = np.zeros([trials,self.N_models,self.IO_inp,1])
        input1 = np.zeros([trials,self.N_models,self.IO_inp,1])
        input2 = np.zeros([trials,self.N_models,self.IO_inp,1])
        input3 = np.zeros([trials,self.N_models,self.IO_inp,1])
        targetfix = np.zeros([trials,self.N_models,self.IO_out,1])
        targetout = np.zeros([trials,self.N_models,self.IO_out,1])

        taskno = self.tasks.index(task)+65
        input0[:,:,taskno] = 1
        input1[:,:,taskno] = 1  
        input2[:,:,taskno] = 1  
        input3[:,:,taskno] = 1 
        input1[:,:,0] = 1   
        input2[:,:,0] = 1 
        targetfix[:,:,0] = 0.85
        targetfix[:,:,1:33] = 0.05
        targetout[:,:,0] = 0.05

        if task in ['go','anti']:
         
            for trial in range(trials):
                modality = np.random.choice([1,2])
                angle = np.random.random()*2*np.pi
                input2[trial,:,1:65] = self.construct_ringinput(modality,angle)
                if task == 'go':
                    targetout[trial,:,1:33] = self.construct_ringinput(0,angle)
                elif task == 'anti':
                    targetout[trial,:,1:33] = self.construct_ringinput(0,np.mod(np.pi+angle,2*np.pi))

        if task in ['rtgo','rtanti']:

            input3[:,:,0] = 1   
            for trial in range(trials):
                modality = np.random.choice([1,2])
                angle = np.random.random()*2*np.pi
                input3[trial,:,1:65] = self.construct_ringinput(modality,angle)
                if task == 'rtgo':
                    targetout[trial,:,1:33] = self.construct_ringinput(0,angle)
                elif task == 'rtanti':
                    targetout[trial,:,1:33] = self.construct_ringinput(0,np.mod(np.pi+angle,2*np.pi))

        if task in ['dm1','dm2','ctxdm1','ctxdm2']:

            modality = int(task[-1])
            offmodality = 3-modality
            for trial in range(trials):

                angle1 = np.random.random()*2*np.pi
                angle2 = np.mod(angle1 + np.random.random()*np.pi + np.pi/2 ,2*np.pi)
                coher = np.random.uniform(low=0.8,high=1.2)
                delta = np.random.uniform(low=0.1,high=0.2)
                if np.random.random()<0.5:
                    angle = angle1
                    coher1 = coher + delta
                    coher2 = coher - delta
                else:
                    angle = angle2
                    coher1 = coher - delta
                    coher2 = coher + delta                    
                input2[trial,:,1:65] += coher1*self.construct_ringinput(modality,angle1)
                input2[trial,:,1:65] += coher2*self.construct_ringinput(modality,angle2)

                if 'ctx' in task:

                    offcoher = np.random.uniform(low=0.8,high=1.2)
                    offdelta = np.random.uniform(low=0.1,high=0.2)
                    if np.random.random()<0.5:
                        offcoher1 = offcoher + offdelta
                        offcoher2 = offcoher - offdelta
                    else:
                        offcoher1 = offcoher - offdelta
                        offcoher2 = offcoher + offdelta                   
                    input2[trial,:,1:65] += offcoher1*self.construct_ringinput(offmodality,angle1)
                    input2[trial,:,1:65] += offcoher2*self.construct_ringinput(offmodality,angle2)

                targetout[trial,:,1:33] = self.construct_ringinput(0,angle)

        if task in ['multsendm']:

            for trial in range(trials):

                modality = np.random.choice([1,2])
                offmodality = 3-modality

                angle1 = np.random.random()*2*np.pi
                angle2 = np.mod(angle1 + np.random.random()*np.pi + np.pi/2 ,2*np.pi)
                coher = np.random.uniform(low=0.8,high=1.2)
                delta = np.random.uniform(low=0.1,high=0.2)

                coher1 = coher + delta
                coher2 = coher - delta
                coeff = np.random.uniform(low=1.2,high=1.4)

                input2[trial,:,1:65] += coeff*coher1*self.construct_ringinput(modality,angle1)
                input2[trial,:,1:65] += coeff*coher2*self.construct_ringinput(modality,angle2)
                input2[trial,:,1:65] += (2-coeff)*coher1*self.construct_ringinput(offmodality,angle1)
                input2[trial,:,1:65] += (2-coeff)*coher2*self.construct_ringinput(offmodality,angle2)

                targetout[trial,:,1:33] = self.construct_ringinput(0,angle1) 

        trialA = [input0,input1,input2,input3,targetfix,targetout]

        return [tf.constant(i,dtype=tf.float32) for i in trialA]

    def construct_trialB(self,task,trials):

        input0 = np.zeros([trials,self.N_models,self.IO_inp,1])
        input1 = np.zeros([trials,self.N_models,self.IO_inp,1])
        input2 = np.zeros([trials,self.N_models,self.IO_inp,1])
        input3 = np.zeros([trials,self.N_models,self.IO_inp,1])
        targetfix = np.zeros([trials,self.N_models,self.IO_out,1])
        targetout = np.zeros([trials,self.N_models,self.IO_out,1])

        taskno = self.tasks.index(task)+65
        input0[:,:,taskno] = 1
        input1[:,:,taskno] = 1  # start and fixation
        input2[:,:,taskno] = 1  # stimulus
        input3[:,:,taskno] = 1  # response
        input1[:,:,0] = 1   
        input2[:,:,0] = 1 
        targetfix[:,:,0] = 0.85
        targetfix[:,:,1:33] = 0.05
        targetout[:,:,0] = 0.05

        if task in ['dlygo','dlyanti']:
         
            for trial in range(trials):
                modality = np.random.choice([1,2])
                angle = np.random.random()*2*np.pi
                input2[trial,:,1:65] = self.construct_ringinput(modality,angle)
                if task == 'dlygo':
                    targetout[trial,:,1:33] = self.construct_ringinput(0,angle)
                elif task == 'dlyanti':
                    targetout[trial,:,1:33] = self.construct_ringinput(0,np.mod(np.pi+angle,2*np.pi))
                
        trialB = [input0,input1,input2,input3,targetfix,targetout]

        return [tf.constant(i,dtype=tf.float32) for i in trialB]

    def construct_trialC(self,task,trials):

        input0 = np.zeros([trials,self.N_models,self.IO_inp,1])
        input1 = np.zeros([trials,self.N_models,self.IO_inp,1]) # start and delay
        input2 = np.zeros([trials,self.N_models,self.IO_inp,1]) # stimulus 1
        input3 = np.zeros([trials,self.N_models,self.IO_inp,1]) # stimulus 2
        input4 = np.zeros([trials,self.N_models,self.IO_inp,1]) # response
        targetfix = np.zeros([trials,self.N_models,self.IO_out,1])
        targetout = np.zeros([trials,self.N_models,self.IO_out,1])

        taskno = self.tasks.index(task)+65
        input0[:,:,taskno] = 1
        input1[:,:,taskno] = 1  
        input2[:,:,taskno] = 1  
        input3[:,:,taskno] = 1  
        input4[:,:,taskno] = 1  
        input1[:,:,0] = 1   
        input2[:,:,0] = 1 
        input3[:,:,0] = 1 
        targetfix[:,:,0] = 0.85
        targetfix[:,:,1:33] = 0.05
        targetout[:,:,0] = 0.05

        if task in ['dlydm1','dlydm2','ctxdlydm1','ctxdlydm2']:

            modality = int(task[-1])
            offmodality = 3-modality
            for trial in range(trials):

                angle1 = np.random.random()*2*np.pi
                angle2 = np.mod(angle1 + np.random.random()*np.pi + np.pi/2 ,2*np.pi)
                coher = np.random.uniform(low=0.8,high=1.2)
                delta = np.random.uniform(low=0.1,high=0.2)
                if np.random.random()<0.5:
                    angle = angle1
                    coher1 = coher + delta
                    coher2 = coher - delta
                else:
                    angle = angle2
                    coher1 = coher - delta
                    coher2 = coher + delta                    
                input2[trial,:,1:65] += coher1*self.construct_ringinput(modality,angle1)
                input3[trial,:,1:65] += coher2*self.construct_ringinput(modality,angle2)

                if 'ctx' in task:

                    offcoher = np.random.uniform(low=0.8,high=1.2)
                    offdelta = np.random.uniform(low=0.1,high=0.2)
                    if np.random.random()<0.5:
                        offcoher1 = offcoher + offdelta
                        offcoher2 = offcoher - offdelta
                    else:
                        offcoher1 = offcoher - offdelta
                        offcoher2 = offcoher + offdelta            
                    input2[trial,:,1:65] += offcoher1*self.construct_ringinput(offmodality,angle1)
                    input3[trial,:,1:65] += offcoher2*self.construct_ringinput(offmodality,angle2)

                targetout[trial,:,1:33] = self.construct_ringinput(0,angle)

        if task in ['multsendlydm']:

            for trial in range(trials):

                modality = np.random.choice([1,2])
                offmodality = 3-modality
            
                angle1 = np.random.random()*2*np.pi
                angle2 = np.mod(angle1 + np.random.random()*np.pi + np.pi/2 ,2*np.pi)
                coher = np.random.uniform(low=0.8,high=1.2)
                delta = np.random.uniform(low=0.1,high=0.2)

                coher1 = coher + delta
                coher2 = coher - delta
                coeff = np.random.uniform(low=1.2,high=1.4)

                if np.random.random()<0.5:
                    input2[trial,:,1:65] += coeff*coher1*self.construct_ringinput(modality,angle1)
                    input3[trial,:,1:65] += coeff*coher2*self.construct_ringinput(modality,angle2)
                    input2[trial,:,1:65] += (2-coeff)*coher1*self.construct_ringinput(offmodality,angle1)
                    input3[trial,:,1:65] += (2-coeff)*coher2*self.construct_ringinput(offmodality,angle2)
                else:
                    input3[trial,:,1:65] += coeff*coher1*self.construct_ringinput(modality,angle1)
                    input2[trial,:,1:65] += coeff*coher2*self.construct_ringinput(modality,angle2)
                    input3[trial,:,1:65] += (2-coeff)*coher1*self.construct_ringinput(offmodality,angle1)
                    input2[trial,:,1:65] += (2-coeff)*coher2*self.construct_ringinput(offmodality,angle2)                    

                targetout[trial,:,1:33] = self.construct_ringinput(0,angle1)
        
        if task in ['dms','dnms']:

            for trial in range(trials):

                modality1 = np.random.choice([1,2])
                modality2 = np.random.choice([1,2])

                angle1 = np.random.random()*2*np.pi

                if np.random.random()<0.5:
                    angle2 = np.mod(angle1 + np.random.random()*np.pi/9 - np.pi/18 ,2*np.pi)
                    targetout[trial,:,1:33] = self.construct_ringinput(0,angle2)
                else:
                    angle2 = np.mod(angle1 + np.random.random()*np.pi + np.pi/2 ,2*np.pi)
                    targetout[trial,:,0] = 0.85

                input2[trial,:,1:65] += self.construct_ringinput(modality1,angle1)
                input3[trial,:,1:65] += self.construct_ringinput(modality2,angle2)

        if task in ['dmc','dnmc']:

            for trial in range(trials):

                modality1 = np.random.choice([1,2])
                modality2 = np.random.choice([1,2])

                if np.random.random()<0.5: #match
                    if np.random.random()<0.5: #first half
                        angle1 = np.random.random()*np.pi/2
                        angle2 = np.random.random()*np.pi/2
                        targetout[trial,:,1:33] = self.construct_ringinput(0,angle2)
                    else: #second half
                        angle1 = np.random.random()*np.pi/2 + np.pi
                        angle2 = np.random.random()*np.pi/2 + np.pi
                        targetout[trial,:,1:33] = self.construct_ringinput(0,angle2)
                else:
                    if np.random.random()<0.5: #first stim first half
                        angle1 = np.random.random()*np.pi/2
                        angle2 = np.random.random()*np.pi/2 + np.pi
                        targetout[trial,:,0] = 0.85
                    else: #first stim second half
                        angle1 = np.random.random()*np.pi/2 + np.pi + np.pi
                        angle2 = np.random.random()*np.pi/2 + np.pi    
                        targetout[trial,:,0] = 0.85                        

                input2[trial,:,1:65] += self.construct_ringinput(modality1,angle1)
                input3[trial,:,1:65] += self.construct_ringinput(modality2,angle2)

        trialC = [input0,input1,input2,input3,input4,targetfix,targetout]

        return [tf.constant(i,dtype=tf.float32) for i in trialC]

    @tf.function
    def trainA(self,trialA,constants,activation,dale,skip,ratio):

        input0    = trialA[0]
        input1    = trialA[1]
        input2    = trialA[2]
        input3    = trialA[3]
        targetfix = trialA[4]
        targetout = trialA[5]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        with tf.GradientTape() as tape:

            if dale:
                Wtop = self.Wraw[:int(self.N_models/2)]
                Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
                W = tf.concat([Wtop,Wbot],axis=0)
            else:
                W = self.Wraw

            input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
            input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
            input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))

            if self.T_jitter>0:
                timesteps1 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
            else:
                timesteps1 = self.T_short
            u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps1,activation)
            if self.T_jitter>0:
                timesteps2 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
            else:
                timesteps2 = self.T_short
            u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,skip,ratio,timesteps2,activation)
            timesteps3 = self.T_test
            u,eta,uall3 = compute_uall(u,eta,W,self.bias,input3,constants,skip,ratio,timesteps3,activation)

            uall = tf.concat([uall1,uall2,uall3],axis=0) #[time,trials,models,units,1]
            readout = tf.math.sigmoid(self.Wout@uall[:,:,:,self.N_input:self.N_input+self.N_output]) #[time,trials,models,units,1]

            cost = tf.zeros([self.N_models])
            cost += tf.reduce_mean((readout[self.T_wait:-timesteps3] - targetfix)**2,axis=(0,1,3,4)) 
            cost += tf.reduce_mean((readout[-timesteps3+self.T_wait:] - targetout)**2,axis=(0,1,3,4)) * 1
            cost += tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4)) * self.ureg
            cost += tf.reduce_mean(tf.abs(self.Wraw),axis=(1,2)) * self.wreg

            grads = tape.gradient(cost, self.tv)
        
        self.opt.apply_gradients(zip(grads, self.tv))

        return cost

    @tf.function
    def trainB(self,trialB,constants,activation,dale,skip,ratio):

        input0    = trialB[0]
        input1    = trialB[1]
        input2    = trialB[2]
        input3    = trialB[3]
        targetfix = trialB[4]
        targetout = trialB[5]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        with tf.GradientTape() as tape:

            if dale:
                Wtop = self.Wraw[:int(self.N_models/2)]
                Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
                W = tf.concat([Wtop,Wbot],axis=0)
            else:
                W = self.Wraw

            input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
            input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
            input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))

            if self.T_jitter>0:
                timesteps1 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
            else:
                timesteps1 = self.T_short
            u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps1,activation)
            if self.T_jitter>0:
                timesteps2 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
            else:
                timesteps2 = self.T_short
            u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,skip,ratio,timesteps2,activation)
            if self.T_jitter>0:
                timesteps3 = tf.random.uniform(shape=(),minval=self.T_long-self.T_jitter, maxval=self.T_long+self.T_jitter, dtype=tf.int32)
            else:
                timesteps3 = self.T_long
            u,eta,uall3 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps3,activation)
            timesteps4 = self.T_test
            u,eta,uall4 = compute_uall(u,eta,W,self.bias,input3,constants,skip,ratio,timesteps4,activation)

            uall = tf.concat([uall1,uall2,uall3,uall4],axis=0) #[time,trials,models,units,1]
            readout = tf.math.sigmoid(self.Wout@uall[:,:,:,self.N_input:self.N_input+self.N_output]) #[time,trials,models,units,1]

            cost = tf.zeros([self.N_models])
            cost += tf.reduce_mean((readout[self.T_wait:-timesteps4] - targetfix)**2,axis=(0,1,3,4)) 
            cost += tf.reduce_mean((readout[-timesteps4+self.T_wait:] - targetout)**2,axis=(0,1,3,4)) * 1
            cost += tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4)) * self.ureg
            cost += tf.reduce_mean(tf.abs(self.Wraw),axis=(1,2)) * self.wreg

            grads = tape.gradient(cost, self.tv)
        
        self.opt.apply_gradients(zip(grads, self.tv))

        return cost

    @tf.function
    def trainC(self,trialC,constants,activation,dale,skip,ratio):

        input0    = trialC[0]
        input1    = trialC[1]
        input2    = trialC[2]
        input3    = trialC[3]
        input4    = trialC[4]
        targetfix = trialC[5]
        targetout = trialC[6]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        with tf.GradientTape() as tape:

            if dale:
                Wtop = self.Wraw[:int(self.N_models/2)]
                Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
                W = tf.concat([Wtop,Wbot],axis=0)
            else:
                W = self.Wraw

            input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
            input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
            input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
            input4 = tf.pad(self.Winp@input4,((0,0),(0,0),(0,self.N_pad),(0,0)))

            if self.T_jitter>0:
                timesteps1 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
            else:
                timesteps1 = self.T_short
            u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps1,activation)
            if self.T_jitter>0:
                timesteps2 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
            else:
                timesteps2 = self.T_short
            u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,skip,ratio,timesteps2,activation)
            if self.T_jitter>0:
                timesteps3 = tf.random.uniform(shape=(),minval=self.T_long-self.T_jitter, maxval=self.T_long+self.T_jitter, dtype=tf.int32)
            else:
                timesteps3 = self.T_long
            u,eta,uall3 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps3,activation)
            if self.T_jitter>0:
                timesteps4 = tf.random.uniform(shape=(),minval=self.T_short-self.T_jitter, maxval=self.T_short+self.T_jitter, dtype=tf.int32)
            else:
                timesteps4 = self.T_short
            u,eta,uall4 = compute_uall(u,eta,W,self.bias,input3,constants,skip,ratio,timesteps4,activation)
            timesteps5 = self.T_test
            u,eta,uall5 = compute_uall(u,eta,W,self.bias,input4,constants,skip,ratio,timesteps5,activation)

            uall = tf.concat([uall1,uall2,uall3,uall4,uall5],axis=0) #[time,trials,models,units,1]
            readout = tf.math.sigmoid(self.Wout@uall[:,:,:,self.N_input:self.N_input+self.N_output]) #[time,trials,models,units,1]

            cost = tf.zeros([self.N_models])
            cost += tf.reduce_mean((readout[self.T_wait:-timesteps5] - targetfix)**2,axis=(0,1,3,4)) 
            cost += tf.reduce_mean((readout[-timesteps5+self.T_wait:] - targetout)**2,axis=(0,1,3,4)) * 1
            cost += tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4)) * self.ureg
            cost += tf.reduce_mean(tf.abs(self.Wraw),axis=(1,2)) * self.wreg

            grads = tape.gradient(cost, self.tv)
        
        self.opt.apply_gradients(zip(grads, self.tv))

        return cost

    def evalA(self,trialA,constants,activation,dale):

        input0    = trialA[0]
        input1    = trialA[1]
        input2    = trialA[2]
        input3    = trialA[3]
        targetfix = trialA[4]
        targetout = trialA[5]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,0.,1.,self.T_long,activation)

        timesteps1 = self.T_short
        u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,0.,1.,timesteps1,activation)
        timesteps2 = self.T_short
        u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,0.,1.,timesteps2,activation)
        timesteps3 = self.T_test
        u,eta,uall3 = compute_uall(u,eta,W,self.bias,input3,constants,0.,1.,timesteps3,activation)

        uall = tf.concat([uall1,uall2,uall3],axis=0) #[time,trials,models,units,1]
        uinf = tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4))
        readout = tf.math.sigmoid(self.Wout@uall[:,:,:,self.N_input:self.N_input+self.N_output]) #[time,trials,models,units,1]
        
        comparefix1 = tf.cast(readout[self.T_wait:-timesteps3,:,:,0,0] > 0.5,dtype=tf.float32)
        comparefix2 = tf.cast(readout[-timesteps3+self.T_wait:,:,:,0,0] < 0.5,dtype=tf.float32)
        modelangles = self.construct_ringangle(readout[-timesteps3+self.T_wait:,:,:,1:33,0])
        targetangles = self.construct_ringangle(targetout[:,:,1:33,0])
        deltaangles = tf.math.mod(modelangles - targetangles, 2*np.pi)
        deltaangles = tf.math.minimum(deltaangles, 2*np.pi-deltaangles)
        compareangles = tf.cast(deltaangles < 0.2*np.pi,dtype=tf.float32)
        performancefix1 = tf.reduce_mean(comparefix1,axis=(0,1))
        performancefix2 = tf.reduce_mean(comparefix2,axis=(0,1))
        performanceres = tf.reduce_mean(compareangles,axis=(0,1))

        return performanceres,performancefix1*(2/3)+performancefix2*performanceres*(1/3),uinf,readout

    def evalB(self,trialB,constants,activation,dale):

        input0    = trialB[0]
        input1    = trialB[1]
        input2    = trialB[2]
        input3    = trialB[3]
        targetfix = trialB[4]
        targetout = trialB[5]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,0.,1.,self.T_long,activation)

        timesteps1 = self.T_short
        u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,0.,1.,timesteps1,activation)
        timesteps2 = self.T_short
        u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,0.,1.,timesteps2,activation)
        timesteps3 = self.T_long
        u,eta,uall3 = compute_uall(u,eta,W,self.bias,input1,constants,0.,1.,timesteps3,activation)
        timesteps4 = self.T_test
        u,eta,uall4 = compute_uall(u,eta,W,self.bias,input3,constants,0.,1.,timesteps4,activation)

        uall = tf.concat([uall1,uall2,uall3,uall4],axis=0) #[time,trials,models,units,1]
        uinf = tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4))
        readout = tf.math.sigmoid(self.Wout@uall[:,:,:,self.N_input:self.N_input+self.N_output]) #[time,trials,models,units,1]
        
        comparefix1 = tf.cast(readout[self.T_wait:-timesteps4,:,:,0,0] > 0.5,dtype=tf.float32)
        comparefix2 = tf.cast(readout[-timesteps4+self.T_wait:,:,:,0,0] < 0.5,dtype=tf.float32)
        modelangles = self.construct_ringangle(readout[-timesteps4+self.T_wait:,:,:,1:33,0])
        targetangles = self.construct_ringangle(targetout[:,:,1:33,0])
        deltaangles = tf.math.mod(modelangles - targetangles, 2*np.pi)
        deltaangles = tf.math.minimum(deltaangles, 2*np.pi-deltaangles)
        compareangles = tf.cast(deltaangles < 0.2*np.pi,dtype=tf.float32)
        performancefix1 = tf.reduce_mean(comparefix1,axis=(0,1))
        performancefix2 = tf.reduce_mean(comparefix2,axis=(0,1))
        performanceres = tf.reduce_mean(compareangles,axis=(0,1))

        return performanceres,performancefix1*(4/5)+performancefix2*performanceres*(1/5),uinf,readout

    def evalC(self,trialC,constants,activation,dale):

        input0    = trialC[0]
        input1    = trialC[1]
        input2    = trialC[2]
        input3    = trialC[3]
        input4    = trialC[4]
        targetfix = trialC[5]
        targetout = trialC[6]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input4 = tf.pad(self.Winp@input4,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,0.,1.,self.T_long,activation)

        timesteps1 = self.T_short
        u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,0.,1.,timesteps1,activation)
        timesteps2 = self.T_short
        u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,0.,1.,timesteps2,activation)
        timesteps3 = self.T_long
        u,eta,uall3 = compute_uall(u,eta,W,self.bias,input1,constants,0.,1.,timesteps3,activation)
        timesteps4 = self.T_short
        u,eta,uall4 = compute_uall(u,eta,W,self.bias,input3,constants,0.,1.,timesteps4,activation)
        timesteps5 = self.T_test
        u,eta,uall5 = compute_uall(u,eta,W,self.bias,input4,constants,0.,1.,timesteps5,activation)

        uall = tf.concat([uall1,uall2,uall3,uall4,uall5],axis=0) #[time,trials,models,units,1]
        uinf = tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4))
        readout = tf.math.sigmoid(self.Wout@uall[:,:,:,self.N_input:self.N_input+self.N_output]) #[time,trials,models,units,1]
        
        if tf.reduce_all(targetout[:,:,0,0] < 0.5):
            comparefix1 = tf.cast(readout[self.T_wait:-timesteps5,:,:,0,0] > 0.5,dtype=tf.float32)
            comparefix2 = tf.cast(readout[-timesteps5+self.T_wait:,:,:,0,0] < 0.5,dtype=tf.float32)
            modelangles = self.construct_ringangle(readout[-timesteps5+self.T_wait:,:,:,1:33,0])
            targetangles = self.construct_ringangle(targetout[:,:,1:33,0])
            deltaangles = tf.math.mod(modelangles - targetangles, 2*np.pi)
            deltaangles = tf.math.minimum(deltaangles, 2*np.pi-deltaangles)
            compareangles = tf.cast(deltaangles < 0.2*np.pi,dtype=tf.float32)
            performancefix1 = tf.reduce_mean(comparefix1,axis=(0,1))
            performancefix2 = tf.reduce_mean(comparefix2,axis=(0,1))
            performanceres = tf.reduce_mean(compareangles,axis=(0,1))

            return performanceres,performancefix1*(5/6)+performancefix2*performanceres*(1/6),uinf,readout
        
        else:
            comparefix1 = tf.cast(readout[self.T_wait:-timesteps5,:,:,0,0] > 0.5,dtype=tf.float32)
            performancefix1 = tf.reduce_mean(comparefix1,axis=(0,1))

            timebroadcast = tf.ones([timesteps5-self.T_wait,1,1],dtype=tf.float32)
            targetfix2 = timebroadcast*tf.cast(targetout[:,:,0,0] > 0.5,dtype=tf.float32)[tf.newaxis]
            frac_targetfix2 = tf.reduce_mean(targetfix2,axis=(0,1))
            readoutfix2 = tf.cast(readout[-timesteps5+self.T_wait:,:,:,0,0] > 0.5,dtype=tf.float32)
            comparefix2 = tf.reduce_mean(readoutfix2*targetfix2,axis=(0,1))
            comparenofix2 = tf.reduce_mean((1-readoutfix2)*(1-targetfix2),axis=(0,1))
            performancefix2 = comparefix2 + comparenofix2

            modelangles = self.construct_ringangle(readout[-timesteps5+self.T_wait:,:,:,1:33,0])
            targetangles = self.construct_ringangle(targetout[:,:,1:33,0])
            deltaangles = tf.math.mod(modelangles - targetangles, 2*np.pi)
            deltaangles = tf.math.minimum(deltaangles, 2*np.pi-deltaangles)
            compareangles = tf.cast(deltaangles < 0.2*np.pi,dtype=tf.float32)
            performanceres = tf.reduce_mean(compareangles*(1 - targetfix2),axis=(0,1))/(1-frac_targetfix2)

            return performanceres,performancefix1*(5/6)+(comparefix2 + comparenofix2*performanceres)*(1/6),uinf,readout

    def construct_trial(self,trials):
        if self.task in self.tasksA:
            return self.construct_trialA(self.task,trials)
        elif self.task in self.tasksB:
            return self.construct_trialB(self.task,trials)
        elif self.task in self.tasksC:
            return self.construct_trialC(self.task,trials)
        elif self.task in ['multitask']:

            Ainput0 = np.zeros([9*trials,self.N_models,self.IO_inp,1])
            Ainput1 = np.zeros([9*trials,self.N_models,self.IO_inp,1])
            Ainput2 = np.zeros([9*trials,self.N_models,self.IO_inp,1])
            Ainput3 = np.zeros([9*trials,self.N_models,self.IO_inp,1])
            Atargetfix = np.zeros([9*trials,self.N_models,self.IO_out,1])
            Atargetout = np.zeros([9*trials,self.N_models,self.IO_out,1])
            
            for ta in range(9):
                temptrial = self.construct_trialA(self.tasksA[ta],trials)
                Ainput0   [ta*trials:(ta+1)*trials] = temptrial[0]
                Ainput1   [ta*trials:(ta+1)*trials] = temptrial[1]
                Ainput2   [ta*trials:(ta+1)*trials] = temptrial[2]
                Ainput3   [ta*trials:(ta+1)*trials] = temptrial[3]
                Atargetfix[ta*trials:(ta+1)*trials] = temptrial[4]
                Atargetout[ta*trials:(ta+1)*trials] = temptrial[5]
            trialA = [Ainput0,Ainput1,Ainput2,Ainput3,Atargetfix,Atargetout]
            
            Binput0 = np.zeros([2*trials,self.N_models,self.IO_inp,1])
            Binput1 = np.zeros([2*trials,self.N_models,self.IO_inp,1])
            Binput2 = np.zeros([2*trials,self.N_models,self.IO_inp,1])
            Binput3 = np.zeros([2*trials,self.N_models,self.IO_inp,1])
            Btargetfix = np.zeros([2*trials,self.N_models,self.IO_out,1])
            Btargetout = np.zeros([2*trials,self.N_models,self.IO_out,1])
            
            for ta in range(2):
                temptrial = self.construct_trialB(self.tasksB[ta],trials)
                Binput0   [ta*trials:(ta+1)*trials] = temptrial[0]
                Binput1   [ta*trials:(ta+1)*trials] = temptrial[1]
                Binput2   [ta*trials:(ta+1)*trials] = temptrial[2]
                Binput3   [ta*trials:(ta+1)*trials] = temptrial[3]
                Btargetfix[ta*trials:(ta+1)*trials] = temptrial[4]
                Btargetout[ta*trials:(ta+1)*trials] = temptrial[5]
            trialB = [Binput0,Binput1,Binput2,Binput3,Btargetfix,Btargetout]

            Cinput0 = np.zeros([9*trials,self.N_models,self.IO_inp,1])
            Cinput1 = np.zeros([9*trials,self.N_models,self.IO_inp,1])
            Cinput2 = np.zeros([9*trials,self.N_models,self.IO_inp,1])
            Cinput3 = np.zeros([9*trials,self.N_models,self.IO_inp,1])
            Cinput4 = np.zeros([9*trials,self.N_models,self.IO_inp,1])
            Ctargetfix = np.zeros([9*trials,self.N_models,self.IO_out,1])
            Ctargetout = np.zeros([9*trials,self.N_models,self.IO_out,1])
            
            for ta in range(9):
                temptrial = self.construct_trialC(self.tasksC[ta],trials)
                Cinput0   [ta*trials:(ta+1)*trials] = temptrial[0]
                Cinput1   [ta*trials:(ta+1)*trials] = temptrial[1]
                Cinput2   [ta*trials:(ta+1)*trials] = temptrial[2]
                Cinput3   [ta*trials:(ta+1)*trials] = temptrial[3]
                Cinput4   [ta*trials:(ta+1)*trials] = temptrial[4]
                Ctargetfix[ta*trials:(ta+1)*trials] = temptrial[5]
                Ctargetout[ta*trials:(ta+1)*trials] = temptrial[6]
            trialC = [Cinput0,Cinput1,Cinput2,Cinput3,Cinput4,Ctargetfix,Ctargetout]
        
            return [[tf.constant(i,dtype=tf.float32) for i in trialA],[tf.constant(i,dtype=tf.float32) for i in trialB],[tf.constant(i,dtype=tf.float32) for i in trialC]]

    def eval(self,constants,activation,dale):
        trial = self.construct_trial(trials=self.evaltrials)
        if self.task in self.tasksA:
            return self.evalA(trial,constants,activation,dale)
        elif self.task in self.tasksB:
            return self.evalB(trial,constants,activation,dale)
        elif self.task in self.tasksC:
            return self.evalC(trial,constants,activation,dale)
        elif self.task in ['multitask']:

            performance = np.zeros([20,self.N_models])
            performanceres = np.zeros([20,self.N_models])
            uinf = np.zeros([20,self.N_models])

            for ta in range(9):
                performanceresA,performanceA,uinfA,_ = self.evalA([i[ta*self.evaltrials:(ta+1)*self.evaltrials] for i in trial[0]],constants,activation,dale)
                performance[ta] = performanceA.numpy()
                performanceres[ta] = performanceresA.numpy()
                uinf[ta] = uinfA.numpy()
 
            for ta in range(2):
                performanceresB,performanceB,uinfB,_ = self.evalB([i[ta*self.evaltrials:(ta+1)*self.evaltrials] for i in trial[1]],constants,activation,dale)
                performance[9+ta] = performanceB.numpy()
                performanceres[9+ta] = performanceresB.numpy()
                uinf[9+ta] = uinfB.numpy()

            for ta in range(9):
                performanceresC,performanceC,uinfC,_ = self.evalC([i[ta*self.evaltrials:(ta+1)*self.evaltrials] for i in trial[2]],constants,activation,dale)
                performance[11+ta] = performanceC.numpy()
                performanceres[11+ta] = performanceresC.numpy()
                uinf[11+ta] = uinfC.numpy()

            return performanceres,performance,uinf

    def train(self,trial,constants,activation,dale,skip,ratio):
        if self.task in self.tasksA:
            return self.trainA(trial,constants,activation,dale,skip,ratio)
        elif self.task in self.tasksB:
            return self.trainB(trial,constants,activation,dale,skip,ratio)
        elif self.task in self.tasksC:
            return self.trainC(trial,constants,activation,dale,skip,ratio)
        elif self.task in ['multitask']:
            costA = self.trainA(trial[0],constants,activation,dale,skip,ratio)
            costB = self.trainB(trial[1],constants,activation,dale,skip,ratio)
            costC = self.trainC(trial[2],constants,activation,dale,skip,ratio)
            return costA*0.45 + costB*0.1 + costC*0.45

    def trainmodel(self,constants,iterations,activation,dale,skip,ratio):
        costs = []
       
        for iter in tf.range(iterations):
            trial = self.construct_trial(trials=self.trials)
            cost = self.train(trial,constants,activation,dale,skip,ratio)
            costs.append(cost.numpy())
    
        return np.array(costs)
    
    def inputoutput(self,constants,activation,dale):
        
        trial = self.construct_trial(trials=self.evaltrials)
        if self.task in self.tasksA:
            _,_,_,readout = self.evalA(trial,constants,activation,dale)
            return trial,readout
        elif self.task in self.tasksB:
            _,_,_,readout = self.evalB(trial,constants,activation,dale)
            return trial,readout
        elif self.task in self.tasksC:
            _,_,_,readout = self.evalC(trial,constants,activation,dale)
            return trial,readout
        elif self.task in ['multitask']:

            readouts = []
            for ta in range(9):
                _,_,_,readout = self.evalA([i[ta*self.evaltrials:(ta+1)*self.evaltrials] for i in trial[0]],constants,activation,dale)
                readouts.append(readout.numpy())
 
            for ta in range(2):
                _,_,_,readout = self.evalB([i[ta*self.evaltrials:(ta+1)*self.evaltrials] for i in trial[1]],constants,activation,dale)
                readouts.append(readout.numpy())

            for ta in range(9):
                _,_,_,readout = self.evalC([i[ta*self.evaltrials:(ta+1)*self.evaltrials] for i in trial[2]],constants,activation,dale)
                readouts.append(readout.numpy())

            return trial,np.array(readouts)


# graph = self.trainC.get_concrete_function(trial,constants,activation,skip,ratio).graph
# tf.compat.v1.profiler.profile(graph,options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())

    def uallA(self,trialA,constants,activation,dale,skip,ratio):

        input0    = trialA[0]
        input1    = trialA[1]
        input2    = trialA[2]
        input3    = trialA[3]
        targetfix = trialA[4]
        targetout = trialA[5]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,uall0 = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        timesteps1 = self.T_short
        u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps1,activation)
        timesteps2 = self.T_short
        u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,skip,ratio,timesteps2,activation)
        timesteps3 = self.T_test
        u,eta,uall3 = compute_uall(u,eta,W,self.bias,input3,constants,skip,ratio,timesteps3,activation)

        uall = tf.concat([uall0[-1:],uall1,uall2,uall3],axis=0) #[time,trials,models,units,1]
        return uall

    def uallB(self,trialB,constants,activation,dale,skip,ratio):

        input0    = trialB[0]
        input1    = trialB[1]
        input2    = trialB[2]
        input3    = trialB[3]
        targetfix = trialB[4]
        targetout = trialB[5]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,uall0 = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        timesteps1 = self.T_short
        u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps1,activation)
        timesteps2 = self.T_short
        u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,skip,ratio,timesteps2,activation)
        timesteps3 = self.T_long
        u,eta,uall3 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps3,activation)
        timesteps4 = self.T_test
        u,eta,uall4 = compute_uall(u,eta,W,self.bias,input3,constants,skip,ratio,timesteps4,activation)

        uall = tf.concat([uall0[-1:],uall1,uall2,uall3,uall4],axis=0) #[time,trials,models,units,1]
        return uall

    def uallC(self,trialC,constants,activation,dale,skip,ratio):

        input0    = trialC[0]
        input1    = trialC[1]
        input2    = trialC[2]
        input3    = trialC[3]
        input4    = trialC[4]
        targetfix = trialC[5]
        targetout = trialC[6]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input4 = tf.pad(self.Winp@input4,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,uall0 = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        timesteps1 = self.T_short
        u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps1,activation)
        timesteps2 = self.T_short
        u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,skip,ratio,timesteps2,activation)
        timesteps3 = self.T_long
        u,eta,uall3 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps3,activation)
        timesteps4 = self.T_short
        u,eta,uall4 = compute_uall(u,eta,W,self.bias,input3,constants,skip,ratio,timesteps4,activation)
        timesteps5 = self.T_test
        u,eta,uall5 = compute_uall(u,eta,W,self.bias,input4,constants,skip,ratio,timesteps5,activation)

        uall = tf.concat([uall0[-1:],uall1,uall2,uall3,uall4,uall5],axis=0) #[time,trials,models,units,1]
        return uall

    def uall(self,constants,activation,dale,trial,skip,ratio):
        if self.task in self.tasksA:
            return self.uallA(trial,constants,activation,dale,skip,ratio)
        elif self.task in self.tasksB:
            return self.uallB(trial,constants,activation,dale,skip,ratio)
        elif self.task in self.tasksC:
            return self.uallC(trial,constants,activation,dale,skip,ratio)

    def evalskipA(self,trialA,constants,activation,dale,skip,ratio):

        input0    = trialA[0]
        input1    = trialA[1]
        input2    = trialA[2]
        input3    = trialA[3]
        targetfix = trialA[4]
        targetout = trialA[5]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        timesteps1 = self.T_short
        u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps1,activation)
        timesteps2 = self.T_short
        u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,skip,ratio,timesteps2,activation)
        timesteps3 = self.T_test
        u,eta,uall3 = compute_uall(u,eta,W,self.bias,input3,constants,skip,ratio,timesteps3,activation)

        uall = tf.concat([uall1,uall2,uall3],axis=0) #[time,trials,models,units,1]
        uinf = tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4))
        readout = tf.math.sigmoid(self.Wout@uall[:,:,:,self.N_input:self.N_input+self.N_output]) #[time,trials,models,units,1]
        
        comparefix1 = tf.cast(readout[self.T_wait:-timesteps3,:,:,0,0] > 0.5,dtype=tf.float32)
        comparefix2 = tf.cast(readout[-timesteps3+self.T_wait:,:,:,0,0] < 0.5,dtype=tf.float32)
        modelangles = self.construct_ringangle(readout[-timesteps3+self.T_wait:,:,:,1:33,0])
        targetangles = self.construct_ringangle(targetout[:,:,1:33,0])
        deltaangles = tf.math.mod(modelangles - targetangles, 2*np.pi)
        deltaangles = tf.math.minimum(deltaangles, 2*np.pi-deltaangles)
        compareangles = tf.cast(deltaangles < 0.2*np.pi,dtype=tf.float32)
        performancefix1 = tf.reduce_mean(comparefix1,axis=(0,1))
        performancefix2 = tf.reduce_mean(comparefix2,axis=(0,1))
        performanceres = tf.reduce_mean(compareangles,axis=(0,1))

        return performanceres,performancefix1*(2/3)+performancefix2*performanceres*(1/3),uinf,readout

    def evalskipB(self,trialB,constants,activation,dale,skip,ratio):

        input0    = trialB[0]
        input1    = trialB[1]
        input2    = trialB[2]
        input3    = trialB[3]
        targetfix = trialB[4]
        targetout = trialB[5]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        timesteps1 = self.T_short
        u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps1,activation)
        timesteps2 = self.T_short
        u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,skip,ratio,timesteps2,activation)
        timesteps3 = self.T_long
        u,eta,uall3 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps3,activation)
        timesteps4 = self.T_test
        u,eta,uall4 = compute_uall(u,eta,W,self.bias,input3,constants,skip,ratio,timesteps4,activation)

        uall = tf.concat([uall1,uall2,uall3,uall4],axis=0) #[time,trials,models,units,1]
        uinf = tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4))
        readout = tf.math.sigmoid(self.Wout@uall[:,:,:,self.N_input:self.N_input+self.N_output]) #[time,trials,models,units,1]
        
        comparefix1 = tf.cast(readout[self.T_wait:-timesteps4,:,:,0,0] > 0.5,dtype=tf.float32)
        comparefix2 = tf.cast(readout[-timesteps4+self.T_wait:,:,:,0,0] < 0.5,dtype=tf.float32)
        modelangles = self.construct_ringangle(readout[-timesteps4+self.T_wait:,:,:,1:33,0])
        targetangles = self.construct_ringangle(targetout[:,:,1:33,0])
        deltaangles = tf.math.mod(modelangles - targetangles, 2*np.pi)
        deltaangles = tf.math.minimum(deltaangles, 2*np.pi-deltaangles)
        compareangles = tf.cast(deltaangles < 0.2*np.pi,dtype=tf.float32)
        performancefix1 = tf.reduce_mean(comparefix1,axis=(0,1))
        performancefix2 = tf.reduce_mean(comparefix2,axis=(0,1))
        performanceres = tf.reduce_mean(compareangles,axis=(0,1))

        return performanceres,performancefix1*(4/5)+performancefix2*performanceres*(1/5),uinf,readout

    def evalskipC(self,trialC,constants,activation,dale,skip,ratio):

        input0    = trialC[0]
        input1    = trialC[1]
        input2    = trialC[2]
        input3    = trialC[3]
        input4    = trialC[4]
        targetfix = trialC[5]
        targetout = trialC[6]

        trials = tf.shape(input0)[0]
        u = tf.zeros([trials,self.N_models,constants['N'],1])
        eta = tf.random.normal(tf.shape(u))
        
        if dale:
            Wtop = self.Wraw[:int(self.N_models/2)]
            Wbot = tf.abs(self.Wraw[int(self.N_models/2):])*constants['mask']
            W = tf.concat([Wtop,Wbot],axis=0)
        else:
            W = self.Wraw

        input0 = tf.pad(self.Winp@input0,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input1 = tf.pad(self.Winp@input1,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input2 = tf.pad(self.Winp@input2,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input3 = tf.pad(self.Winp@input3,((0,0),(0,0),(0,self.N_pad),(0,0)))
        input4 = tf.pad(self.Winp@input4,((0,0),(0,0),(0,self.N_pad),(0,0)))
        u,eta,_ = compute_uall(u,eta,W,self.bias,input0,constants,skip,ratio,self.T_long,activation)

        timesteps1 = self.T_short
        u,eta,uall1 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps1,activation)
        timesteps2 = self.T_short
        u,eta,uall2 = compute_uall(u,eta,W,self.bias,input2,constants,skip,ratio,timesteps2,activation)
        timesteps3 = self.T_long
        u,eta,uall3 = compute_uall(u,eta,W,self.bias,input1,constants,skip,ratio,timesteps3,activation)
        timesteps4 = self.T_short
        u,eta,uall4 = compute_uall(u,eta,W,self.bias,input3,constants,skip,ratio,timesteps4,activation)
        timesteps5 = self.T_test
        u,eta,uall5 = compute_uall(u,eta,W,self.bias,input4,constants,skip,ratio,timesteps5,activation)

        uall = tf.concat([uall1,uall2,uall3,uall4,uall5],axis=0) #[time,trials,models,units,1]
        uinf = tf.reduce_mean(tf.abs(uall),axis=(0,1,3,4))
        readout = tf.math.sigmoid(self.Wout@uall[:,:,:,self.N_input:self.N_input+self.N_output]) #[time,trials,models,units,1]
        
        if tf.reduce_all(targetout[:,:,0,0] < 0.5):
            comparefix1 = tf.cast(readout[self.T_wait:-timesteps5,:,:,0,0] > 0.5,dtype=tf.float32)
            comparefix2 = tf.cast(readout[-timesteps5+self.T_wait:,:,:,0,0] < 0.5,dtype=tf.float32)
            modelangles = self.construct_ringangle(readout[-timesteps5+self.T_wait:,:,:,1:33,0])
            targetangles = self.construct_ringangle(targetout[:,:,1:33,0])
            deltaangles = tf.math.mod(modelangles - targetangles, 2*np.pi)
            deltaangles = tf.math.minimum(deltaangles, 2*np.pi-deltaangles)
            compareangles = tf.cast(deltaangles < 0.2*np.pi,dtype=tf.float32)
            performancefix1 = tf.reduce_mean(comparefix1,axis=(0,1))
            performancefix2 = tf.reduce_mean(comparefix2,axis=(0,1))
            performanceres = tf.reduce_mean(compareangles,axis=(0,1))

            return performanceres,performancefix1*(5/6)+performancefix2*performanceres*(1/6),uinf,readout
        
        else:
            comparefix1 = tf.cast(readout[self.T_wait:-timesteps5,:,:,0,0] > 0.5,dtype=tf.float32)
            performancefix1 = tf.reduce_mean(comparefix1,axis=(0,1))

            timebroadcast = tf.ones([timesteps5-self.T_wait,1,1],dtype=tf.float32)
            targetfix2 = timebroadcast*tf.cast(targetout[:,:,0,0] > 0.5,dtype=tf.float32)[tf.newaxis]
            frac_targetfix2 = tf.reduce_mean(targetfix2,axis=(0,1))
            readoutfix2 = tf.cast(readout[-timesteps5+self.T_wait:,:,:,0,0] > 0.5,dtype=tf.float32)
            comparefix2 = tf.reduce_mean(readoutfix2*targetfix2,axis=(0,1))
            comparenofix2 = tf.reduce_mean((1-readoutfix2)*(1-targetfix2),axis=(0,1))
            performancefix2 = comparefix2 + comparenofix2

            modelangles = self.construct_ringangle(readout[-timesteps5+self.T_wait:,:,:,1:33,0])
            targetangles = self.construct_ringangle(targetout[:,:,1:33,0])
            deltaangles = tf.math.mod(modelangles - targetangles, 2*np.pi)
            deltaangles = tf.math.minimum(deltaangles, 2*np.pi-deltaangles)
            compareangles = tf.cast(deltaangles < 0.2*np.pi,dtype=tf.float32)
            performanceres = tf.reduce_mean(compareangles*(1 - targetfix2),axis=(0,1))/(1-frac_targetfix2)

            return performanceres,performancefix1*(5/6)+(comparefix2 + comparenofix2*performanceres)*(1/6),uinf,readout

    def evalskip(self,constants,activation,dale,skip,ratio):
        trial = self.construct_trial(trials=self.evaltrials)
        if self.task in self.tasksA:
            return self.evalskipA(trial,constants,activation,dale,skip,ratio)
        elif self.task in self.tasksB:
            return self.evalskipB(trial,constants,activation,dale,skip,ratio)
        elif self.task in self.tasksC:
            return self.evalskipC(trial,constants,activation,dale,skip,ratio)