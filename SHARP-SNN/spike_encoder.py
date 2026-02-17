import numpy as np

class SpikeEncoder:
    def __init__(self,time_steps=50,method='rate'):
        self.time_steps=time_steps
        self.method=method

    def encode(self,data):
        '''Converts continuous data array into spike trains'''
        data=np.clip(data,0.0,1.0)
        encoded_spikes=np.zeros((self.time_steps,len(data)))
        if self.method=='Rate':
            # Rate encoding value = Probability of spike at each step
            for t in range(self.time_steps): # Generate random threshold for each input
                rand_vals=np.random.rand(len(data))
                encoded_spikes[t]=(rand_vals<data).astype(float)
        elif self.method=='Temporal':
            # Temporal encoding value = Time of first spike: Value = 1 - (spike_time / total_time)
            # Higher vallue = Earlier spike
            spike_times=((1.0-data)*(self.time_steps-1)).astype(int)
            for i,t in enumerate(spike_times):
                encoded_spikes[t,i]=1.0
        return encoded_spikes

    def decode(self,spike_train):
        '''Estimates original value from spike train (Rate decoding)'''
        return np.mean(spike_train,axis=0)
