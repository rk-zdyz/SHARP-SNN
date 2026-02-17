import numpy as np
class SynapseLayer:
    def __init__ (self,n_pre,n_post,learning_rate=0.01,tau=20.0):
        self.n_pre=n_pre
        self.n_post=n_post
        self.lr=learning_rate
        self.tau=tau # Time constant for STDP Window
        
        #Initialize weights randomly [0.1,0.5]
        self.weights = np.random.uniform(0.1,0.5,(n_pre,n_post))

        # Track last spike times for STDP
        self.pre_spike_times=np.full(n_pre,-np.inf)
        self.post_spike_times=np.full(n_post,-np.inf)

    def forward(self,pre_spikes):
        '''Computes input to post-neurons: w*x'''
        return np.dot(pre_spikes,self.weights)
    
    def update_stdp(self,pre_spikes,post_spikes,current_time):
        '''Applies Spike-Timing Dependent Plasticity rule'''
        # Update spike times
        pre_indices=np.where(pre_spikes>0)[0]
        post_indices=np.where(post_spikes>0)[0]
        self.pre_spike_times[pre_indices]=current_time
        self.post_spike_times[post_indices]=current_time

        # LTP: Pre spiked BEFORE Post (Casual)
        # Check all pre that spiked recently against current post spikes
        for post_idx in post_indices:
            dt=current_time-self.pre_spike_times
            # Only consider casual events within reasonable window
            valid_mask=(dt>0)&(dt<4*self.tau)
            # Weight change: dw = lr * exp(-dt/tau)
            dw = self.lr * np.exp(-dt[valid_mask] / self.tau)
            self.weights[valid_mask,post_idx]+=dw
            
        # LTD: Post spiked BEFORE Pre (Acasual)
        # Check all post that spiked recently against current pre spikes
        for pre_idx in pre_indices:
            dt=self.post_spike_times-current_time
            dt_vals=self.post_spike_times-current_time
            valid_mask=(dt_vals<0)&(dt_vals>-4*self.tau)

            # Weight change: dw=-lr*exp(dt/tau)
            dw=-self.lr*np.exp(dt_vals[valid_mask]/self.tau)
            self.weights[pre_idx,valid_mask]+=dw

        # Clip weights to prevent explosion
        self.weights=np.clip(self.weights,0.0,1.0)