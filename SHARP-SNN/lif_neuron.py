import numpy as np
class LIFNeuron:
    def __init__(self,neuron_id,threshold=1.0,decay=0.9,reset_potential=0.0):
        self.id=neuron_id
        self.threshold=threshold
        self.decay=decay
        self.reset_potential=reset_potential
        self.potential=0.0
        self.spike_history=[]
        self.is_active = True

        self.original_threshold=threshold
    def step(self,weighted_input,current_time):
        if not self.is_active:
            return 0 # Dead neuron

        # Integrate: Decay previous potential and add new input    
        self.potential=((self.potential*self.decay)+weighted_input) 

        # Fire: Check if potential exceeds threshold
        if self.potential>=self.threshold:
            self.potential=self.reset_potential
            self.spike_history.append(current_time)
            return 1 # Spike
        else:
            return 0 # No spike

    def get_spike_rate(self,window=100):
        if not self.spike_history:
            return 0.0
        recent_spikes=[t for t in self.spike_history if t>= (self.spike_history[-1]-window)]
        return len(recent_spikes)/window

    def inject_fault(self,fault_type):
        if fault_type== "Silent":
            self.threshold = 999.0 # Impossible to fire
        elif fault_type== "Hyperactive":
            self.threshold = 0.01 # Fires on noise
        elif fault_type == "Dead":
            self.is_active=False # Neuron is dead

    def repair(self):
        self.threshold=self.original_threshold
        self.potential=0.0
        self.is_active= True