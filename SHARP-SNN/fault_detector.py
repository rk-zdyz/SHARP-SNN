from collections import deque

class FaultType:
    HEALTHY="HEALTHY"
    SILENT="SILENT"
    HYPERACTIVE="HYPERACTIVE"
    DEAD="DEAD"

class FaultDetector:
    def __init__(self,window_size=150): # Increased from 50 to 150 (slower, more stable detection)
        self.window_size=window_size
        self.spike_history={}

    def record_spike(self,neuron_id,spiked):
        if neuron_id not in self.spike_history:
            self.spike_history[neuron_id]=deque(maxlen=self.window_size)
        self.spike_history[neuron_id].append(spiked)

    def clear_history(self, neuron_id):
        if neuron_id in self.spike_history:
            del self.spike_history[neuron_id]

    def detect_fault(self,neuron):
        if not neuron.is_active:
            return FaultType.DEAD
        history = self.spike_history.get(neuron.id,[])
        if len(history)<self.window_size:
            return FaultType.HEALTHY 

        spike_rate=sum(history)/len(history)

        if spike_rate<0.005:
            return FaultType.SILENT
        elif spike_rate>0.2:
            return FaultType.HYPERACTIVE
        else:
            return FaultType.HEALTHY