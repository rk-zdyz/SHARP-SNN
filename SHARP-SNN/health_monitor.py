class HealthMonitor:
    def __init__(self,neuron_ids):
        self.health_scores={nid:1.0 for nid in neuron_ids}
        self.healing_threshold=0.8 # Higher threshold to catch degradation earlier
        # Track healing: {nid: 0.0 to 1.0}
        self.healing_progress = {} 

    def update_health(self,neuron_id,fault_type):
        # If healing, don't degrade further
        if neuron_id in self.healing_progress:
            return

        current_score=self.health_scores[neuron_id]

        if fault_type == "HEALTHY":
            new_score=min(1.0,current_score+0.05)
        else:
            penalty=0.1
            if fault_type=="DEAD": penalty=1.0 # Instant drop to 0
            new_score=max(0.0, current_score-penalty)
        self.health_scores[neuron_id]=new_score

    def needs_healing(self, neuron_id):
        # If already healing, return False for "needs NEW healing setup"
        # But we need a way to check if we should start.
        if neuron_id in self.healing_progress:
            return False
        return self.health_scores[neuron_id]<self.healing_threshold
    
    def is_healing(self, neuron_id):
        return neuron_id in self.healing_progress

    def start_healing(self, neuron_id):
        self.healing_progress[neuron_id] = 0.0
    
    def tick_healing(self, neuron_id):
        if neuron_id not in self.healing_progress:
            return False
        
        self.healing_progress[neuron_id] += 0.05 # 20 steps to heal
        if self.healing_progress[neuron_id] >= 1.0:
            return True # Done
        return False
    
    def complete_healing(self, neuron_id, fault_type=None):
        if neuron_id in self.healing_progress:
            del self.healing_progress[neuron_id]
        
        # Only reset health to 1.0 if the neuron is effectively "cured" and active.
        # For DEAD neurons (which are replaced), we want them to stay at 0.0 health visually.
        # Normalize to upper
        if fault_type and str(fault_type).upper() != "DEAD":
            self.reset_health(neuron_id)

    def reset_health(self,neuron_id):
        self.health_scores[neuron_id]=1.0
        if neuron_id in self.healing_progress:
            del self.healing_progress[neuron_id]