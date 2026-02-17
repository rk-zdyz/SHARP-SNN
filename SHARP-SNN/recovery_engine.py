import numpy as numpy
from fault_detector import FaultType

class RecoveryEngine:
    def __init__(self,network):
        self.net=network
        self.log=[]
        self.tune_count={} # Track tuning attempts per neuron

    def heal_neuron(self, neuron_id, fault_type):
        neuron = self.net.get_neuron(neuron_id)
        action = "No action taken"
        
        # Normalize fault type to upper case to match FaultType constants
        fault_type = str(fault_type).upper()

        if fault_type == FaultType.DEAD:
            # Strategy 1: Replace with backup
            backup = self.net.get_available_backup()
            
            if backup:
                self._activate_backup(original_id=neuron_id, backup_neuron=backup)
                action = f"Replaced degraded neuron {neuron_id} with backup {backup.id}"
            
            else:
                # Strategy 4: Redistribution
                self._redistribute_weights(neuron_id)
                action = f"No backups. Redistributed weights of {neuron_id} to neighbors"
             
        elif fault_type == FaultType.SILENT:
            # Circuit Breaker: Don't tune forever
            if self.tune_count.get(neuron_id, 0) > 5:
                action = f"Max tuning reached for {neuron_id}. Ignoring."
            else:
                self.tune_count[neuron_id] = self.tune_count.get(neuron_id, 0) + 1
                
                # Strategy 2: Weight Boost
                self._adjust_weights(neuron_id, factor=1.5)
                # Strategy 3: Threshold Adjust (Lower it)
                neuron.threshold *= 0.8
                action = f"Boosted weights (1.5x) and lowered threshold for {neuron_id} (Attempt {self.tune_count[neuron_id]})"
                
        elif fault_type == FaultType.HYPERACTIVE:
            # Strategy 3: Threshold Adjust (Raise it)
            neuron.threshold *= 1.5
            action = f"Raised threshold for hyperactive neuron {neuron.id}"

        # Only reset health for active neurons (Silent/Hyperactive), not for Dead ones we replaced
        if fault_type != FaultType.DEAD:
            self.net.health_monitor.reset_health(neuron_id)

        # Clear fault history to prevent immediate re-detection
        self.net.fault_detector.clear_history(neuron_id)
        
        self.log.append(action)
        return action

    def _activate_backup(self, original_id, backup_neuron):
        # 1. Copy weights from original to backup + slight boost (20%) to prevent immediate silence
        # Assuming input_synapses.weights is (n_in, n_hidden+n_backup)
        self.net.input_synapses.weights[:, backup_neuron.id] = self.net.input_synapses.weights[:, original_id] * 1.2
        
        # 2. Enable backup
        backup_neuron.is_active = True
        # Copy other properties if needed, e.g. threshold
        backup_neuron.threshold = self.net.get_neuron(original_id).threshold

        # 3. Disable original
        self.net.get_neuron(original_id).is_active = False

        # 4. Update health monitor
        self.net.health_monitor.reset_health(backup_neuron.id)
        
        # Update active neuron list in network
        if original_id in self.net.active_neuron_ids:
            self.net.active_neuron_ids.remove(original_id)
        if backup_neuron.id not in self.net.active_neuron_ids:
            self.net.active_neuron_ids.append(backup_neuron.id)

    def _redistribute_weights(self, neuron_id):
        # Distribute dead neuron's weights to neighbors (other active neurons)
        dead_weights = self.net.input_synapses.weights[:, neuron_id]
        redistribution_factor = 0.1 # Adds 10% of dead neuron's weight
        
        for nid in self.net.active_neuron_ids:
            if nid == neuron_id: continue
            
            # Add fraction of dead weights to neighbor
            self.net.input_synapses.weights[:, nid] += dead_weights * redistribution_factor
            
            # Increment Scar Tissue count
            self.net.redistribution_counts[nid] = self.net.redistribution_counts.get(nid, 0) + 1
        
        # Clip weights to stay in valid range
        numpy.clip(self.net.input_synapses.weights, 0.0, 1.0, out=self.net.input_synapses.weights)

        # Remove the dead neuron from active list to stop monitoring it
        if neuron_id in self.net.active_neuron_ids:
            self.net.active_neuron_ids.remove(neuron_id)

    def _adjust_weights(self, neuron_id, factor):
        # Scale weights for a specific neuron
        # Add a tiny bit of randomness to prevent "resonance loops" where it tunes -> fails -> tunes -> fails forever
        jitter = numpy.random.uniform(0.9, 1.1)
        final_factor = factor * jitter
        
        self.net.input_synapses.weights[:, neuron_id] *= final_factor
        
        # Clip weights
        numpy.clip(self.net.input_synapses.weights[:, neuron_id], 0.0, 1.0, out=self.net.input_synapses.weights[:, neuron_id])
