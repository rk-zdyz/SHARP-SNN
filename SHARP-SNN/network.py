import numpy as np
from lif_neuron import LIFNeuron
from synapse import SynapseLayer
from spike_encoder import SpikeEncoder
from fault_detector import FaultDetector, FaultType
from health_monitor import HealthMonitor
from recovery_engine import RecoveryEngine

class SharpSNN:
    def __init__(self, n_in, n_hidden, n_out, n_backup=2):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_backup = n_backup

        # Layers
        self.neurons = [LIFNeuron(i) for i in range(n_hidden+n_backup)]

        # Initialize Backups as inactive
        for i in range(n_hidden, n_hidden+n_backup):
            self.neurons[i].is_active= False
        self.active_neuron_ids = list(range(n_hidden))

        # Synapses
        self.input_synapses = SynapseLayer(n_in, n_hidden+n_backup)

        # Components
        self.encoder = SpikeEncoder()
        self.fault_detector = FaultDetector()
        self.health_monitor = HealthMonitor(range(n_hidden+n_backup))
        self.recovery_engine = RecoveryEngine(self)
        self.redistribution_counts = {} # Track "scar tissue"

    def get_neuron(self, nid):
        return self.neurons[nid]

    def get_available_backup(self):
        # Only check backup neurons (indices from n_hidden onwards)
        for i in range(self.n_hidden, self.n_hidden + self.n_backup):
            n = self.neurons[i]
            if not n.is_active:
                return n
        return None

    def forward(self, input_data, time_steps = 50, learn=True):
        spikes = self.encoder.encode(input_data)
        output_spikes=[]

        for t in range(time_steps):
            # 1. Input -> Hidden
            in_spike=spikes[t]
            hidden_input= self.input_synapses.forward(in_spike)

            current_hidden_spikes = np.zeros(len(self.neurons))

            # 2. Hidden Layer Update
            for nid in self.active_neuron_ids:
                neuron=self.neurons[nid]
                spike=neuron.step(hidden_input[nid],t)
                current_hidden_spikes[nid]=spike
                
                # 3. Fault Monitoring
                self.fault_detector.record_spike(nid,spike)

            # 4. Learning (STDP)
            if learn: 
                self.input_synapses.update_stdp(in_spike, current_hidden_spikes,t)

            # 5. Check Health and Heal
            self._check_and_heal() 
            
            output_spikes.append(current_hidden_spikes)
            
            # Update Energy Metric (Total spikes in this step)
            self.current_energy = np.sum(current_hidden_spikes)
            
        return np.array(output_spikes)

    def _check_and_heal(self):
        for nid in self.active_neuron_ids:

            # 1. Detect
            fault = self.fault_detector.detect_fault(self.neurons[nid])

            # 2. Monitor
            self.health_monitor.update_health(nid, fault)

            # 3. Heal 
            # 3. Heal 
            if self.health_monitor.is_healing(nid):
                # Continue healing
                done = self.health_monitor.tick_healing(nid)
                if done:
                    # We need to determine fault type again for the fix
                    fault = self.fault_detector.detect_fault(self.neurons[nid])
                    if not self.neurons[nid].is_active: fault = "Dead" 
                    
                    action = self.recovery_engine.heal_neuron(nid, fault)
                    self.health_monitor.complete_healing(nid, fault)
                    
                    # Log differently based on severity
                    if "Replaced" in action or "Redistributed" in action:
                        print(f" [✔] CRITICAL RECOVERY COMPLETE: {action}")
                    elif "Max tuning reached" in action:
                         pass # SILENCE! Stop spamming the user.
                    else:
                        print(f" [i] Auto-Tuned Neuron {nid}: {action}")
            
            elif self.health_monitor.needs_healing(nid):
                # Only announce critical failures loudly
                if fault == "DEAD" or fault == "Dead":
                    print(f" [!] CRITICAL FAILURE DETECTED: Neuron {nid} is DEAD. Initiating Recovery Protocol...")
                # For maintenance tasks (Silent/Hyperactive), start silently to avoid spam
                self.health_monitor.start_healing(nid)

    def get_state(self):
        """Returns the current state of the network for visualization."""
        neuron_states = []
        for i, neuron in enumerate(self.neurons):
            
            healing_progress = self.health_monitor.healing_progress.get(neuron.id, 0.0)
            is_healing = self.health_monitor.is_healing(neuron.id)
            scar_tissue = self.redistribution_counts.get(neuron.id, 0)

            state = {
                "id": neuron.id,
                "potential": neuron.potential,
                "threshold": neuron.threshold,
                "is_active": neuron.is_active,
                "is_backup": i >= self.n_hidden, # Simple check based on index
                "fault": "Healthy",
                "health": self.health_monitor.health_scores.get(neuron.id, 1.0),
                "healing_progress": healing_progress,
                "is_healing": is_healing,
                "scar_tissue": scar_tissue
            }
            # Check fault status from health monitor
            if not neuron.is_active:
                state["fault"] = "Dead"
            elif is_healing:
                state["fault"] = "Healing..."
            elif self.health_monitor.health_scores.get(neuron.id, 1.0) < 1.0:
                state["fault"] = "Degraded"
            
            neuron_states.append(state)

        # Synapse weights (simplified for now, sending full matrix might be heavy)
        # We can send just significant weights or the whole thing if small.
        # Let's send non-zero weights or > 0.1
        synapses = []
        rows, cols = self.input_synapses.weights.shape
        for r in range(rows):
            for c in range(cols):
                w = self.input_synapses.weights[r, c]
                if w > 0.05:
                    synapses.append({"source": f"in_{r}", "target": c, "weight": w})

        return {
            "neurons": neuron_states,
            "synapses": synapses,
            "active_ids": self.active_neuron_ids,
            "logs": self.recovery_engine.log[-5:] if self.recovery_engine.log else [],
            "energy": self.current_energy if hasattr(self, 'current_energy') else 0
        }