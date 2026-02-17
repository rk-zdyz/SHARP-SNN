
import numpy as np
from network import SharpSNN
from fault_detector import FaultType

def run_demo():
    print("=== SHARP-SNN Self-Healing Demo ===")
    
    # 1. Initialize Network
    try:
        n_in = int(input("Enter number of Input Neurons (default 10): ") or 10)
        n_hidden = int(input("Enter number of Hidden Neurons (default 5): ") or 5)
        n_out = int(input("Enter number of Output Neurons (default 2): ") or 2)
        n_backup = int(input("Enter number of Backup Neurons (default 2): ") or 2)
    except ValueError:
        print("Invalid input! Using defaults: 10, 5, 2, 2")
        n_in, n_hidden, n_out, n_backup = 10, 5, 2, 2
    
    print(f"Initializing Network: Inputs={n_in}, Hidden={n_hidden} (+{n_backup} backups)")
    snn = SharpSNN(n_in, n_hidden, n_out, n_backup)
    
    # 2. Generate Random Input Data
    print("\nGenerating random input data...")
    # n_in input channels, random values 0-1
    input_data = np.random.rand(n_in) * 0.8
    
    # 3. Normal Operation
    print("\nStep 1: Normal Operation (100 steps)")
    snn.forward(input_data, time_steps=20, learn=True)
    print("Network running normally. Checking Active Neurons:")
    print(f"Active Neurons: {snn.active_neuron_ids}")
    
    # 4. Inject Fault (Dead Neuron)
    if not snn.active_neuron_ids:
        print("Error: No active neurons to inject fault into!")
        return

    target_neuron = snn.active_neuron_ids[0]
    print(f"\nStep 2: Injecting DEAD fault into Neuron {target_neuron}")
    snn.neurons[target_neuron].inject_fault("Dead") # Using "Dead" as per LIFNeuron implementation
    
    # 5. Run to Trigger Healing
    print("Running network to detect and heal...")
    # Running multiple steps to allow fault detector to accumulate history and detect
    for i in range(5):
        print(f"  Batch {i+1}...")
        snn.forward(input_data, time_steps=20, learn=False)
        
        # Check if the target is still active
        if target_neuron not in snn.active_neuron_ids:
            print(f"  -> Neuron {target_neuron} is no longer active!")
            break

    # 6. Verify Outcome
    print("\n=== Post-Healing Status ===")
    print(f"Active Neurons: {snn.active_neuron_ids}")
    print(f"Recovery Log: {snn.recovery_engine.log}")
    
    # Check if a backup was activated
    # Backups start after hidden neurons
    backups = list(range(n_hidden, n_hidden + n_backup))
    activated_backups = [nid for nid in snn.active_neuron_ids if nid in backups]
    
    if activated_backups:
        print(f"SUCCESS: Backup Neuron(s) {activated_backups} activated!")
    else:
        print("Note: No backup activated (maybe redistribution strategy used or detection lag).")

if __name__ == "__main__":
    run_demo()
