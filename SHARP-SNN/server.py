
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import time
import threading
from network import SharpSNN

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sharp_snn_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize SNN
n_in = 10
n_hidden = 5
n_out = 2
n_backup = 2
snn = SharpSNN(n_in, n_hidden, n_out, n_backup)

simulation_running = False
simulation_speed = 0.1 # seconds per step

def simulation_thread():
    global simulation_running
    print("Simulation started")
    step = 0
    while True:
        if simulation_running:
            # Generate random input
            input_data = np.random.rand(n_in) * 1.0
            
            # Run one step (using small time_steps=1 for real-time feel)
            # Note: The original forward runs for 'time_steps'. 
            # Ideally we'd modify forward to run 1 step or just call internal update 
            # But let's just run short bursts
            snn.forward(input_data, time_steps=5, learn=True)
            
            # Emit state
            state = snn.get_state()
            socketio.emit('snn_update', state)
            
            step += 1
            
        time.sleep(simulation_speed)

# Start background thread
thread = threading.Thread(target=simulation_thread)
thread.daemon = True
thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('snn_update', snn.get_state())

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('toggle_simulation')
def handle_toggle(data):
    global simulation_running
    simulation_running = data['running']
    print(f"Simulation {'started' if simulation_running else 'stopped'}")

@socketio.on('inject_fault')
def handle_fault(data):
    neuron_id = data['id']
    fault_type = data['type'] # "Dead", "Silent", "Hyperactive"
    print(f"Injecting {fault_type} fault into Neuron {neuron_id}")
    
    # Inject fault
    if 0 <= neuron_id < len(snn.neurons):
        snn.neurons[neuron_id].inject_fault(fault_type)
        
        # Force immediate health update so UI reflects it even if paused
        detected_fault = snn.fault_detector.detect_fault(snn.neurons[neuron_id])
        snn.health_monitor.update_health(neuron_id, detected_fault)
        
        socketio.emit('log_message', {'msg': f"Injected {fault_type} fault into Neuron {neuron_id}"})
        socketio.emit('snn_update', snn.get_state())

@socketio.on('reset_network')
def handle_reset():
    global snn
    snn = SharpSNN(n_in, n_hidden, n_out, n_backup)
    socketio.emit('log_message', {'msg': "Network Reset"})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
